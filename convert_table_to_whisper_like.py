#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV/DataFrame -> whisper-like JSON per File_number.

Why this script:
- If you do not have real word timestamps, each source row can be split into word tokens.
- Token start/end are synthesized inside row [Start_Time, End_Time]:
  - even: equal durations
  - length: proportional to token length (default)

Output schema is whisper-like and compatible with code that expects:
{
  "text": "...",
  "language": "uk",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 12.34,
      "text": "...",
      "words": [{"word": "...", "start": 0.0, "end": 0.8, "probability": 0.93}],
      ...
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):  # type: ignore
        return x


def _normalize_conf(value: Any) -> float:
    """Normalize confidence to [0, 1]. Supports percents in [0, 100]."""
    try:
        x = float(value)
    except Exception:
        return 1.0
    if x > 1.5:
        x = x / 100.0
    return min(1.0, max(0.0, x))


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not (v == v):  # NaN
            return None
        return v
    except Exception:
        return None


def _sanitize_text(s: Any) -> str:
    if s is None:
        return ""
    if isinstance(s, float) and s != s:
        return ""
    return " ".join(str(s).split())


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")


_TOKEN_RE = re.compile(r"\S+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def _token_weight(token: str) -> int:
    # Prefer alnum length, fallback to raw stripped length.
    core = re.sub(r"[^\w\u0400-\u04FF]+", "", token, flags=re.UNICODE)
    w = len(core) if core else len(token.strip())
    return max(1, w)


def build_word_timestamps(
    text: str,
    start: float,
    end: float,
    probability: float,
    strategy: str = "length",
) -> List[Dict[str, Any]]:
    """
    Split one row text into token-level 'words' with synthesized timestamps.
    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    duration = max(1e-6, end - start)
    if len(tokens) == 1:
        return [{"word": tokens[0], "start": float(start), "end": float(end), "probability": float(probability)}]

    if strategy not in {"length", "even"}:
        raise ValueError("strategy must be one of {'length','even'}")

    if strategy == "even":
        weights = [1] * len(tokens)
    else:
        weights = [_token_weight(tok) for tok in tokens]

    total = float(sum(weights))
    out: List[Dict[str, Any]] = []
    cur = float(start)

    for i, tok in enumerate(tokens):
        if i == len(tokens) - 1:
            nxt = float(end)
        else:
            frac = weights[i] / total
            nxt = cur + duration * frac
            nxt = min(float(end), max(cur, nxt))

        out.append(
            {
                "word": tok,
                "start": float(cur),
                "end": float(nxt),
                "probability": float(probability),
            }
        )
        cur = nxt

    return out


_PUNCT_NO_SPACE_BEFORE = r",\.\?\!:\;…%\)\]\»”"
_PUNCT_NO_SPACE_AFTER = r"\(\[\«“"
_RX_MULTI_SPACE = re.compile(r"\s+")
_RX_SPACE_BEFORE = re.compile(rf"\s+([{_PUNCT_NO_SPACE_BEFORE}])")
_RX_SPACE_AFTER = re.compile(rf"([{_PUNCT_NO_SPACE_AFTER}])\s+")


def join_words_punct_aware(words: List[str]) -> str:
    if not words:
        return ""
    s = " ".join(w.strip() for w in words if w and str(w).strip())
    s = _RX_SPACE_BEFORE.sub(r"\1", s)
    s = _RX_SPACE_AFTER.sub(r"\1 ", s)
    s = _RX_MULTI_SPACE.sub(" ", s).strip()
    return s


def _is_terminal_token(token: str) -> bool:
    return token.endswith((".", "?", "!", "…"))


def convert_table_to_whisper_like(
    df: pd.DataFrame,
    out_dir: str | Path = "woz_ukr_whisper",
    *,
    group_col: str = "File_number",
    start_col: str = "Start_Time",
    end_col: str = "End_Time",
    text_col_primary: str = "Text_ukr",
    text_col_fallback: str = "Text",
    conf_col: str = "Confidence",
    language: str = "uk",
    gap_threshold: float = 0.6,
    punct_gap_threshold: float = 0.2,
    max_segment_dur: float = 30.0,
    word_timing_strategy: str = "length",  # one of: length/even
    clip_end_to_next_start: bool = True,
    min_word_duration: float = 1e-3,
    drop_suspicious_overlaps: bool = True,
    suspicious_min_original_duration: float = 120.0,
    suspicious_max_clipped_duration: float = 10.0,
    suspicious_min_ratio: float = 20.0,
    indent: Optional[int] = 2,
) -> None:
    required = [group_col, start_col, end_col, conf_col]
    text_present = False
    if text_col_primary in df.columns:
        required.append(text_col_primary)
        text_present = True
    if text_col_fallback in df.columns:
        required.append(text_col_fallback)
        text_present = True
    if not text_present:
        raise ValueError(
            f"Need at least one text column: '{text_col_primary}' or '{text_col_fallback}'"
        )
    _ensure_columns(df, required)

    work = df.copy()
    work[start_col] = work[start_col].apply(_to_float)
    work[end_col] = work[end_col].apply(_to_float)
    work[conf_col] = work[conf_col].apply(_normalize_conf)

    work = work.dropna(subset=[start_col, end_col]).reset_index(drop=True)
    work = work[work[end_col] >= work[start_col]].reset_index(drop=True)

    def _pick_text(row: pd.Series) -> str:
        primary = _sanitize_text(row.get(text_col_primary)) if text_col_primary in row else ""
        if primary:
            return primary
        fallback = _sanitize_text(row.get(text_col_fallback)) if text_col_fallback in row else ""
        return fallback

    work["__text_final__"] = work.apply(_pick_text, axis=1)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for file_num, group in tqdm(work.groupby(group_col), desc="Files"):
        group = group.sort_values(by=[start_col, end_col]).reset_index(drop=True)

        # Optional protection for noisy tables where one row spans most of the file and overlaps many rows.
        # For each row, clip end time to the next strictly greater start time (if any).
        if clip_end_to_next_start and len(group) > 1:
            starts = group[start_col].astype(float).tolist()
            ends = group[end_col].astype(float).tolist()
            clipped_ends = ends[:]
            keep_mask = [True] * len(starts)
            n = len(starts)
            for i in range(n):
                cur_start = starts[i]
                next_strict_start = None
                for j in range(i + 1, n):
                    if starts[j] > cur_start:
                        next_strict_start = starts[j]
                        break
                if next_strict_start is None:
                    continue
                if clipped_ends[i] > next_strict_start:
                    original_dur = max(0.0, ends[i] - cur_start)
                    clipped_dur = max(min_word_duration, next_strict_start - cur_start)
                    ratio = original_dur / clipped_dur if clipped_dur > 0 else float("inf")
                    is_suspicious = (
                        drop_suspicious_overlaps
                        and original_dur >= suspicious_min_original_duration
                        and clipped_dur <= suspicious_max_clipped_duration
                        and ratio >= suspicious_min_ratio
                    )
                    if is_suspicious:
                        keep_mask[i] = False
                    else:
                        clipped_ends[i] = max(cur_start + min_word_duration, next_strict_start)

            group = group.loc[keep_mask].reset_index(drop=True)
            group[end_col] = [e for e, keep in zip(clipped_ends, keep_mask) if keep]

        segments: List[Dict[str, Any]] = []
        seg_id = 0
        current_words: List[Dict[str, Any]] = []
        seg_start: Optional[float] = None
        seg_end: Optional[float] = None
        last_end: Optional[float] = None
        full_tokens: List[str] = []

        def flush_segment() -> None:
            nonlocal seg_id, current_words, seg_start, seg_end
            if not current_words:
                return
            seg_text = join_words_punct_aware([w["word"] for w in current_words])

            avg_logprob = (
                sum(math.log(max(1e-12, float(w.get("probability", 1e-6)))) for w in current_words)
                / len(current_words)
            )

            segments.append(
                {
                    "id": seg_id,
                    "seek": 0,
                    "start": float(seg_start if seg_start is not None else 0.0),
                    "end": float(seg_end if seg_end is not None else 0.0),
                    "text": seg_text,
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": float(avg_logprob),
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                    "words": current_words,
                }
            )
            seg_id += 1
            current_words = []
            seg_start = None
            seg_end = None

        for _, row in group.iterrows():
            row_start = float(row[start_col])
            row_end = float(row[end_col])
            prob = float(row[conf_col])
            text = _sanitize_text(row["__text_final__"])
            if not text:
                continue

            row_words = build_word_timestamps(
                text=text,
                start=row_start,
                end=row_end,
                probability=prob,
                strategy=word_timing_strategy,
            )
            if not row_words:
                continue

            full_tokens.extend([w["word"] for w in row_words])

            for w in row_words:
                w_start = float(w["start"])
                w_end = float(w["end"])

                start_new = False
                if last_end is None or (w_start - last_end) > gap_threshold:
                    start_new = True
                elif current_words and _is_terminal_token(current_words[-1]["word"]):
                    if (w_start - last_end) >= punct_gap_threshold:
                        start_new = True
                if not start_new and seg_start is not None and (w_end - seg_start) > max_segment_dur:
                    start_new = True

                if start_new:
                    flush_segment()
                    seg_start = w_start

                current_words.append(w)
                seg_end = w_end
                last_end = w_end

        flush_segment()

        full_text = join_words_punct_aware(full_tokens)
        transcript: Dict[str, Any] = {
            "text": full_text,
            "segments": segments,
            "language": language,
        }

        out_file = out_path / f"{str(file_num)}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=indent)

    print(f"Done. Files saved in: {out_path.resolve()}")


def _read_input_table(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path, encoding="utf-8")
    if suffix == ".tsv":
        return pd.read_csv(input_path, encoding="utf-8", delimiter="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    return pd.read_csv(input_path, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Table -> whisper-like JSON with synthetic word timestamps.")
    ap.add_argument("-i", "--input", required=True, help="Path to CSV/TSV/XLSX")
    ap.add_argument("-o", "--out_dir", default="woz_ukr_whisper", help="Output dir")
    ap.add_argument("--group_col", default="File_number")
    ap.add_argument("--start_col", default="Start_Time")
    ap.add_argument("--end_col", default="End_Time")
    ap.add_argument("--text_col_primary", default="Text_ukr")
    ap.add_argument("--text_col_fallback", default="Text")
    ap.add_argument("--conf_col", default="Confidence")
    ap.add_argument("--language", default="uk")

    ap.add_argument("--gap_threshold", type=float, default=0.6)
    ap.add_argument("--punct_gap_threshold", type=float, default=0.2)
    ap.add_argument("--max_segment_dur", type=float, default=30.0)
    ap.add_argument(
        "--word_timing_strategy",
        choices=["length", "even"],
        default="length",
        help="How to distribute row duration among tokenized words.",
    )
    ap.add_argument(
        "--clip_end_to_next_start",
        action="store_true",
        default=True,
        help="Clip row end to the next strictly greater start (default: on).",
    )
    ap.add_argument(
        "--no_clip_end_to_next_start",
        action="store_false",
        dest="clip_end_to_next_start",
        help="Disable end-time clipping against next start.",
    )
    ap.add_argument(
        "--drop_suspicious_overlaps",
        action="store_true",
        default=True,
        help="Drop anomalous long rows that overlap many following rows (default: on).",
    )
    ap.add_argument(
        "--keep_suspicious_overlaps",
        action="store_false",
        dest="drop_suspicious_overlaps",
        help="Keep suspicious overlapping rows (not recommended).",
    )
    ap.add_argument("--indent", type=int, default=2)
    args = ap.parse_args()

    df = _read_input_table(Path(args.input))
    convert_table_to_whisper_like(
        df,
        out_dir=args.out_dir,
        group_col=args.group_col,
        start_col=args.start_col,
        end_col=args.end_col,
        text_col_primary=args.text_col_primary,
        text_col_fallback=args.text_col_fallback,
        conf_col=args.conf_col,
        language=args.language,
        gap_threshold=args.gap_threshold,
        punct_gap_threshold=args.punct_gap_threshold,
        max_segment_dur=args.max_segment_dur,
        word_timing_strategy=args.word_timing_strategy,
        clip_end_to_next_start=args.clip_end_to_next_start,
        drop_suspicious_overlaps=args.drop_suspicious_overlaps,
        indent=args.indent,
    )


if __name__ == "__main__":
    main()

### Example 
# import pandas as pd

# df_eng = pd.read_csv("/Users/pelmeshek1706/Desktop/projects/airest_notebooks/data/dcapwoz_all_plus_ukr_gemma.csv")
# df_eng
# convert_table_to_whisper_like(
#     df_eng,
#     out_dir="woz_eng_whisper_traslated_gemma_wisper",
#     language="ua",
#     text_col_primary="Text",#"translated_text_gemma",
#     # text_col_fallback="Text",
#     word_timing_strategy="length",  # или "even"
# )