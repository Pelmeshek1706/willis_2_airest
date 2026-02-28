#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Canonical CSV/DataFrame -> whisper-like JSON per file id.

Design:
- One input row (turn) becomes exactly one output segment.
- Segment end is preserved exactly from input row.
- Segment start can be pre-aligned to previous end to remove overlaps.
- No clipping, merging, splitting, or dropping rows.
- Word timestamps are synthesized evenly within each row span.
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
    def tqdm(items, **kwargs):  # type: ignore
        return items


_TOKEN_RE = re.compile(r"\S+", re.UNICODE)
_RX_MULTI_SPACE = re.compile(r"\s+")


def _collapse_ws(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return _RX_MULTI_SPACE.sub(" ", str(value)).strip()


def _to_finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalize_confidence(value: Any) -> float:
    number = _to_finite_float(value)
    if number is None:
        return 1.0
    if number > 1.5:
        number = number / 100.0
    return min(1.0, max(0.0, number))


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def _sanitize_file_stem(value: Any, default: str = "file") -> str:
    stem = _collapse_ws(value)
    if not stem:
        return default
    stem = stem.replace("/", "_").replace("\\", "_").replace("\x00", "")
    while ".." in stem:
        stem = stem.replace("..", "_")
    stem = re.sub(r"[^\w\.-]+", "_", stem, flags=re.UNICODE).strip(" ._")
    if not stem:
        return default
    return stem[:128]


def _reserve_unique_stem(stem: str, used: set[str]) -> str:
    if stem not in used:
        used.add(stem)
        return stem
    index = 2
    while f"{stem}_{index}" in used:
        index += 1
    unique_stem = f"{stem}_{index}"
    used.add(unique_stem)
    return unique_stem


def _build_even_word_timestamps(
    text: str,
    start: float,
    end: float,
    probability: float,
) -> List[Dict[str, Any]]:
    tokens = _tokenize(text)
    if not tokens:
        return []

    if len(tokens) == 1:
        return [{"word": tokens[0], "start": float(start), "end": float(end), "probability": float(probability)}]

    duration = end - start
    if duration <= 0:
        return [
            {"word": token, "start": float(start), "end": float(start), "probability": float(probability)}
            for token in tokens
        ]

    per_word = duration / len(tokens)
    words: List[Dict[str, Any]] = []
    current_start = float(start)
    for index, token in enumerate(tokens):
        if index == len(tokens) - 1:
            current_end = float(end)
        else:
            current_end = current_start + per_word
        words.append(
            {
                "word": token,
                "start": float(current_start),
                "end": float(current_end),
                "probability": float(probability),
            }
        )
        current_start = current_end
    return words


def _choose_text(row: pd.Series, primary_col: str, fallback_col: str) -> str:
    primary = _collapse_ws(row.get(primary_col)) if primary_col in row else ""
    if primary:
        return primary
    return _collapse_ws(row.get(fallback_col)) if fallback_col in row else ""


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")


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
    align_start_to_prev_end: bool = True,
    indent: Optional[int] = 2,
) -> None:
    required = [group_col, start_col, end_col]
    text_present = (text_col_primary in df.columns) or (text_col_fallback in df.columns)
    if not text_present:
        raise ValueError(
            f"Need at least one text column: '{text_col_primary}' or '{text_col_fallback}'"
        )
    _ensure_columns(df, required)

    work = df.copy().reset_index(drop=False).rename(columns={"index": "__row_index__"})
    work[start_col] = work[start_col].apply(_to_finite_float)
    work[end_col] = work[end_col].apply(_to_finite_float)
    if conf_col not in work.columns:
        work[conf_col] = 1.0
    work[conf_col] = work[conf_col].apply(_normalize_confidence)
    work["__text_final__"] = work.apply(
        lambda row: _choose_text(row, text_col_primary, text_col_fallback),
        axis=1,
    )

    invalid_times = work[work[start_col].isna() | work[end_col].isna()]
    if not invalid_times.empty:
        sample_rows = invalid_times["__row_index__"].head(10).tolist()
        raise ValueError(
            f"Found rows with non-finite {start_col}/{end_col}: {sample_rows}"
        )

    backwards = work[work[end_col] < work[start_col]]
    if not backwards.empty:
        sample_rows = backwards["__row_index__"].head(10).tolist()
        raise ValueError(
            f"Found rows where {end_col} < {start_col}: {sample_rows}"
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    used_file_stems: set[str] = set()
    start_adjustments_total = 0

    for file_number, group in tqdm(work.groupby(group_col, sort=False), desc="Files"):
        group = group.sort_values(by=["__row_index__"], kind="stable").reset_index(drop=True)
        if align_start_to_prev_end:
            adjusted_starts: List[float] = []
            prev_end: Optional[float] = None
            for segment_id, (_, row) in enumerate(group.iterrows()):
                row_start = float(row[start_col])
                row_end = float(row[end_col])
                adjusted_start = row_start
                if prev_end is not None and adjusted_start < prev_end:
                    adjusted_start = float(prev_end)
                    start_adjustments_total += 1
                if row_end < adjusted_start:
                    raise ValueError(
                        f"Adjusted row has end < start in file {file_number}, segment {segment_id}, "
                        f"row_index={int(row['__row_index__'])}, start={adjusted_start}, end={row_end}"
                    )
                adjusted_starts.append(adjusted_start)
                prev_end = row_end
            group = group.copy()
            group["__start_adjusted__"] = adjusted_starts
        else:
            group = group.copy()
            group["__start_adjusted__"] = group[start_col].astype(float)

        segments: List[Dict[str, Any]] = []
        segment_texts: List[str] = []

        for segment_id, (_, row) in enumerate(group.iterrows()):
            row_start = float(row["__start_adjusted__"])
            row_end = float(row[end_col])
            row_text = _collapse_ws(row["__text_final__"])
            row_prob = float(row[conf_col])

            words = _build_even_word_timestamps(
                text=row_text,
                start=row_start,
                end=row_end,
                probability=row_prob,
            )
            if words:
                avg_logprob = sum(math.log(max(1e-12, float(word["probability"]))) for word in words) / len(words)
            else:
                avg_logprob = math.log(max(1e-12, row_prob))

            segments.append(
                {
                    "id": segment_id,
                    "seek": 0,
                    "start": row_start,
                    "end": row_end,
                    "text": row_text,
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": float(avg_logprob),
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                    "words": words,
                }
            )
            if row_text:
                segment_texts.append(row_text)

        transcript: Dict[str, Any] = {
            "text": _collapse_ws(" ".join(segment_texts)),
            "segments": segments,
            "language": language,
        }

        safe_stem = _sanitize_file_stem(file_number, default="file")
        safe_stem = _reserve_unique_stem(safe_stem, used_file_stems)
        out_file = out_path / f"{safe_stem}.json"
        with open(out_file, "w", encoding="utf-8") as output_file:
            json.dump(transcript, output_file, ensure_ascii=False, indent=indent)

    if align_start_to_prev_end:
        print(
            f"Done. Files saved in: {out_path.resolve()} | "
            f"start-adjusted rows: {start_adjustments_total}"
        )
    else:
        print(f"Done. Files saved in: {out_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical table -> whisper-like JSON (1 row = 1 segment)."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to CSV/TSV/XLSX")
    parser.add_argument("-o", "--out_dir", default="woz_ukr_whisper", help="Output dir")
    parser.add_argument("--group_col", default="File_number")
    parser.add_argument("--start_col", default="Start_Time")
    parser.add_argument("--end_col", default="End_Time")
    parser.add_argument("--text_col_primary", default="Text_ukr")
    parser.add_argument("--text_col_fallback", default="Text")
    parser.add_argument("--conf_col", default="Confidence")
    parser.add_argument("--language", default="uk")
    parser.add_argument(
        "--align_start_to_prev_end",
        action="store_true",
        default=True,
        help="If row start is before previous row end, set start to previous end (default: on).",
    )
    parser.add_argument(
        "--no_align_start_to_prev_end",
        action="store_false",
        dest="align_start_to_prev_end",
        help="Disable start alignment to previous end.",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    dataframe = _read_input_table(Path(args.input))
    convert_table_to_whisper_like(
        dataframe,
        out_dir=args.out_dir,
        group_col=args.group_col,
        start_col=args.start_col,
        end_col=args.end_col,
        text_col_primary=args.text_col_primary,
        text_col_fallback=args.text_col_fallback,
        conf_col=args.conf_col,
        language=args.language,
        align_start_to_prev_end=args.align_start_to_prev_end,
        indent=args.indent,
    )


if __name__ == "__main__":
    main()
