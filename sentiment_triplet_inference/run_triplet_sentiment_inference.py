#!/usr/bin/env python3
"""
Triplet sentiment inference over OpenWillis-like JSON transcripts.

Compares 3 analyzers:
1) classic English VADER (`vader`)
2) external `vader-ua` package (`vader-ua`)
3) improved in-project UA VADER-like analyzer (`vader-ua-improved`)

For each input JSON file <n>.json, writes:
- sentiment_<n>_vader.csv
- sentiment_<n>_vader-ua.csv
- sentiment_<n>_vader-ua-improved.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class Turn:
    turn_id: int
    text: str
    start: Optional[float]
    end: Optional[float]
    speaker: Optional[str]


def _norm_score_dict(scores: Dict[str, Any]) -> Dict[str, float]:
    """Normalize score keys to VADER-like output keys."""
    if not isinstance(scores, dict):
        return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

    if {"neg", "neu", "pos", "compound"}.issubset(scores.keys()):
        return {
            "neg": float(scores.get("neg", 0.0)),
            "neu": float(scores.get("neu", 0.0)),
            "pos": float(scores.get("pos", 0.0)),
            "compound": float(scores.get("compound", 0.0)),
        }

    # HF-like schema fallback
    return {
        "neg": float(scores.get("negative", 0.0)),
        "neu": float(scores.get("neutral", 0.0)),
        "pos": float(scores.get("positive", 0.0)),
        "compound": float(scores.get("compound", 0.0)),
    }


def _load_improved_analyzer(improved_module_path: Path):
    """Load in-project improved analyzer class from a file path."""
    spec = importlib.util.spec_from_file_location("ukrainian_vader_improved", improved_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load improved analyzer module from: {improved_module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["ukrainian_vader_improved"] = module
    spec.loader.exec_module(module)

    cls = getattr(module, "UkrainianSentimentIntensityAnalyzer", None)
    if cls is None:
        raise ImportError(
            f"UkrainianSentimentIntensityAnalyzer not found in module: {improved_module_path}"
        )
    return cls()


def _load_base_vader_ua(vader_ua_root: Path):
    """Load external vader-ua package analyzer (with relative imports)."""
    root = vader_ua_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"vader-ua root does not exist: {root}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from analyzer import UkrainianSentimentIntensityAnalyzer as BaseUAAnalyzer

    return BaseUAAnalyzer()


def _iter_input_files(inputs: Sequence[str], glob_pattern: str) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            files.extend(sorted(p.glob(glob_pattern)))
        elif p.is_file():
            files.append(p)

    # unique + stable order
    seen = set()
    unique_files: List[Path] = []
    for f in files:
        rf = f.resolve()
        if rf in seen:
            continue
        seen.add(rf)
        unique_files.append(rf)

    return unique_files


def _extract_turns_from_json(data: Dict[str, Any], speaker_label: Optional[str] = None) -> Tuple[List[Turn], str]:
    """
    OpenWillis-like turn extraction from Whisper-style json.
    - Prefers `segments[*].text` as turns.
    - Uses `text` as full text if available, else joins turns.
    """
    raw_segments = data.get("segments", []) if isinstance(data, dict) else []
    turns: List[Turn] = []

    if isinstance(raw_segments, list):
        for idx, seg in enumerate(raw_segments):
            if not isinstance(seg, dict):
                continue

            text = str(seg.get("text", "")).strip()
            if not text:
                continue

            speaker = seg.get("speaker") or seg.get("speaker_label") or seg.get("speaker_id")
            speaker = str(speaker) if speaker is not None else None

            if speaker_label and speaker and speaker != speaker_label:
                continue

            start = seg.get("start")
            end = seg.get("end")
            try:
                start = float(start) if start is not None else None
            except Exception:
                start = None
            try:
                end = float(end) if end is not None else None
            except Exception:
                end = None

            turns.append(
                Turn(
                    turn_id=len(turns),
                    text=text,
                    start=start,
                    end=end,
                    speaker=speaker,
                )
            )

    full_text = ""
    if isinstance(data, dict):
        full_text = str(data.get("text", "") or "").strip()

    if not full_text:
        full_text = " ".join(t.text for t in turns).strip()

    return turns, full_text


def _build_sentiment_df(
    analyzer_name: str,
    analyzer: Any,
    file_id: str,
    turns: List[Turn],
    full_text: str,
    json_path: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for turn in turns:
        raw = analyzer.polarity_scores(turn.text)
        score = _norm_score_dict(raw)
        rows.append(
            {
                "row_type": "turn",
                "file_id": file_id,
                "json_path": str(json_path),
                "analyzer": analyzer_name,
                "turn_id": turn.turn_id,
                "speaker": turn.speaker,
                "start": turn.start,
                "end": turn.end,
                "text": turn.text,
                "neg": score["neg"],
                "neu": score["neu"],
                "pos": score["pos"],
                "compound": score["compound"],
            }
        )

    raw_summary = analyzer.polarity_scores(full_text)
    summary = _norm_score_dict(raw_summary)
    rows.append(
        {
            "row_type": "summary",
            "file_id": file_id,
            "json_path": str(json_path),
            "analyzer": analyzer_name,
            "turn_id": None,
            "speaker": None,
            "start": None,
            "end": None,
            "text": full_text,
            "neg": summary["neg"],
            "neu": summary["neu"],
            "pos": summary["pos"],
            "compound": summary["compound"],
        }
    )

    return pd.DataFrame(rows)


def run_inference(
    input_paths: Sequence[str],
    output_dir: Path,
    vader_ua_root: Path,
    improved_module_path: Path,
    glob_pattern: str,
    speaker_label: Optional[str] = None,
) -> None:
    files = _iter_input_files(input_paths, glob_pattern=glob_pattern)
    if not files:
        raise FileNotFoundError("No input JSON files found from provided paths.")

    output_dir.mkdir(parents=True, exist_ok=True)

    classic_vader = SentimentIntensityAnalyzer()
    base_vader_ua = _load_base_vader_ua(vader_ua_root)
    improved_vader_ua = _load_improved_analyzer(improved_module_path)

    analyzers: List[Tuple[str, Any]] = [
        # ("vader", classic_vader),
        ("vader-ua", base_vader_ua),
        ("vader-ua-improved", improved_vader_ua),
    ]

    for idx, json_path in enumerate(files, start=1):
        file_id = json_path.stem
        print(f"[{idx}/{len(files)}] Processing: {json_path}")

        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            print(f"  - Failed to read JSON: {json_path}")
            traceback.print_exc()
            continue

        turns, full_text = _extract_turns_from_json(payload, speaker_label=speaker_label)
        if not turns and not full_text:
            print(f"  - Skipped (no text/segments): {json_path}")
            continue

        for analyzer_type, analyzer in analyzers:
            try:
                df = _build_sentiment_df(
                    analyzer_name=analyzer_type,
                    analyzer=analyzer,
                    file_id=file_id,
                    turns=turns,
                    full_text=full_text,
                    json_path=json_path,
                )
            except Exception:
                print(f"  - Analyzer failed ({analyzer_type}) on {json_path}")
                traceback.print_exc()
                continue

            out_file = output_dir / f"sentiment_{file_id}_{analyzer_type}.csv"
            df.to_csv(out_file, index=False)
            print(f"  - Saved: {out_file}")


def _parse_args() -> argparse.Namespace:
    default_project_root = Path(__file__).resolve().parents[1]
    default_improved = (
        default_project_root
        / "openwillis"
        / "openwillis-speech"
        / "src"
        / "openwillis"
        / "speech"
        / "util"
        / "speech"
        / "ukrainian_vader.py"
    )
    default_vader_ua_root = Path("/Users/pelmeshek1706/Downloads/Telegram Desktop/vader-ua")

    parser = argparse.ArgumentParser(
        description="Run sentiment inference with 3 analyzers (classic VADER, vader-ua, vader-ua-improved)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSON files and/or directories.",
    )
    parser.add_argument(
        "--glob",
        default="*.json",
        help="Glob pattern for directory inputs (default: *.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "outputs"),
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--vader-ua-root",
        default=str(default_vader_ua_root),
        help="Path to external vader-ua project root.",
    )
    parser.add_argument(
        "--improved-module-path",
        default=str(default_improved),
        help="Path to improved in-project ukrainian_vader.py.",
    )
    parser.add_argument(
        "--speaker-label",
        default=None,
        help="Optional speaker label filter (OpenWillis-like behavior).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference(
        input_paths=args.inputs,
        output_dir=Path(args.output_dir).expanduser(),
        vader_ua_root=Path(args.vader_ua_root).expanduser(),
        improved_module_path=Path(args.improved_module_path).expanduser(),
        glob_pattern=args.glob,
        speaker_label=args.speaker_label,
    )


if __name__ == "__main__":
    main()
