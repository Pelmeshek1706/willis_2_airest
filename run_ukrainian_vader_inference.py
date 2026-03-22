"""Quick inference demo for the Ukrainian VADER-like analyzer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "openwillis" / "openwillis-speech" / "src"
MODULE_PATH = SRC_ROOT / "openwillis" / "speech" / "util" / "speech" / "ukrainian_vader.py"
SPEC = importlib.util.spec_from_file_location("ukrainian_vader", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load analyzer module from: {MODULE_PATH}")
UK_VADER = importlib.util.module_from_spec(SPEC)
sys.modules["ukrainian_vader"] = UK_VADER
SPEC.loader.exec_module(UK_VADER)
UkrainianSentimentIntensityAnalyzer = UK_VADER.UkrainianSentimentIntensityAnalyzer


def main() -> None:
    analyzer = UkrainianSentimentIntensityAnalyzer()

    examples = [
        "Це щастя і радість.",
        "Це зрада і злочин.",
        "Це не поганий результат.",
        "Краса є, але фінал поганий.",
        "Мир і любов!!!",
        "Мені сумно 😢",
        "Супер концерт, дуже комфортно!",
        "Насильство і ненависть - це жахливо.",
    ]

    for text in examples:
        print(text)
        print(analyzer.polarity_scores(text))
        print("-" * 60)


if __name__ == "__main__":
    main()
