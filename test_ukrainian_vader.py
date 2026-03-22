from __future__ import annotations

import importlib.util
import sys
import unittest
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


class TestUkrainianVader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyzer = UkrainianSentimentIntensityAnalyzer()

    def test_positive_text(self) -> None:
        score = self.analyzer.polarity_scores("Щастя, любов і мир.")
        self.assertGreater(score["compound"], 0.2)

    def test_negative_text(self) -> None:
        score = self.analyzer.polarity_scores("Зрада, злочин і насильство.")
        self.assertLess(score["compound"], -0.2)

    def test_negation_flip(self) -> None:
        plain = self.analyzer.polarity_scores("Це поганий результат.")
        negated = self.analyzer.polarity_scores("Це не поганий результат.")
        self.assertLess(plain["compound"], 0)
        self.assertGreater(negated["compound"], 0)

    def test_but_contrast(self) -> None:
        score = self.analyzer.polarity_scores("Краса є, але поганий фінал.")
        self.assertLess(score["compound"], 0)

    def test_punctuation_emphasis(self) -> None:
        base = self.analyzer.polarity_scores("Радість")
        emph = self.analyzer.polarity_scores("Радість!!!")
        self.assertGreater(emph["compound"], base["compound"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
