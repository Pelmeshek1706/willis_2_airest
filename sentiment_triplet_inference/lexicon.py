from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class LexiconStats:
    entries: int
    phrase_entries: int
    max_ngram: int


def _to_float(raw: str) -> float | None:
    value = (raw or "").strip().replace(",", ".")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def load_tonsum_lexicon(path: str | Path) -> Dict[str, float]:
    """
    Load tonSUM 1.0 TSV into a dictionary:
      key: normalized word/phrase (lowercase, trimmed)
      value: average sentiment score in [-2, 2]
    """
    lexicon: Dict[str, float] = {}
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8", errors="replace") as source:
        for row_index, line in enumerate(source):
            raw = line.rstrip("\n")
            if not raw:
                continue

            columns = raw.split("\t")
            if row_index == 0 and columns and "Word//word combination" in columns[0]:
                continue

            if not columns:
                continue

            token = columns[0].strip().lower()
            if not token:
                continue

            ratings = [_to_float(value) for value in columns[1:9]]
            ratings = [value for value in ratings if value is not None]

            avg_col = _to_float(columns[9]) if len(columns) > 9 else None
            score = avg_col if avg_col is not None else _mean(ratings)
            if score is None:
                continue

            if token in lexicon:
                lexicon[token] = (lexicon[token] + score) / 2.0
            else:
                lexicon[token] = score

    return lexicon


def summarize_lexicon(lexicon: Dict[str, float]) -> LexiconStats:
    phrase_entries = 0
    max_ngram = 1
    for key in lexicon:
        ngram = len(key.split())
        if ngram > 1:
            phrase_entries += 1
        if ngram > max_ngram:
            max_ngram = ngram

    return LexiconStats(entries=len(lexicon), phrase_entries=phrase_entries, max_ngram=max_ngram)
