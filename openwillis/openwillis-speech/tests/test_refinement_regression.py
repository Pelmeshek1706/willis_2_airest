"""Regression coverage for the class-based speech refinement API."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import openwillis.speech as legacy_speech
from openwillis.speech.util.speech import coherence as legacy_coherence
from phonova import SpeechAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_CASES = [
    (
        "en",
        PROJECT_ROOT / "tmp" / "role_labeled_whisper_like_stub_batch_eng_26-03-2026" / "300.json",
    ),
    (
        "ua",
        PROJECT_ROOT / "tmp" / "role_labeled_whisper_like_stub_batch_ukr_26-03-2026" / "300.json",
    ),
]
SPEAKERS = ["participant", "interviewer"]
TURN_MODES = ["speaker", "segment"]


def _load_fixture(path: Path) -> dict:
    """Load one Whisper-like transcript fixture from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame values before parity assertions."""
    normalized = frame.reset_index(drop=True).copy()
    for column in normalized.columns:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        if numeric.notna().any() or normalized[column].isna().all():
            numeric = numeric.astype("float64")
            numeric.loc[numeric == 0] = 0.0
            normalized[column] = numeric
    return normalized


def _assert_frames_match(legacy_frame: pd.DataFrame, refined_frame: pd.DataFrame) -> None:
    """Assert full-frame parity with a small tolerance for float math."""
    assert list(legacy_frame.columns) == list(refined_frame.columns)
    expected = _normalize_frame(legacy_frame)
    actual = _normalize_frame(refined_frame)
    pd.testing.assert_frame_equal(
        expected,
        actual,
        check_dtype=False,
        check_exact=False,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.fixture(scope="module")
def gemma_analyzers() -> dict[str, SpeechAnalyzer]:
    """Create one reusable analyzer per language for the Gemma parity suite."""
    return {language: SpeechAnalyzer(language=language, coherence_backend="gemma") for language, _ in FIXTURE_CASES}


@pytest.mark.parametrize("language,fixture_path", FIXTURE_CASES, ids=["english", "ukrainian"])
@pytest.mark.parametrize("speaker_label", SPEAKERS)
@pytest.mark.parametrize("whisper_turn_mode", TURN_MODES)
def test_gemma_speech_analyzer_matches_legacy(
    gemma_analyzers: dict[str, SpeechAnalyzer],
    language: str,
    fixture_path: Path,
    speaker_label: str,
    whisper_turn_mode: str,
) -> None:
    """Ensure the new OOP entry point matches the legacy functional output."""
    transcript = _load_fixture(fixture_path)
    legacy_coherence.COHERENCE_BACKEND = "gemma"

    legacy_result = legacy_speech.speech_characteristics(
        json_conf=transcript,
        language=language,
        speaker_label=speaker_label,
        min_coherence_turn_length=2,
        option="coherence",
        whisper_turn_mode=whisper_turn_mode,
    )
    refined_result = gemma_analyzers[language].analyze_transcript(
        json_conf=transcript,
        speaker_label=speaker_label,
        min_coherence_turn_length=2,
        option="coherence",
        whisper_turn_mode=whisper_turn_mode,
    )

    for legacy_frame, refined_frame in zip(legacy_result, refined_result):
        _assert_frames_match(legacy_frame, refined_frame)


@pytest.mark.parametrize("language,fixture_path", FIXTURE_CASES, ids=["english", "ukrainian"])
def test_bert_speech_analyzer_smoke(language: str, fixture_path: Path) -> None:
    """Verify that the new analyzer can initialize and execute the BERT backend."""
    transcript = _load_fixture(fixture_path)
    analyzer = SpeechAnalyzer(language=language, coherence_backend="bert")
    words, turns, summary = analyzer.analyze_transcript(
        json_conf=transcript,
        speaker_label="participant",
        min_coherence_turn_length=2,
        option="coherence",
        whisper_turn_mode="speaker",
    )

    assert not words.empty
    assert not turns.empty
    assert not summary.empty
