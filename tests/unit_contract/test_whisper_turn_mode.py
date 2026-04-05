import pytest

from tests.helpers.module_loaders import load_local_characteristics_util, load_local_speech_attribute


def _measures():
    speech_attribute = load_local_speech_attribute()
    return speech_attribute.get_config(speech_attribute.__file__, "text.json")


def _segments_with_speakers():
    return [
        {
            "speaker": "participant",
            "text": "hello there",
            "words": [
                {"start": 0.0, "end": 0.2, "word": "hello"},
                {"start": 0.2, "end": 0.4, "word": "there"},
            ],
        },
        {
            "speaker": "participant",
            "text": "again friend",
            "words": [
                {"start": 0.5, "end": 0.7, "word": "again"},
                {"start": 0.7, "end": 0.9, "word": "friend"},
            ],
        },
        {
            "speaker": "interviewer",
            "text": "question now",
            "words": [
                {"start": 1.0, "end": 1.2, "word": "question"},
                {"start": 1.2, "end": 1.4, "word": "now"},
            ],
        },
    ]


def _segments_without_speakers():
    return [
        {
            "text": "hello there",
            "words": [
                {"start": 0.0, "end": 0.2, "word": "hello"},
                {"start": 0.2, "end": 0.4, "word": "there"},
            ],
        },
        {
            "text": "question now",
            "words": [
                {"start": 0.5, "end": 0.7, "word": "question"},
                {"start": 0.7, "end": 0.9, "word": "now"},
            ],
        },
    ]


def test_normalize_whisper_turn_mode_none_defaults_to_auto():
    cutil = load_local_characteristics_util()
    assert cutil.normalize_whisper_turn_mode(None) == "auto"


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("auto", "auto"),
        (" Auto ", "auto"),
        ("speaker", "speaker"),
        (" SPEAKER ", "speaker"),
        ("segment", "segment"),
        (" Segment ", "segment"),
    ],
)
def test_normalize_whisper_turn_mode_accepts_valid_values(raw_value, expected):
    cutil = load_local_characteristics_util()
    assert cutil.normalize_whisper_turn_mode(raw_value) == expected


@pytest.mark.parametrize("raw_value", ["", "merge", "segments", "speaker_turns"])
def test_normalize_whisper_turn_mode_rejects_invalid_values(raw_value):
    cutil = load_local_characteristics_util()
    with pytest.raises(ValueError):
        cutil.normalize_whisper_turn_mode(raw_value)


def test_auto_behaves_like_speaker_when_diarization_labels_exist():
    cutil = load_local_characteristics_util()
    measures = _measures()
    item_data = cutil.create_index_column(_segments_with_speakers(), measures)

    auto_turns = cutil.create_turns_whisper(item_data, measures, whisper_turn_mode="auto")
    speaker_turns = cutil.create_turns_whisper(item_data, measures, whisper_turn_mode="speaker")

    assert auto_turns.to_dict("records") == speaker_turns.to_dict("records")
    assert len(auto_turns) == 2


def test_auto_behaves_like_segment_when_speaker_labels_are_absent():
    cutil = load_local_characteristics_util()
    measures = _measures()
    item_data = cutil.create_index_column(_segments_without_speakers(), measures)

    auto_turns = cutil.create_turns_whisper(item_data, measures, whisper_turn_mode="auto")
    segment_turns = cutil.create_turns_whisper(item_data, measures, whisper_turn_mode="segment")

    assert auto_turns.to_dict("records") == segment_turns.to_dict("records")
    assert len(auto_turns) == 2
