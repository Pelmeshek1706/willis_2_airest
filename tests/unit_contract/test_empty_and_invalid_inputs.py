import pytest

from tests.helpers.module_loaders import load_local_characteristics_util, load_local_speech_attribute


def _measures():
    speech_attribute = load_local_speech_attribute()
    return speech_attribute.get_config(speech_attribute.__file__, "text.json")


def test_filter_whisper_handles_empty_segments_without_crashing():
    speech_attribute = load_local_speech_attribute()
    measures = speech_attribute.get_config(speech_attribute.__file__, "text.json")

    filtered_words, utterances = speech_attribute.filter_whisper(
        {"segments": []},
        measures,
        whisper_turn_mode="segment",
    )

    assert filtered_words == []
    assert utterances.empty


def test_filter_json_transcribe_ignores_segments_without_words_or_timestamps():
    cutil = load_local_characteristics_util()
    measures = _measures()

    item_data = cutil.create_index_column(
        [
            {"speaker": "participant"},
            {"speaker": "participant", "words": [{"word": "hello"}]},
            {"speaker": "participant", "words": [{"start": 0.0, "word": "partial"}]},
            {"speaker": "participant", "words": [{"end": 0.2, "word": "partial"}]},
        ],
        measures,
    )

    filtered_words = cutil.filter_json_transcribe(item_data, measures)
    assert filtered_words == []


def test_create_turns_whisper_fails_fast_on_invalid_mode():
    cutil = load_local_characteristics_util()
    measures = _measures()

    with pytest.raises(ValueError):
        cutil.create_turns_whisper([], measures, whisper_turn_mode="merge")


def test_filter_whisper_handles_minimal_empty_word_lists_without_crashing():
    speech_attribute = load_local_speech_attribute()
    measures = speech_attribute.get_config(speech_attribute.__file__, "text.json")

    filtered_words, utterances = speech_attribute.filter_whisper(
        {"segments": [{"speaker": "participant", "text": "hi", "words": []}]},
        measures,
        whisper_turn_mode="speaker",
    )

    assert filtered_words == []
    assert utterances.empty


def test_speech_characteristics_returns_schema_stable_empty_outputs_for_empty_payload():
    speech_attribute = load_local_speech_attribute()

    word_df, turn_df, summary_df = speech_attribute.speech_characteristics({}, language="en")

    assert speech_attribute.get_config(speech_attribute.__file__, "text.json")["file_length"] in summary_df.columns
    assert speech_attribute.get_config(speech_attribute.__file__, "text.json")["turn_words"] in turn_df.columns
    assert speech_attribute.get_config(speech_attribute.__file__, "text.json")["word_pause"] in word_df.columns
    assert len(word_df) == 1
    assert len(turn_df) == 1
    assert len(summary_df) == 1
