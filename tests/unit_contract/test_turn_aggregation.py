import pandas as pd

from tests.helpers.module_loaders import load_local_characteristics_util, load_local_speech_attribute


def _measures():
    speech_attribute = load_local_speech_attribute()
    return speech_attribute.get_config(speech_attribute.__file__, "text.json")


def _whisper_json():
    return {
        "segments": [
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
    }


def test_speaker_mode_merges_consecutive_segments_and_preserves_metadata():
    cutil = load_local_characteristics_util()
    measures = _measures()
    item_data = cutil.create_index_column(_whisper_json()["segments"], measures)

    turns = cutil.create_turns_whisper(item_data, measures, whisper_turn_mode="speaker")

    assert len(turns) == 2

    first_turn = turns.iloc[0]
    assert first_turn[measures["utterance_text"]] == "hello there again friend"
    assert first_turn[measures["utterance_ids"]] == (0, 3)
    assert first_turn[measures["words_ids"]] == [0, 1, 2, 3]
    assert first_turn[measures["speaker_label"]] == "participant"

    second_turn = turns.iloc[1]
    assert second_turn[measures["utterance_text"]] == "question now"
    assert second_turn[measures["utterance_ids"]] == (4, 5)
    assert second_turn[measures["speaker_label"]] == "interviewer"


def test_segment_mode_keeps_one_turn_per_segment():
    cutil = load_local_characteristics_util()
    measures = _measures()
    item_data = cutil.create_index_column(_whisper_json()["segments"], measures)

    turns = cutil.create_turns_whisper(item_data, measures, whisper_turn_mode="segment")

    assert len(turns) == 3
    assert turns.iloc[0][measures["utterance_ids"]] == (0, 1)
    assert turns.iloc[1][measures["utterance_ids"]] == (2, 3)
    assert turns.iloc[2][measures["utterance_ids"]] == (4, 5)


def test_process_transcript_passes_speaker_scope_for_whisper_speaker_mode(monkeypatch):
    speech_attribute = load_local_speech_attribute()
    captured = {}

    monkeypatch.setattr(speech_attribute, "common_summary_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        speech_attribute,
        "filter_whisper",
        lambda json_conf, measures, whisper_turn_mode="auto": (["word"], pd.DataFrame({"utterance": [1]})),
    )

    def fake_process_language_feature(*args, **kwargs):
        captured["speaker_filter_label"] = kwargs["speaker_filter_label"]
        captured["coherence_speaker_label"] = kwargs["coherence_speaker_label"]
        return args[0]

    monkeypatch.setattr(
        speech_attribute.cutil,
        "process_language_feature",
        fake_process_language_feature,
        raising=False,
    )

    df_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]
    out = speech_attribute.process_transcript(
        df_list=df_list,
        json_conf=_whisper_json(),
        measures={},
        min_turn_length=1,
        min_coherence_turn_length=1,
        speaker_label="participant",
        source="whisper",
        language="en",
        option="coherence",
        whisper_turn_mode="speaker",
    )

    assert out is df_list
    assert captured == {
        "speaker_filter_label": "participant",
        "coherence_speaker_label": "participant",
    }


def test_process_transcript_passes_speaker_scope_for_whisper_auto_mode(monkeypatch):
    speech_attribute = load_local_speech_attribute()
    captured = {}

    monkeypatch.setattr(speech_attribute, "common_summary_feature", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        speech_attribute,
        "filter_whisper",
        lambda json_conf, measures, whisper_turn_mode="auto": (["word"], pd.DataFrame({"utterance": [1]})),
    )

    def fake_process_language_feature(*args, **kwargs):
        captured["speaker_filter_label"] = kwargs["speaker_filter_label"]
        captured["coherence_speaker_label"] = kwargs["coherence_speaker_label"]
        return args[0]

    monkeypatch.setattr(
        speech_attribute.cutil,
        "process_language_feature",
        fake_process_language_feature,
        raising=False,
    )

    speech_attribute.process_transcript(
        df_list=[pd.DataFrame(), pd.DataFrame(), pd.DataFrame()],
        json_conf=_whisper_json(),
        measures={},
        min_turn_length=1,
        min_coherence_turn_length=1,
        speaker_label="participant",
        source="whisper",
        language="en",
        option="coherence",
        whisper_turn_mode="auto",
    )

    assert captured == {
        "speaker_filter_label": "participant",
        "coherence_speaker_label": "participant",
    }
