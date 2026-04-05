import math

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
                "speaker": "interviewer",
                "text": "question now",
                "words": [
                    {"start": 0.8, "end": 1.0, "word": "question"},
                    {"start": 1.0, "end": 1.2, "word": "now"},
                ],
            },
        ]
    }


def test_create_index_column_assigns_consecutive_old_idx_values():
    cutil = load_local_characteristics_util()
    measures = _measures()

    item_data = cutil.create_index_column(_whisper_json()["segments"], measures)
    observed_indices = [
        word[measures["old_index"]]
        for segment in item_data
        for word in segment["words"]
    ]

    assert observed_indices == [0, 1, 2, 3]


def test_filter_json_transcribe_flattens_words_and_carries_speaker_labels():
    cutil = load_local_characteristics_util()
    measures = _measures()

    item_data = cutil.create_index_column(_whisper_json()["segments"], measures)
    filtered_words = cutil.filter_json_transcribe(item_data, measures)

    assert [word["word"] for word in filtered_words] == ["hello", "there", "question", "now"]
    assert [word["speaker"] for word in filtered_words] == [
        "participant",
        "participant",
        "interviewer",
        "interviewer",
    ]
    assert math.isnan(filtered_words[0][measures["pause"]])
    assert filtered_words[2][measures["pause"]] == 0.4


def test_filter_whisper_returns_flat_words_and_required_utterance_columns():
    speech_attribute = load_local_speech_attribute()
    measures = speech_attribute.get_config(speech_attribute.__file__, "text.json")

    filtered_words, utterances = speech_attribute.filter_whisper(
        _whisper_json(),
        measures,
        whisper_turn_mode="segment",
    )

    assert len(filtered_words) == 4
    assert len(utterances) == 2
    assert set(
        [
            measures["utterance_ids"],
            measures["utterance_text"],
            measures["phrases_ids"],
            measures["phrases_texts"],
            measures["words_ids"],
            measures["words_texts"],
            measures["speaker_label"],
        ]
    ).issubset(utterances.columns)

    first_turn = utterances.iloc[0]
    assert first_turn[measures["utterance_ids"]] == (0, 1)
    assert first_turn[measures["words_ids"]] == [0, 1]
    assert first_turn[measures["speaker_label"]] == "participant"
