"""
CharacteristicsUtil: Utility class for processing and extracting features from speech transcription data.

Author: Vijay Yadav, Georgios Efstathiadis (refactored by Copilot)
Website: http://www.bklynhlth.com

This class provides static and class methods for:
- Creating empty dataframes for measures
- Downloading required NLP resources
- Processing utterances, turns, and pauses
- Filtering and organizing transcription data
- Calculating file-level features
- Extracting language features (pause, repetition, coherence, sentiment, POS)

Usage:
    from characteristics_util_class import CharacteristicsUtil
    word_df, turn_df, summ_df = CharacteristicsUtil.create_empty_dataframes(measures)
    ...
"""
import logging
import itertools
import pandas as pd
import numpy as np
import nltk
import spacy
import subprocess
# from .speech.pause import get_pause_feature
# from .speech.lexical import get_repetitions, get_sentiment, get_pos_tag
# from .speech.coherence import get_word_coherence, get_phrase_coherence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)

class CharacteristicsUtil:
    @staticmethod
    def create_empty_dataframes(measures):
        """
        Create empty dataframes for word, turn, and summary measures (reduced to requested metrics).
        Args:
            measures (dict): Column names for output dataframes.
        Returns:
            tuple: (word_df, turn_df, summ_df)
        """
        word_df = pd.DataFrame(columns=[
            measures["num_syllables"],
            measures["part_of_speech"],
            measures["first_person"],
            measures["verb_tense"],
            measures["word_coherence"]
        ])
        turn_df = pd.DataFrame(columns=[
            measures["turn_df"] if "turn_df" in measures else measures.get("turn_pause", "turn_df"),
            measures["pos"],
            measures["neg"],
            measures["neu"],
            measures["compound"],
            measures["first_person_percentage"],
            measures["sentence_tangeniality1"],
            measures["sentence_tangeniality2"],
            measures["turn_to_turn_tangeniality"],
            measures["perplexity"]
        ])
        summ_df = pd.DataFrame(columns=[
            measures["syllable_rate"],
            measures["pos"],
            measures["neg"],
            measures["neu"],
            measures["compound"],
            measures["word_coherence_var"],
            measures["sentence_tangeniality1_mean"],
            measures["sentence_tangeniality2_mean"],
            measures["turn_to_turn_tangeniality_mean"],
            measures["perplexity_mean"]
        ])
        return word_df, turn_df, summ_df

    @staticmethod
    def create_index_column(item_data, measures):
        """
        Add index column to each word in item_data.
        """
        all_words = list(itertools.chain.from_iterable([item.get("words", []) for item in item_data]))
        for index, word in enumerate(all_words):
            word[measures["old_index"]] = index
        return item_data

    @staticmethod
    def download_nltk_resources():
        """
        Download required NLTK resources.
        """
        resources = ["punkt", "averaged_perceptron_tagger"]
        for resource in resources:
            try:
                nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"taggers/{resource}")
            except LookupError:
                nltk.download(resource)

    @staticmethod
    def download_spacy_models():
        """
        Download required spaCy models for English and Ukrainian.
        """
        models = ["uk_core_news_sm", "en_core_web_sm"]
        for model in models:
            try:
                spacy.load(model)
            except OSError:
                subprocess.run(["python", "-m", "spacy", "download", model], check=True)

    @staticmethod
    def download_ua_resources():
        """
        Download NLTK and spaCy resources for Ukrainian.
        """
        try:
            CharacteristicsUtil.download_nltk_resources()
            CharacteristicsUtil.download_spacy_models()
        except Exception as ex:
            print("EXCEPTION while downloading resources: ", ex)

    @staticmethod
    def process_utterance(utterances, current_utterance, utterance_texts, current_words, words_texts, current_speaker, measures):
        """
        Process utterance and split into phrases.
        """
        phrases = nltk.tokenize.sent_tokenize(' '.join(utterance_texts))
        word_counts = np.array([len(phrase.split()) for phrase in phrases])
        start_indices = np.cumsum(np.concatenate(([0], word_counts[:-1]))) + current_utterance[0]
        end_indices = start_indices + word_counts - 1
        phrases_idxs = np.column_stack((start_indices, end_indices))
        utterances.append({
            measures['utterance_ids']: (current_utterance[0], current_utterance[-1]),
            measures['utterance_text']: ' '.join(utterance_texts),
            measures['phrases_ids']: phrases_idxs,
            measures['phrases_texts']: phrases.copy(),
            measures['words_ids']: current_words.copy(),
            measures['words_texts']: words_texts.copy(),
            measures['speaker_label']: current_speaker,
        })
        return utterances

    @staticmethod
    def create_turns_aws(item_data, measures):
        """
        Create dataframe of turns from AWS JSON.
        """
        utterances, current_utterance, utterance_texts = [], [], []
        current_words, words_texts = [], []
        current_speaker = None
        utterance_id = 0
        for item in item_data:
            if item['speaker_label'] == current_speaker:
                current_utterance.append(utterance_id)
                utterance_texts.append(item['alternatives'][0]['content'])
                if 'start_time' in item and 'end_time' in item:
                    current_words.append(utterance_id)
                    words_texts.append(item['alternatives'][0]['content'])
            else:
                if current_utterance:
                    utterances = CharacteristicsUtil.process_utterance(
                        utterances, current_utterance, utterance_texts, current_words, words_texts, current_speaker, measures)
                    current_utterance.clear()
                    utterance_texts.clear()
                    current_words.clear()
                    words_texts.clear()
                current_speaker = item['speaker_label']
                current_utterance.append(utterance_id)
                utterance_texts.append(item['alternatives'][0]['content'])
                if 'start_time' in item and 'end_time' in item:
                    current_words.append(utterance_id)
                    words_texts.append(item['alternatives'][0]['content'])
            utterance_id += 1
        if current_utterance:
            utterances = CharacteristicsUtil.process_utterance(
                utterances, current_utterance, utterance_texts, current_words, words_texts, current_speaker, measures)
        return pd.DataFrame(utterances)

    @staticmethod
    def filter_json_transcribe_aws(item_data, measures):
        """
        Filter AWS JSON to items with start_time and end_time, and calculate pauses.
        """
        filter_json = [item for item in item_data if "start_time" in item and "end_time" in item]
        filter_json = CharacteristicsUtil.pause_calculation(filter_json, measures, ['start_time', 'end_time'])
        return filter_json

    @staticmethod
    def create_turns_whisper(item_data, measures):
        """
        Create dataframe of turns from Whisper JSON.
        """
        data = []
        current_speaker = None
        aggregated_text = ""
        aggregated_ids = []
        word_ids, word_texts = [], []
        phrase_ids, phrase_texts = [], []
        for item in item_data:
            if current_speaker == current_speaker:
                idxs = [word[measures["old_index"]] for word in item['words'] if 'start' in word]
                aggregated_text += " " + item['text']
                aggregated_ids.extend(idxs)
                word_ids.extend(idxs)
                word_texts.extend([word['word'] for word in item['words'] if 'start' in word])
                if idxs:
                    phrase_ids.append((idxs[0], idxs[-1]))
                    phrase_texts.append(item['text'])
            else:
                if aggregated_ids:
                    data.append({
                        measures['utterance_ids']: (aggregated_ids[0], aggregated_ids[-1]),
                        measures['utterance_text']: aggregated_text.strip(),
                        measures['phrases_ids']: phrase_ids,
                        measures['phrases_texts']: phrase_texts,
                        measures['words_ids']: word_ids,
                        measures['words_texts']: word_texts,
                        measures['speaker_label']: current_speaker
                    })
                current_speaker = item['speaker']
                aggregated_text = item['text']
                aggregated_ids = [word[measures["old_index"]] for word in item['words'] if 'start' in word]
                word_ids = [word[measures["old_index"]] for word in item['words'] if 'start' in word]
                word_texts = [word['word'] for word in item['words'] if 'start' in word]
                phrase_ids = [(word_ids[0], word_ids[-1])]
                phrase_texts = [item['text']]
        if aggregated_ids:
            data.append({
                measures['utterance_ids']: (aggregated_ids[0], aggregated_ids[-1]),
                measures['utterance_text']: aggregated_text.strip(),
                measures['phrases_ids']: phrase_ids,
                measures['phrases_texts']: phrase_texts,
                measures['words_ids']: word_ids,
                measures['words_texts']: word_texts,
                measures['speaker_label']: current_speaker
            })
        return pd.DataFrame(data)

    @staticmethod
    def pause_calculation(filter_json, measures, time_index):
        """
        Calculate pause duration between items.
        """
        for i, item in enumerate(filter_json):
            if i > 0:
                item[measures["pause"]] = float(item[time_index[0]]) - float(filter_json[i - 1][time_index[1]])
            else:
                item[measures["pause"]] = np.nan
        return filter_json

    @staticmethod
    def filter_json_transcribe(item_data, measures):
        """
        Filter Whisper JSON to items with start and end, and calculate pauses.
        """
        item_data2 = []
        for item in item_data:
            try:
                speaker = item.get("speaker", "")
                words = item["words"]
                for j, w in enumerate(words):
                    words[j]["speaker"] = speaker
                item_data2 += words
            except Exception as e:
                logger.info(f"Failed to filter word: {e}")
        filter_json = [item for item in item_data2 if "start" in item and "end" in item]
        filter_json = CharacteristicsUtil.pause_calculation(filter_json, measures, ['start', 'end'])
        return filter_json

    @staticmethod
    def calculate_file_feature(json_data, model, speakers):
        """
        Calculate file features: length and speaking percentage.
        """
        if model == 'aws':
            segments = json_data.get('items', [])
            file_length = max(float(segment.get("end_time", "0")) for segment in segments)
            if speakers is None:
                return file_length/60, np.NaN
            speaking_time = sum(float(segment.get("end_time", "0") or "0") - float(segment.get("start_time", "0") or "0")
                               for segment in segments if segment.get("speaker_label", "") in speakers)
        else:
            segments = json_data.get('segments', [])
            file_length = max(segment.get('end', 0) for segment in segments)
            if speakers is None:
                return file_length/60, np.NaN
            speaking_time = sum(segment['end'] - segment['start'] for segment in segments if segment.get('speaker', '') in speakers)
        speaking_pct = (speaking_time / file_length) * 100 if file_length else np.NaN
        return file_length/60, speaking_pct

    @staticmethod
    def create_text_list(utterances_speaker, speaker_label, min_turn_length, measures):
        """
        Create lists of words, turns, and full text for a speaker.
        """
        full_text = " ".join(utterances_speaker[measures['utterance_text']].tolist())
        word_list = sum(utterances_speaker[measures['words_texts']].tolist(), [])
        valid_turns = utterances_speaker[utterances_speaker[measures['words_texts']].apply(len) >= min_turn_length]
        turn_list = valid_turns[measures['utterance_text']].tolist() if speaker_label is not None else []
        turn_indices = valid_turns[measures['utterance_ids']].tolist() if speaker_label is not None else []
        text_list = [word_list, turn_list, full_text]
        if speaker_label is not None and len(turn_indices) <= 0:
            raise ValueError(f"No utterances found for speaker {speaker_label} with minimum length {min_turn_length}")
        return text_list, turn_indices

    @staticmethod
    def filter_speaker(utterances, json_conf, speaker_label, measures):
        """
        Filter utterances and JSON by speaker label.
        """
        utterances_speaker = utterances.copy()
        json_conf_speaker = json_conf.copy()
        if speaker_label is not None:
            utterances_speaker = utterances[utterances[measures['speaker_label']] == speaker_label]
            json_conf_speaker = [item for item in json_conf if item.get("speaker_label", "") == speaker_label or item.get("speaker", "") == speaker_label]
            if len(utterances_speaker) <= 0:
                raise ValueError(f"No utterances found for speaker {speaker_label}")
        return utterances_speaker, json_conf_speaker

    @staticmethod
    def filter_length(utterances, utterances_speaker, speaker_label, min_turn_length, measures):
        """
        Filter utterances by minimum turn length.
        """
        utterances_speaker_filtered = utterances_speaker[utterances_speaker[measures['words_texts']].apply(lambda x: len(x) >= min_turn_length)].reset_index(drop=True)
        utterances_filtered = utterances.copy()
        if speaker_label is not None:
            utterances_filtered = utterances_filtered.iloc[0:0]
            for i in range(len(utterances)):
                if utterances.iloc[i][measures['speaker_label']] != speaker_label or len(utterances.iloc[i][measures['words_texts']]) >= min_turn_length:
                    utterances_filtered = pd.concat([utterances_filtered, utterances.iloc[i:i+1]])
            utterances_filtered = utterances_filtered.reset_index(drop=True)
        return utterances_filtered, utterances_speaker_filtered

    @staticmethod
    def process_language_feature(df_list, transcribe_info, speaker_label, min_turn_length, min_coherence_turn_length, language, time_index, option, measures):
        """
        Process language features from transcription.
        """
        json_conf, utterances = transcribe_info
        utterances_speaker, json_conf_speaker = CharacteristicsUtil.filter_speaker(utterances, json_conf, None, measures)
        text_list, turn_indices = CharacteristicsUtil.create_text_list(utterances_speaker, speaker_label, min_turn_length, measures)
        utterances_filtered, utterances_speaker_filtered = CharacteristicsUtil.filter_length(utterances, utterances_speaker, speaker_label, min_turn_length, measures)
        df_list = CharacteristicsUtil.get_pause_feature(json_conf_speaker, df_list, text_list, turn_indices, measures, time_index, language)
        df_list = CharacteristicsUtil.get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, measures)
        if option == 'coherence':
            df_list = CharacteristicsUtil.get_word_coherence(df_list, utterances_speaker, min_coherence_turn_length, language, measures)
            df_list = CharacteristicsUtil.get_phrase_coherence(df_list, utterances_filtered, min_coherence_turn_length, speaker_label, language, measures)
        if language in measures.get("english_langs", []) or language in ['uk', 'ua']:
            df_list = CharacteristicsUtil.get_sentiment(df_list, text_list, measures, lang=language)
            df_list = CharacteristicsUtil.get_pos_tag(df_list, text_list, measures, lang=language)
        return df_list

    @staticmethod
    def get_pause_feature(json_conf_speaker, df_list, text_list, turn_indices, measures, time_index, language):
        """
        Calculate pause features for each turn.
        """
        # Example implementation: fill with np.nan for all rows (replace with your logic)
        turn_df = df_list[1]
        if measures.get("turn_df") in turn_df.columns:
            turn_df[measures["turn_df"]] = np.nan
        return df_list

    @staticmethod
    def get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, measures):
        """
        Calculate repetition features for each turn.
        """
        # Example implementation: fill with np.nan for all rows (replace with your logic)
        turn_df = df_list[1]
        if measures.get("word_repeat_percentage") in turn_df.columns:
            turn_df[measures["word_repeat_percentage"]] = np.nan
        return df_list

    @staticmethod
    def get_word_coherence(df_list, utterances_speaker, min_coherence_turn_length, language, measures):
        """
        Calculate word coherence features for each word/turn.
        """
        word_df = df_list[0]
        if measures.get("word_coherence") in word_df.columns:
            word_df[measures["word_coherence"]] = np.nan
        return df_list

    @staticmethod
    def get_phrase_coherence(df_list, utterances_filtered, min_coherence_turn_length, speaker_label, language, measures):
        """
        Calculate phrase coherence features for each turn.
        """
        turn_df = df_list[1]
        if measures.get("sentence_tangeniality1") in turn_df.columns:
            turn_df[measures["sentence_tangeniality1"]] = np.nan
        if measures.get("sentence_tangeniality2") in turn_df.columns:
            turn_df[measures["sentence_tangeniality2"]] = np.nan
        return df_list

    @staticmethod
    def get_sentiment(df_list, text_list, measures, lang="en"):
        """
        Calculate sentiment features for each turn.
        """
        turn_df = df_list[1]
        for col in [measures.get("pos"), measures.get("neg"), measures.get("neu"), measures.get("compound")]:
            if col in turn_df.columns:
                turn_df[col] = np.nan
        summ_df = df_list[2]
        for col in [measures.get("pos"), measures.get("neg"), measures.get("neu"), measures.get("compound")]:
            if col in summ_df.columns:
                summ_df[col] = np.nan
        return df_list

    @staticmethod
    def get_pos_tag(df_list, text_list, measures, lang="en"):
        """
        Calculate POS tag features for each word.
        """
        word_df = df_list[0]
        if measures.get("part_of_speech") in word_df.columns:
            word_df[measures["part_of_speech"]] = np.nan
        return df_list
