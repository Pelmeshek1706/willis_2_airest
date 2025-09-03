"""
SpeechAttribute: A class for processing and analyzing speech transcription data from various sources (AWS, Whisper, Vosk).

Author: Vijay Yadav (refactored by GitHub Copilot)
Website: http://www.bklynhlth.com

This class provides methods to:
- Load configuration for output dataframe columns
- Detect transcription source type
- Filter and process transcription data
- Summarize speech features
- Handle multiple languages and speaker labels

Usage:
    from speech_attribute_class import SpeechAttribute
    sa = SpeechAttribute()
    df_list = sa.speech_characteristics(json_conf, language="en", ...)
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from .util.characteristics_util_class import CharacteristicsUtil

class SpeechAttribute:
    def __init__(self, config_file="text.json"):
        """
        Initialize SpeechAttribute with configuration file.
        Args:
            config_file (str): Name of the config file for measures.
        """
        self.logger = logging.getLogger(__name__)
        self.measures = self.get_config(os.path.abspath(__file__), config_file)

    def get_config(self, filepath, json_file):
        """
        Load configuration for output dataframe columns.
        Args:
            filepath (str): Path to current file.
            json_file (str): Name of config file.
        Returns:
            dict: Measures for dataframe columns.
        """
        dir_name = os.path.dirname(filepath)
        measure_path = os.path.abspath(os.path.join(dir_name, f"config/{json_file}"))
        with open(measure_path) as file:
            measures = json.load(file)
        return measures

    def is_amazon_transcribe(self, json_conf):
        """
        Check if JSON is from Amazon Transcribe.
        Args:
            json_conf (dict): JSON response.
        Returns:
            bool: True if Amazon Transcribe.
        """
        return "jobName" in json_conf and "results" in json_conf

    def is_whisper_transcribe(self, json_conf):
        """
        Check if JSON is from Whisper Transcribe.
        Args:
            json_conf (dict): JSON response.
        Returns:
            bool: True if Whisper Transcribe.
        """
        if "segments" in json_conf:
            if len(json_conf["segments"]) > 0:
                if "words" in json_conf["segments"][0]:
                    return True
        return False

    def filter_transcribe(self, json_conf):
        """
        Filter and process AWS Transcribe JSON.
        Args:
            json_conf (dict): AWS Transcribe JSON.
        Returns:
            tuple: (filtered_json, utterances_df)
        """
        item_data = json_conf["results"]["items"]
        for i, item in enumerate(item_data):
            item[self.measures["old_index"]] = i
        utterances = CharacteristicsUtil.create_turns_aws(item_data, self.measures)
        filter_json = CharacteristicsUtil.filter_json_transcribe_aws(item_data, self.measures)
        return filter_json, utterances

    def filter_whisper(self, json_conf):
        """
        Filter and process Whisper Transcribe JSON.
        Args:
            json_conf (dict): Whisper Transcribe JSON.
        Returns:
            tuple: (filtered_json, utterances_df)
        """
        item_data = json_conf["segments"]
        item_data = CharacteristicsUtil.create_index_column(item_data, self.measures)
        utterances = CharacteristicsUtil.create_turns_whisper(item_data, self.measures)
        filter_json = CharacteristicsUtil.filter_json_transcribe(item_data, self.measures)
        return filter_json, utterances

    def filter_vosk(self, json_conf):
        """
        Filter and process Vosk JSON.
        Args:
            json_conf (list): Vosk JSON list.
        Returns:
            tuple: (filtered_json, utterances_df)
        """
        words = []
        words_ids = []
        for i, item in enumerate(json_conf):
            item[self.measures["old_index"]] = i
            if "word" in item:
                words.append(item["word"])
                words_ids.append(i)
        text = " ".join(words)
        utterances = pd.DataFrame({
            self.measures["utterance_ids"]: [(0, len(json_conf) - 1)],
            self.measures["utterance_text"]: [text],
            self.measures['words_ids']: [words_ids],
            self.measures['words_texts']: [words],
            self.measures['phrases_ids']: [[]],
            self.measures['phrases_texts']: [[]],
            self.measures['speaker_label']: [""]
        })
        return json_conf, utterances

    def common_summary_feature(self, df_summ, json_data, model, speaker_label):
        """
        Calculate file features based on JSON data.
        Args:
            df_summ (pd.DataFrame): Summary dataframe.
            json_data (dict or list): JSON data.
            model (str): Model name.
            speaker_label (str): Speaker label.
        Returns:
            pd.DataFrame: Updated summary dataframe.
        """
        try:
            if model == 'vosk':
                if len(json_data) > 0 and 'end' in json_data[-1]:
                    last_dict = json_data[-1]
                    df_summ['file_length'] = [last_dict['end']]
            else:
                if model == 'aws':
                    json_data = json_data["results"]
                    fl_length, spk_pct = CharacteristicsUtil.calculate_file_feature(json_data, model, speaker_label)
                else:
                    fl_length, spk_pct = CharacteristicsUtil.calculate_file_feature(json_data, model, speaker_label)
                df_summ['file_length'] = [fl_length]
                df_summ['speaker_percentage'] = [spk_pct]
        except Exception as e:
            self.logger.info("Error in file length calculation: %s", e)
        return df_summ

    def process_transcript(self, df_list, json_conf, min_turn_length, min_coherence_turn_length, speaker_label, source, language, option):
        """
        Process transcript and update dataframes.
        Args:
            df_list (list): List of dataframes.
            json_conf (dict): Transcribed JSON.
            min_turn_length (int): Min words per turn.
            min_coherence_turn_length (int): Min words per turn for coherence.
            speaker_label (str): Speaker label.
            source (str): Model name.
            language (str): Language type.
            option (str): 'simple' or 'coherence'.
        Returns:
            list: Updated list of dataframes.
        """
        self.common_summary_feature(df_list[2], json_conf, source, speaker_label)
        if source == 'whisper':
            info = self.filter_whisper(json_conf)
        elif source == 'aws':
            info = self.filter_transcribe(json_conf)
        else:
            info = self.filter_vosk(json_conf)
        if len(info[0]) > 0 and len(info[1]) > 0:
            df_list = CharacteristicsUtil.process_language_feature(
                df_list, info, speaker_label, min_turn_length, min_coherence_turn_length,
                language, self.get_time_columns(source), option, self.measures
            )
        return df_list

    def get_time_columns(self, source):
        """
        Get time column names for source.
        Args:
            source (str): Model name.
        Returns:
            list: Time column names.
        """
        if source == 'aws':
            return ["start_time", "end_time"]
        else:
            return ["start", "end"]

    def speech_characteristics(self, json_conf, language="en", speaker_label=None, min_turn_length=1, min_coherence_turn_length=5, option='coherence'):
        """
        Main entry point: Analyze speech characteristics from transcript JSON.
        Args:
            json_conf (dict): Transcribed JSON.
            language (str): Language type.
            speaker_label (str): Speaker label.
            min_turn_length (int): Min words per turn.
            min_coherence_turn_length (int): Min words per turn for coherence.
            option (str): 'simple' or 'coherence'.
        Returns:
            list: List of dataframes [word_df, turn_df, summ_df].
        """
        df_list = CharacteristicsUtil.create_empty_dataframes(self.measures)
        try:
            if option not in ['simple', 'coherence']:
                raise ValueError("Invalid option. Use 'simple' or 'coherence'.")
            if bool(json_conf):
                language = language[:2].lower() if (language and len(language) >= 2) else "na"
                if language in self.measures.get("english_langs", []):
                    CharacteristicsUtil.download_nltk_resources()
                if language in ["ua", "uk"]:
                    CharacteristicsUtil.download_ua_resources()
                if self.is_whisper_transcribe(json_conf):
                    df_list = self.process_transcript(df_list, json_conf, min_turn_length, min_coherence_turn_length, speaker_label, 'whisper', language, option)
                elif self.is_amazon_transcribe(json_conf):
                    df_list = self.process_transcript(df_list, json_conf, min_turn_length, min_coherence_turn_length, speaker_label, 'aws', language, option)
                else:
                    df_list = self.process_transcript(df_list, json_conf, min_turn_length, min_coherence_turn_length, speaker_label, 'vosk', language, option)
        except Exception as e:
            self.logger.info(f"Error in Speech Characteristics: {e}")
        finally:
            for df in df_list:
                df.loc[0] = np.nan if df.empty else df.loc[0]
        return df_list
