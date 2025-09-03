"""
SpeechProcessor: Класс для удобной работы с транскрипциями речи и извлечения характеристик.

Использует:
- SpeechAttribute (логика обработки и анализа)
- CharacteristicsUtil (утилиты для фичей и NLP)

Пример использования:
    from speech_processor import SpeechProcessor
    processor = SpeechProcessor()
    result = processor.analyze(json_conf, language="en", ...)
"""
import os
import numpy as np
import logging
from .speech_attribute_class import SpeechAttribute
from .util.characteristics_util_class import CharacteristicsUtil

class SpeechProcessor:
    def __init__(self, config_file="text.json"):
        """
        Инициализация процессора речи с конфигом.
        Args:
            config_file (str): Имя файла конфигурации для measures.
        """
        self.logger = logging.getLogger(__name__)
        self.attribute = SpeechAttribute(config_file)
        self.measures = self.attribute.measures

    def analyze(self, json_conf, language="en", speaker_label=None, min_turn_length=1, min_coherence_turn_length=5, option='coherence'):
        """
        Главный метод для анализа транскрипта речи.
        Args:
            json_conf (dict): JSON с транскрипцией.
            language (str): Язык.
            speaker_label (str): Метка спикера.
            min_turn_length (int): Мин. слов в реплике.
            min_coherence_turn_length (int): Мин. слов для когерентности.
            option (str): 'simple' или 'coherence'.
        Returns:
            dict: Словарь с датафреймами и метаданными.
        """
        try:
            df_list = self.attribute.speech_characteristics(
                json_conf,
                language=language,
                speaker_label=speaker_label,
                min_turn_length=min_turn_length,
                min_coherence_turn_length=min_coherence_turn_length,
                option=option
            )
            return {
                "word_df": df_list[0],
                "turn_df": df_list[1],
                "summ_df": df_list[2],
                "measures": self.measures
            }
        except Exception as e:
            self.logger.error(f"Speech analysis error: {e}")
            return None

    def get_measures(self):
        """
        Получить measures-конфиг.
        Returns:
            dict: measures
        """
        return self.measures

    def download_resources(self, language):
        """
        Скачать необходимые NLP ресурсы для языка.
        Args:
            language (str): Язык ('en', 'uk', 'ua', ...)
        """
        if language in self.measures.get("english_langs", []):
            CharacteristicsUtil.download_nltk_resources()
        if language in ["ua", "uk"]:
            CharacteristicsUtil.download_ua_resources()

    def get_time_columns(self, source):
        """
        Получить названия временных колонок для источника.
        Args:
            source (str): 'aws', 'vosk', 'whisper'
        Returns:
            list: Названия колонок
        """
        return self.attribute.get_time_columns(source)
