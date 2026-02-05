import openwillis.speech
print()
print("Stay here....")
print(openwillis.speech.__file__)

import openwillis.speech as ows
import openwillis.transcribe as owt
print("continue...")

import os
import json
import traceback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pandas as pd

from openwillis.speech.util.speech import coherence

coherence.COHERENCE_BACKEND = "gemma"   # or "gemma"
print(f"Using coherence backend: {coherence.COHERENCE_BACKEND}")
import torch; print("torch version - ", torch.__version__)
INPUT_DIR   = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/woz_end_whisper_test_ukr')
OUTPUT_SOURCE_DIR = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_ukr_norm_gemma')
OUTPUT_DIR  = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_ukr_norm_gemma_updated_sentiment-5-2-26')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

nums = []
for json_file in tqdm(INPUT_DIR.glob('*.json')):
    name = json_file.stem
    print(f"Processing file: {name}")

    if str(name) in nums:
        print(f'Пропускаю {name}')
        continue
    try:
        # 4.1 – Чтение JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            transcript_json = json.load(f)

        words_new, turns_new, summary_new = ows.speech_characteristics(
            json_conf=transcript_json,
            option="simple",
            language="ua",
            speaker_label="SPEAKER_A",
            feature_groups=["sentiment", "first_person"],
        )
    except RuntimeError as rexc:
        # skip.append(str(name))
        print(f'❌ Пропускаю {json_file} из-за ошибки RuntimeError:')
    except Exception as exc:
        # Если возникла ошибка – выводим трассировку и пропускаем файл
        print(f'❌ Ошибка при обработке {json_file}:')
        traceback.print_exc()
        continue
    try:

        words_path_in = OUTPUT_SOURCE_DIR / f"words_{name}.csv"
        turns_path_in = OUTPUT_SOURCE_DIR / f"turns_{name}.csv"
        summ_path_in = OUTPUT_SOURCE_DIR / f"summary_sc_{name}.csv"

        words_path_out = OUTPUT_DIR / f"words_{name}.csv"
        turns_path_out = OUTPUT_DIR / f"turns_{name}.csv"
        summ_path_out = OUTPUT_DIR / f"summary_sc_{name}.csv"

        if not (words_path_in.exists() and turns_path_in.exists() and summ_path_in.exists()):
            print(f"❌ Missing source CSVs for {name}, skipping.")
            continue

        words_old = pd.read_csv(words_path_in)
        turns_old = pd.read_csv(turns_path_in)
        summ_old = pd.read_csv(summ_path_in)

        words_cols = ["first_person"]
        turns_cols = [
            "sentiment_pos","sentiment_neg","sentiment_neu","sentiment_overall",
            "mattr_5","mattr_10","mattr_25","mattr_50","mattr_100",
            "first_person_percentage","first_person_sentiment_positive","first_person_sentiment_negative"
        ]
        summ_cols = [
            "sentiment_pos","sentiment_neg","sentiment_neu","sentiment_overall",
            "mattr_5","mattr_10","mattr_25","mattr_50","mattr_100",
            "first_person_percentage","first_person_sentiment_positive",
            "first_person_sentiment_negative","first_person_sentiment_overall"
        ]

        words_old[words_cols] = words_new[words_cols].values
        turns_old[turns_cols] = turns_new[turns_cols].values
        summ_old[summ_cols] = summary_new[summ_cols].values

        words_old.to_csv(words_path_out, index=False)
        turns_old.to_csv(turns_path_out, index=False)
        summ_old.to_csv(summ_path_out, index=False)
    except Exception as exc:
        print(f'❌ Ошибка при сохранении CSV для {name}:')
        traceback.print_exc()

print('=== Завершено ===')