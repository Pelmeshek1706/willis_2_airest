import openwillis.speech
print()
print("Stay here....")
print(openwillis.speech.__file__)

import openwillis.speech as ows
# import openwillis.transcribe as owt
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
language = "en"
print(f"Using coherence backend: {coherence.COHERENCE_BACKEND}")
import torch; print("torch version - ", torch.__version__)
INPUT_DIR   = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/woz_end_whisper_test')
OUTPUT_DIR  = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_eng_norm_gemma_test_sentiment-11-2-26') # improved_vader_ua - 11-2-26
# OUTPUT_DIR  = Path('/Users/pelmeshek1706/Desktop/projects/airest_notebooks/result_oppenwillis_ukr_norm_gemma_test_sentiment-11-2-26') # default vader - 11-2-26
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



nums = []
files = list(INPUT_DIR.glob("*.json"))  # materialize to know total

for json_file in tqdm(files, total=len(files), desc="Processing"):
    name = json_file.stem
    print(f"Processing file: {name}")

    # if str(name) in nums:
    #     print(f'Пропускаю {name}')
    #     continue
    if str(name) != "300":
        print(f'Пропускаю {name}')
        continue
    try:
        # 4.1 – Чтение JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            transcript_json = json.load(f)

        words_new, turns_new, summary_new = ows.speech_characteristics(
            json_conf=transcript_json,
            option="simple",
            language=language,
            speaker_label="SPEAKER_A",
            feature_groups=["sentiment", "first_person"],
            min_coherence_turn_length=1
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
        words_path_out = OUTPUT_DIR / f"words_{name}.csv"
        turns_path_out = OUTPUT_DIR / f"turns_{name}.csv"
        summ_path_out = OUTPUT_DIR / f"summary_sc_{name}.csv"

        words_new.to_csv(words_path_out, index=False)
        turns_new.to_csv(turns_path_out, index=False)
        summary_new.to_csv(summ_path_out, index=False)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] ✅ Сохранено: words_{name}.csv, turns_{name}.csv, summary_sc_{name}.csv')

    except Exception as exc:
        print(f'❌ Ошибка при сохранении CSV для {name}:')
        traceback.print_exc()

print('=== Завершено ===')