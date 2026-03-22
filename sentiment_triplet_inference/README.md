# Triplet Sentiment Inference

Скрипт сравнивает 3 анализатора сентимента на OpenWillis/Whisper JSON:

- `vader` (классический английский VADER)
- `vader-ua` (обычный внешний `vader-ua`)
- `vader-ua-improved` (улучшенный класс из проекта)

## Файлы

- Скрипт: `/Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/sentiment_triplet_inference/run_triplet_sentiment_inference.py`
- Выходы по умолчанию: `/Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/sentiment_triplet_inference/outputs`

## Запуск (один файл)

```bash
airest/.venv/bin/python /Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/sentiment_triplet_inference/run_triplet_sentiment_inference.py \
  --inputs /Users/pelmeshek1706/Desktop/projects/airest_notebooks/woz_end_whisper_test_ukr/300.json
```

## Запуск (вся директория)

```bash
airest/.venv/bin/python /Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/sentiment_triplet_inference/run_triplet_sentiment_inference.py \
  --inputs /Users/pelmeshek1706/Desktop/projects/airest_notebooks/woz_end_whisper_test_ukr
```

## Имена выходных файлов

Для каждого входного `<n>.json` создаются 3 CSV:

- `sentiment_<n>_vader.csv`
- `sentiment_<n>_vader-ua.csv`
- `sentiment_<n>_vader-ua-improved.csv`

Каждый CSV содержит:

- строки `row_type=turn` (по сегментам)
- строку `row_type=summary` (по полному тексту)
