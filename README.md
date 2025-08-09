## Introduction

Within the AIREST project, the sentiment analysis capabilities of NLP algorithms were verified using a corpus of Telegram user messages (COSMUS dataset). The evaluation includes several sentiment analysis models, allowing identification of the most reliable option for further usage.

## COSMUS Dataset Overview

- **COSMUS** is a sentiment analysis dataset composed of Telegram messages in Ukrainian and Russian languages .
- Data is stored in Parquet format, containing approximately ~12,200 entries .
- Each entry consists of message content (`document_content`), sentiment labels (`annotator_sentiment`), and additional metadata fields: `response_id`, `document_id`, `user_id`, `response_timestamp`, `annotation_date`, `username`, `unique_document_id`, `language_wc`, `document_length`, `gpt_labels_v1`, `language_gpt`, `language_manual`, `language`, `stratification_label`, and `df_set` .
- Sentiment labels are manually annotated with values: "negative", "neutral", "positive", "mixed" .
- Dataset License: MIT .
- Associated tags include: `sentiment`, `social networks`, `Telegram` .
- Objective: analyzing sentiment polarity of social media posts .
- Accessible through Hugging Face Datasets and integrated with Pandas .
- Single available split: `train` on the entire dataset .
- Primary languages: Ukrainian and Russian .

## Section 1.1 — Sentiment Analysis

The main model evaluated is **YShynkarov/ukr-roberta-cosmus-sentiment**, a fine-tuned variant of `ukr-roberta` adapted specifically to Ukrainian and Russian texts. Additionally, several other models were tested for comparative evaluation:

- **cardiffnlp/twitter-xlm-roberta-base-sentiment**  
- **tabularisai/multilingual-sentiment-analysis**  
- **Vader Sentiment via OpenWillis** *(texts translated from Ukrainian to English using Yehor/kulyk-uk-en)*

### Comparative Results Table

| Model                                                         | Accuracy | Negative (P / R / F1) | Neutral (P / R / F1) | Positive (P / R / F1) | Macro Avg (P / R / F1) | Weighted Avg (P / R / F1) |
|:--------------------------------------------------------------|---------:|----------------------:|---------------------:|----------------------:|-----------------------:|--------------------------:|
| `YShynkarov/ukr-roberta-cosmus-sentiment`                     |   76.80% |    0.90 / 0.66 / 0.76 |   0.71 / 0.87 / 0.78 |    0.73 / 0.77 / 0.75 |     0.78 / 0.77 / 0.76 |             0.79 / 0.77 / 0.77 |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment`               |   67.12% |    0.75 / 0.60 / 0.67 |   0.60 / 0.81 / 0.69 |    0.79 / 0.52 / 0.63 |     0.71 / 0.65 / 0.66 |             0.70 / 0.67 / 0.67 |
| `tabularisai/multilingual-sentiment-analysis`                 |   49.56% |    0.52 / 0.69 / 0.59 |   0.58 / 0.24 / 0.34 |    0.42 / 0.64 / 0.50 |     0.50 / 0.52 / 0.48 |             0.52 / 0.50 / 0.47 |
| `Vader Sentiment (OpenWillis, uk-en translation)`             |   43.55% |    0.77 / 0.03 / 0.05 |   0.42 / 0.98 / 0.59 |    0.78 / 0.13 / 0.23 |     0.66 / 0.38 / 0.29 |             0.63 / 0.44 / 0.30 |

### Evaluation Methodology

1. **Data loading and filtering**: Dataset loaded using Hugging Face Datasets; records labeled as "mixed" were removed.
2. **Label encoding**: Textual labels ("negative", "neutral", "positive") were mapped to numerical values (-1, 0, +1).
3. **Model initialization**: Pipelines for sentiment analysis were set up for each model using identical tokenization and inference parameters.
4. **Predictions**: The sentiment category for each text entry was predicted using the respective model’s inference pipeline.
5. **Vader Sentiment testing**: Due to the Vader Sentiment Analyzer supporting English input only, all texts were first translated from Ukrainian to English using the Yehor/kulyk-uk-en model, specifically fine-tuned for Ukrainian-English translation. After translation, texts were labeled using the Vader algorithm provided by OpenWillis.
6. **Metric computation**: Accuracy, precision, recall, and F1 scores were calculated for negative, neutral, and positive classes using sklearn metrics.
7. **Result comparison**: Performance metrics were summarized in the comparative results table.

## POS Tagging Evaluation for Ukrainian (spaCy-uk vs. UD Gold)

## Introduction

As part of the AIREST project, we conducted an evaluation of part-of-speech (POS) tagging accuracy for the Ukrainian language.  
The goal was to ensure that the **spaCy-uk** model used in OpenWillis achieves expert-level accuracy, comparable to manually annotated gold-standard data, and does not fall behind alternative solutions by more than the acceptable threshold.  

## UD_Ukrainian-ParlaMint Dataset Overview

- **UD_Ukrainian-ParlaMint** is a corpus of Ukrainian parliamentary speeches annotated according to the [Universal Dependencies](https://universaldependencies.org/) standards.  
- The dataset includes **morphological and syntactic annotations** that have been manually verified and corrected.  
- Annotation resources include **UD_Ukrainian-IU** and the **VESUM** dictionary.  
- In the metadata, the annotation status for lemmas, UPOS, morphological features, and dependencies is *manual native*.  
- The annotation covers **UPOS** (universal parts of speech), **XPOS**, morphological features (FEATS, including `Tense`), and syntactic dependencies (HEAD, DEPREL).  
- The data format is CoNLL-U, fully compatible with UD processing tools.  
- The main objective of using this dataset is to assess the accuracy of POS tagging and the extraction of the `Tense` feature for verbs.

## Section 1.1 — POS Tagging Evaluation

Two tools were evaluated:
- **spaCy-uk** — a Universal Dependencies–based model integrated into OpenWillis for Ukrainian.  
- **Stanza-uk** — an NLP library from Stanford NLP, trained on Ukrainian UD corpora, used here as a reference point.

Evaluation methodology:
1. **Data**: Gold-standard annotation from UD_Ukrainian-ParlaMint was used as the reference.
2. **Model predictions**: For each token, UPOS and Tense labels were obtained from spaCy-uk and Stanza-uk.
3. **Token alignment**: Character-based alignment was applied to correctly compare model predictions with the gold standard.
4. **Metrics**:  
   - UPOS: Macro-averaged F1 score computed across 10 target UPOS classes.  
   - Tense: Macro-F1 score computed among verbs (`VERB`).
5. **Success criteria**:
   - spaCy-uk Macro-F1 ≥ target threshold (comparable to the English OpenWillis model).
   - The difference between spaCy-uk and Stanza-uk Macro-F1 ≤ 2 percentage points; exceeding this threshold indicates potential model misconfiguration rather than unavoidable linguistic ambiguity.

### Comparative Results Table

| Dataset | spaCy UPOS F1 | Stanza UPOS F1 | Δ F1 UPOS (vs Stanza) |
|:--------|--------------:|---------------:|----------------------:|
| **dev**   | 0.972         | 0.975          | 0.285%                |
| **test**  | 0.973         | 0.972          | -0.112%               |
| **train** | 0.974         | 0.974          | 0.071%                |

### Analysis and Conclusion

Across all splits (dev, test, train), **spaCy-uk** demonstrates macro-F1 scores above 0.97 for UPOS tagging, consistently matching or nearly matching the performance of **Stanza-uk**.  
The observed differences between the two models are within ±0.3 percentage points — well below the predefined 2 p.p. threshold.  

This confirms that **spaCy-uk** maintains near-expert accuracy for POS tagging on Ukrainian, making it fully reliable for downstream features in OpenWillis that depend on POS counts (e.g., first-person pronoun percentage, noun–verb ratio, adjective density).  
Given these results, spaCy-uk can be confidently used as the primary POS tagging module for Ukrainian in production workflows without compromising accuracy compared to Stanza-uk.

## Section 1.2 — Tense Evaluation

In addition to UPOS tagging, we evaluated the accuracy of extracting the `Tense` morphological feature from the UD_Ukrainian-ParlaMint dataset.  
The `Tense` feature in Universal Dependencies denotes grammatical tense (e.g., `Past`, `Pres`, `Fut`) and is primarily relevant for verbs, although it can appear in other categories (such as participles tagged as `ADJ`).

Two evaluation settings were applied:

- **VERB only** (`restrict_to_verbs=True`):  
  Evaluation is restricted to tokens where the gold UPOS label is `VERB`.  
  This focuses the metric on canonical verb forms, excluding other parts of speech that may occasionally carry a `Tense` feature.

- **has gold Tense** (`restrict_to_verbs=False` with filtering):  
  Evaluation includes all tokens in the corpus that have a `Tense` value in the gold annotation, regardless of UPOS.  
  This setting captures cases where tense information is attached to non-verb forms (e.g., participles annotated as `ADJ` in UD).

### Comparative Results Table — Tense (Macro-F1)

| Dataset  | spaCy Tense F1<br>(VERB only) | Stanza Tense F1<br>(VERB only) | Δ F1<br>(VERB only) | spaCy Tense F1<br>(has gold Tense) | Stanza Tense F1<br>(has gold Tense) | Δ F1<br>(has gold Tense) |
|:---------|------------------------------:|-------------------------------:|--------------------:|-----------------------------------:|------------------------------------:|-------------------------:|
| **dev**  | 0.972                         | 0.929                          | -4.32%              | 0.981                              | 0.706                               | -27.46%                  |
| **test** | 0.959                         | 0.926                          | -3.31%              | 0.729                              | 0.708                               | -2.09%                   |
| **train**| 0.967                         | 0.952                          | -1.50%              | 0.731                              | 0.719                               | -1.27%                   |

### Analysis and Conclusion

For **VERB only** evaluation, spaCy-uk consistently achieves high macro-F1 scores (0.959–0.972) and performs comparably to Stanza-uk, with differences between -1.5% and -4.3%, remaining within acceptable limits.

In the **has gold Tense** setting, performance differences between spaCy-uk and Stanza-uk are minimal on test and train sets (≤ 2.1%), but a substantial gap (-27.46%) is observed on the dev set.  
This large dev-set gap is likely explained by differences in how each model handles non-verbal forms carrying tense information, such as adjectival participles, which are less frequent and harder to predict accurately.

Overall, **spaCy-uk** demonstrates reliable tense prediction for canonical verbs, making it suitable for downstream analysis involving verb tense usage.  
However, for tasks requiring precise tense detection in non-verb forms, additional model fine-tuning or post-processing rules may be necessary to bridge the performance gap observed in the dev split.