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

### Introduction

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

## Section 1.2 — POS Tagging Evaluation

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

## Section 1.3 — Tense Evaluation

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

## Section 1.4 Validation of Syllable Counter and “Syllables-per-Minute” Metric (Ukrainian)

### Goal (Brief)

For timing biomarkers (SPM, articulation rate), an accurate syllable count is required. English rules from NLTK are not suitable for Ukrainian, so we built a custom spaCy-uk component and validated it on real data, comparing it with a text-based baseline (Pyphen) and an acoustic gold standard (Praat Syllable Nuclei v3).

---

### Dataset: “Telebachennia Toronto” (Summary)

- **Source:** A popular Ukrainian satirical news show “Telebachennia Toronto” / Toronto TV (hosted by the persona Michael Shchur, created by journalist Roman Vintoniv). The show is regularly described in English-language media as a “news with comedy” format aimed at explaining complex topics to a wide audience.
- **Corpus format:** 5–7-second audio clips from episodes, processed with a transcriber; fields: path, transcript, transcript_len, audio_dur_sec. Total of 18,302 clips, average length ≈ 5 s.
- **Important:** Some transcripts don’t fully match the audio (repetitions/interjections/cuts), affecting “text ↔ audio” comparison.
- **Context:** See profiles on [The Fix](https://thefix.media/) and [Kyiv Independent](https://kyivindependent.com/) for more info about the Toronto TV project in English.

---

### Syllable Counting Methods

**Text-based:**
1. **spaCy-uk (our component):** Rules based on Pyphen + Ukrainian-specific patches (apostrophe/hyphens/“йо”, rare syllabic “r/l”).
2. **Pyphen:** Hyphenation as a simple approximation of syllabic structure.
3. **NLTK (SSP):** Universal sonority tokenizer (used as a sanity-check).

**Audio-based:**
4. **Praat Syllable Nuclei v3 (Original):** Detects intensity peaks with voicing filter; also provides phonation time and pauses.
5. **Praat-Like:** Our simplified implementation of the same principle in Python/Parselmouth (without external .praat scripts).

---

### Evaluation Protocol

- For each clip, compute: syll_spacy, syll_pyphen, syll_nltk, syll_praat_original, syll_praat_like; also SPM = syllables / (duration/60).
- **Metrics:** Pairwise MAE, Bias, Pearson r, Spearman ρ, Bland–Altman LOA, and ICC(2,1) (absolute agreement).
- Praat v3 threshold settings were taken from open recommendations, without fine-tuning for noise/channel.

---

### Results (Full Run)

#### 1) Main MAE Summary

| Method Pair              | MAE   |
|--------------------------|-------|
| spaCy-uk vs Pyphen       | 0.517 |
| spaCy-uk vs NLTK         | 3.776 |
| Praat-Like vs Praat Orig | 2.423 |
| spaCy-uk vs Praat Orig   | 9.778 |
| Pyphen vs Praat Orig     | 9.836 |
| NLTK vs Praat Orig       | 6.680 |

Overall ICC(2,1) across all raters: **0.656** (moderate).

#### 2) Within-Class Comparisons (Audio and Text)

##### Audio ↔ Audio (n=18,210)
- MAE = 2.423, Bias = −1.082; r = 0.908, ρ = 0.892; LOA ≈ [−6.94; 4.78]; ICC(2,1) within audio = 0.897.
- **Conclusion:** Praat-Like closely approximates Praat v3; we use Praat Original as the gold standard (it also provides phonation time).

##### Text ↔ Text (n=18,210)
- spaCy-uk vs Pyphen: MAE = 0.517, Bias = −0.106; r = 0.988, ρ = 0.989; LOA is narrow.
- spaCy-uk vs NLTK: MAE = 3.776, Bias = +3.425; r = 0.961, ρ = 0.956.
- Pyphen vs NLTK: MAE = 3.681, Bias = +3.532; r = 0.972, ρ = 0.965.
- ICC(2,1) within text = 0.937.
- **Conclusion:** spaCy-uk and Pyphen are well-aligned; NLTK (SSP) systematically overestimates or segments differently for Ukrainian, so it’s left as a sanity-check.

---

### Interpretation

- Everything is logical within each class:
    - Audio methods show high agreement (ICC ≈ 0.90), i.e., acoustic syllable nucleus detection is robust.
    - Text counters are consistent with each other (spaCy-uk ↔ Pyphen).
- Text ↔ audio diverges significantly (MAE ≈ 9–10; overall ICC ≈ 0.66) due to partial mismatch between transcripts and audio in the corpus — a common situation for auto-generated subtitles/snippets. On subsets where “text ≈ audio,” the divergence is small (by manual check).
- For clinical timing (SPM, articulation rate), it is correct to rely on the audio gold standard (Praat v3) and use its phonation time. The described methodology is widely used precisely without a transcript.


# Section E — Validation of Tangentiality, Perplexity, and Coherence (Ukrainian)

---

## Dataset

- **DAIC‑WOZ Depression Database** — 189 clinical interview sessions, collected via the Wizard‑of‑Oz paradigm (virtual interviewer “Ellie”). [oai_citation:0‡DAIC‑WOZ](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)  
- Sessions are split into **train/dev/test**: 107 / 35 / 47 respectively. [oai_citation:1‡Research on DAIC‑WOZ](https://braininformatics.springeropen.com/articles/10.1186/s40708-023-00185-9)  
- Each participant has a PHQ‑8 assessment (total score from 0 to 24); PHQ‑8 ≥ 10 is used as the binary marker of depression. [oai_citation:2‡Research on DAIC‑WOZ](https://braininformatics.springeropen.com/articles/10.1186/s40708-023-00185-9)  
- Audio recordings (16 kHz), text transcripts, acoustic and visual features (OpenFace / COVAREP, etc.). [oai_citation:3‡DAIC‑WOZ](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf)  

- **Usage in our project:**  
  The English transcripts have been translated into Ukrainian using the **Yehor/kulyk‑en‑uk** model. [oai_citation:4‡Hugging Face](https://huggingface.co/posts/Yehor/884755060089981)  

---

## Methods

- **Embeddings (Gemma):**
  - For both English and Ukrainian we use `google/embeddinggemma-300m` (L2 normalization, cosine similarity).
  - Chosen because of its broad tokenizer coverage, including Ukrainian.

- **Tangentiality:**
  - $1 - \cos(\text{utterance}, \text{topic})$, where “topic” = session‑level centroid.
  - Implemented using EmbeddingGemma.

- **Coherence:**
  - $\cos(\text{utterance}_i,\; \text{utterance}_{i-1})$, $i>0$ (adjacent turns).
  - Also with EmbeddingGemma.

- **Pseudo‑Perplexity:**
  - Causal LM (teacher‑forcing) — `google/gemma‑3‑270m`.
  - Context windows: 256, 2, 5, 7 (principal measure PPL‑256).

---

## Evaluation Protocol

- **Turn‑level:** Pearson $r$, Spearman $\rho$, MAE, RMSE (EN vs UK).
- **Session‑level** (mean, variance): ICC(2,1), paired tests, mean difference %.
- **Success criteria:** $r/\rho \ge 0.7$; ICC $\ge 0.8$; mean difference ≤ ±5%.

---

## Results (Turn‑level)

| Metric             | Pearson r | Spearman ρ | MAE       | RMSE        |
|--------------------|----------:|-----------:|----------:|-------------:|
| Tangentiality      | **0.9739** | **0.9723** | **0.0127** | **0.0155**   |
| Coherence          | 0.2423     | 0.2030     | 0.0210    | 0.0212       |
| Pseudo‑Perplexity  | 0.0601     | 0.2630     | 90,596,920.0 | 193,752,500.0 |

---

## Results (Session‑level)

| Metric             | Summary | ICC(2,1) | Mean diff %      | p‑value (paired t) |
|---------------------|:-------:|---------:|------------------:|---------------------:|
| Tangentiality       | mean    | **0.9534** | **−1.369%**       | 1.12e−34            |
| Coherence           | mean    | 0.0056     | −2.220%           | 5.63e−236           |
| Pseudo‑Perplexity   | mean    | 0.0587     | **+398.24%**      | 1.73e−01            |

---

## Conclusion

- **Tangentiality** in the Gemma stack is validated: high cross‑language agreement (r/ρ > 0.97, ICC = 0.95, mean difference ≈ 1.37%). The criteria are met, so this metric is suitable for further analysis.  
- **Coherence and Pseudo‑Perplexity**, in the current implementation, do **not** meet the consistency criteria; especially PPL depends heavily on tokenization and frequency of subword units for Ukrainian.

## Supporting details from Gemma technical documentation

- Gemma‑3 uses the **SentencePiece tokenizer** with a vocabulary size of ≈ 262,000 units. [oai_citation:0‡Hugging Face](https://huggingface.co/blog/gemma3)  
- Documentation notes that tokenization has been refined for improved multilingual coverage, in particular to reduce the effect “non‑English word → many rare subwords” especially in morphologically rich languages. [oai_citation:1‡Hugging Face](https://huggingface.co/blog/gemma3)  
- Also in Gemma‑3 it is stated that when comparing perplexity across languages one must account for differences in subword segmentation, which affect **number of tokens** and probability distributions. [oai_citation:2‡arXiv](https://arxiv.org/html/2503.19786v1)  

> The transition to the Gemma stack (embeddings + causal LM) is justified: the model supports Ukrainian, shows strong consistency for the key metric (tangentiality), and can serve as a basis for further refinement of the remaining metrics.