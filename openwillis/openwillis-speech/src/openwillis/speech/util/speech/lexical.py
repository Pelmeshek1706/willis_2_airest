# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import pandas as pd
import numpy as np
import string
import logging
from functools import lru_cache
from typing import Dict, List, Optional

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness
import spacy
import traceback
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# NLTK Tag list
TAG_DICT = {"PRP": "Pronoun", "PRP$": "Pronoun", "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", 
            "VBZ": "Verb", "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective", "NN": "Noun", "NNP": "Noun", "NNS": "Noun",
            "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb", "DT": "Determiner"}

TAG_DICT_T = {
    "ua" : {
    "CCONJ": "Conjunction",
    "PRON": "Pronoun",
    "NOUN": "Noun",
    "ADJ": "Adjective",
    "PUNCT": "Punctuation",
    "VERB": "Verb",
    "AUX": "Auxiliary",
    "ADV": "Adverb",
    "ADP": "Adposition",
    "SCONJ": "Subordinating Conjunction",
    "NUM": "Numeral",
    "PROPN": "Proper Noun"
},
    "uk" : {
    "CCONJ": "Conjunction",
    "PRON": "Pronoun",
    "NOUN": "Noun",
    "ADJ": "Adjective",
    "PUNCT": "Punctuation",
    "VERB": "Verb",
    "AUX": "Auxiliary",
    "ADV": "Adverb",
    "ADP": "Adposition",
    "SCONJ": "Subordinating Conjunction",
    "NUM": "Numeral",
    "PROPN": "Proper Noun"
},
    'en' : {
    "PRP": "Pronoun",
    "PRP$": "Pronoun",
    "VB": "Verb",
    "VBD": "Verb",
    "VBG": "Verb",
    "VBN": "Verb",
    "VBP": "Verb",
    "VBZ": "Verb",
    "JJ": "Adjective",
    "JJR": "Adjective",
    "JJS": "Adjective",
    "NN": "Noun",
    "NNP": "Noun",
    "NNS": "Noun",
    "RB": "Adverb",
    "RBR": "Adverb",
    "RBS": "Adverb",
    "DT": "Determiner"}
}

FIRST_PERSON_PRONOUNS = ["I", "me", "my", "mine", "myself"]
FIRST_PERSON_PRONOUNS_T = {'en' : {"I", "me", "my", "mine", "myself"},
                           'ua' : {"я", "мене", "мені", "мною", "мій", "моя", "мої", "моє"},
                           'uk' : {"я", "мене", "мені", "мною", "мій", "моя", "мої", "моє"},}
PRESENT = ["VBP", "VBZ"]
PAST = ["VBD", "VBN"]

VADER_SENTIMENT_COLS = {
    "neg": "sentiment_neg_vader",
    "neu": "sentiment_neu_vader",
    "pos": "sentiment_pos_vader",
    "compound": "sentiment_overall_vader",
}

FIRST_PERSON_VADER_COLS = {
    "positive": "first_person_sentiment_positive_vader",
    "negative": "first_person_sentiment_negative_vader",
    "overall": "first_person_sentiment_overall_vader",
}


class MultilingualSentiment:
    """
    Multilingual sentiment analyzer with sliding-window support for long texts.
    polarity_scores(text) -> {"neg": float, "neu": float, "pos": float, "compound": float}
    """

    def __init__(
        self,
        model_id: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        *,
        window_tokens: Optional[int] = None,
        overlap_tokens: int = 64,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self._pipe = pipeline(
            "sentiment-analysis",
            model=model_id,
            tokenizer=model_id,
            top_k=None,
        )
        self.tokenizer = self._pipe.tokenizer
        self.model = self._pipe.model

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.window_tokens = window_tokens
        self.overlap_tokens = overlap_tokens
        self.batch_size = batch_size

    @staticmethod
    def _label_index_map(model) -> Dict[str, int]:
        id2label = getattr(model.config, "id2label", None) or {}
        mapping: Dict[str, int] = {}

        for idx, label in id2label.items():
            label_name = str(label).lower()
            if "neg" in label_name:
                mapping["neg"] = int(idx)
            elif "neu" in label_name:
                mapping["neu"] = int(idx)
            elif "pos" in label_name:
                mapping["pos"] = int(idx)

        if set(mapping.keys()) != {"neg", "neu", "pos"}:
            mapping = {"neg": 0, "neu": 1, "pos": 2}

        return mapping

    @staticmethod
    def _vader_normalize(score: float, alpha: float = 15.0) -> float:
        normalized = score / ((score * score + alpha) ** 0.5)
        if normalized < -1.0:
            return -1.0
        if normalized > 1.0:
            return 1.0
        return normalized

    @torch.inference_mode()
    def polarity_scores(self, text: str) -> dict:
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        if not text.strip():
            return {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}

        window_tokens = self.window_tokens
        if window_tokens is None:
            window_tokens = int(getattr(self.tokenizer, "model_max_length", 512))
            if window_tokens > 2048:
                window_tokens = 512

        if self.overlap_tokens >= window_tokens:
            raise ValueError("overlap_tokens must be < window_tokens")

        enc = self.tokenizer(
            text,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=window_tokens,
            stride=self.overlap_tokens,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        idx_map = self._label_index_map(self.model)

        probs_list: List[torch.Tensor] = []
        weights_list: List[torch.Tensor] = []
        n = input_ids.size(0)

        for i in range(0, n, self.batch_size):
            ids_batch = input_ids[i : i + self.batch_size]
            mask_batch = attention_mask[i : i + self.batch_size]

            logits = self.model(input_ids=ids_batch, attention_mask=mask_batch).logits
            probs = torch.softmax(logits, dim=-1)

            probs_list.append(probs)
            weights_list.append(mask_batch.sum(dim=1).float().clamp_min(1.0))

        probs_all = torch.cat(probs_list, dim=0)
        weights = torch.cat(weights_list, dim=0)
        avg = (probs_all * weights.unsqueeze(1)).sum(dim=0) / weights.sum()

        neg = float(avg[idx_map["neg"]].item())
        neu = float(avg[idx_map["neu"]].item())
        pos = float(avg[idx_map["pos"]].item())
        compound = self._vader_normalize(pos - neg, alpha=15.0)

        return {
            "neg": neg,
            "neu": neu,
            "pos": pos,
            "compound": compound,
        }


@lru_cache(maxsize=1)
def get_multilingual_sentiment_analyzer() -> MultilingualSentiment:
    return MultilingualSentiment()


@lru_cache(maxsize=1)
def get_vader_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


def _sentiment_values(scores: dict) -> List[float]:
    return [
        scores.get("neg", np.nan),
        scores.get("neu", np.nan),
        scores.get("pos", np.nan),
        scores.get("compound", np.nan),
    ]


def get_mattr(text, lemmatizer, window_size=50):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the Moving Average Type-Token Ratio (MATTR)
     of the input text using the
     LexicalRichness library.

    Parameters:
    ...........
    text : str
        The input text to be analyzed.
    lemmatizer : spacy lemmatizer
        The lemmatizer to be used in the calculation.
    window_size : int
        The size of the window to be used in the calculation.

    Returns:
    ...........
    mattr : float
        The calculated MATTR value.

    ------------------------------------------------------------------------------------------------------
    """

    words = nltk.word_tokenize(text) # [list of words] can be ua, so not specific to en 
    words = [w.translate(str.maketrans('', '', string.punctuation)).lower() for w in words]
    words = [w for w in words if w != '']
    words_texts = [token.lemma_ for token in lemmatizer(' '.join(words))]
    filter_punc = " ".join(words_texts)
    mattr = np.nan

    lex_richness = LexicalRichness(filter_punc)
    if lex_richness.words > 0:
        mattr = lex_richness.mattr(window_size=min(window_size, lex_richness.words))

    return mattr

def get_tag(word_df, word_list, measures, lang = 'en'):
    """
    ------------------------------------------------------------------------------------------------------

    This function performs part-of-speech
     tagging on the input text using NLTK, and returns
     word-level part-of-speech tags.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information.
    word_list: list
        List of transcribed text at the word level.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_df: pandas dataframe
        The updated word_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    # tag_list = nltk.pos_tag(word_list)
    tag_list = get_tag_l(word_list, lang=lang)
    
    # tag_list_pos = [TAG_DICT_T[lang][tag[1]] if tag[1] in TAG_DICT_T[lang].keys() else "Other" for tag in tag_list] # change TAG_LIST
    tag_list_pos = [tag[1] for tag in tag_list]
    word_df[measures["part_of_speech"]] = tag_list_pos
    # words['first_person']
    word_df[measures["first_person"]] = [True if word.lower() in FIRST_PERSON_PRONOUNS_T[lang] else np.nan for word, pos, _ in tag_list]# [word in FIRST_PERSON_PRONOUNS_T[lang] for word in word_list]
    # make non pronouns NaN
    allowed_tags = ["Pronoun", "DET"]
    word_df[measures["first_person"]] = word_df[measures["first_person"]].where(word_df[measures["part_of_speech"]].isin(allowed_tags), np.nan)
    # word_df[measures["first_person"]] = word_df[measures["first_person"]].where(word_df[measures["part_of_speech"]] == "Pronoun", np.nan)

    tag_list_verb = [verb_tense if pos == "Verb" else np.nan for _, pos, verb_tense in tag_list] # ["Present" if tag[1] in PRESENT else "Past" if tag[1] in PAST else "Other" for tag in tag_list]
    word_df[measures["verb_tense"]] = tag_list_verb
    # make non verbs NaN
    word_df[measures["verb_tense"]] = word_df[measures["verb_tense"]].where(word_df[measures["part_of_speech"]] == "Verb", np.nan)

    return word_df

def calculate_first_person_sentiment(
    df,
    measures,
    *,
    first_person_col: Optional[str] = None,
    pos_col: Optional[str] = None,
    neg_col: Optional[str] = None,
):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates a measure of the influence of sentiment on the use of first person pronouns.

    Parameters:
    ...........
    df: pandas dataframe
        A dataframe containing summary information.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    res1: list
        A list containing the calculated measure of the influence of positive sentiment on the use of first person pronouns.
    res2: list
        A list containing the calculated measure of the influence of negative sentiment on the use of first person pronouns.

    ------------------------------------------------------------------------------------------------------
    """
    
    first_person_col = first_person_col or measures["first_person_percentage"]
    pos_col = pos_col or measures["pos"]
    neg_col = neg_col or measures["neg"]

    res1: List[float] = []
    res2: List[float] = []
    for _, row in df.iterrows():
        perc = row.get(first_person_col, np.nan)
        pos = row.get(pos_col, np.nan)
        neg = row.get(neg_col, np.nan)

        if pd.isna(perc) or pd.isna(pos) or pd.isna(neg):
            res1.append(np.nan)
            res2.append(np.nan)
            continue

        res1.append((100 - perc) * pos)
        res2.append(perc * neg)

    return res1, res2

# def calculate_first_person_percentage(text, lang = 'en'):
#     """
#     ------------------------------------------------------------------------------------------------------

#     This function calculates the percentage of first person pronouns in the input text.

#     Parameters:
#     ...........
#     text: str
#         The input text to be analyzed.

#     Returns:
#     ...........
#     float
#         The calculated percentage of first person pronouns in the input text.

#     ------------------------------------------------------------------------------------------------------
#     """
#     lang = 'ua'
#     words = nltk.word_tokenize(text)
#     tags = nltk.pos_tag(words)
#     # filter out non pronouns
#     pronouns = [tag[0] for tag in tags if tag[1] == "PRP" or tag[1] == "PRP$"]
#     if len(pronouns) == 0:
#         return np.nan

#     first_person_pronouns = len([word for word in pronouns if word in FIRST_PERSON_PRONOUNS_T[lang]])
#     return (first_person_pronouns / len(pronouns)) * 100

def calculate_first_person_percentage(text, lang='en'):
    """
    Calculates the percentage of first person pronouns in the input text.
    
    Parameters:
        text (str): The input text to be analyzed.
        lang (str): Language code ('en' for English, 'ua' or 'uk' for Ukrainian).
    
    Returns:
        float: The percentage of first person pronouns in the text, or np.nan if no tokens.
    """
    if lang in ['ua', 'uk']:
        nlp = spacy.load("uk_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")
    
    # Обработка текста
    doc = nlp(text)
    total_tokens = len(doc)
    if total_tokens == 0:
        return np.nan

    first_person_count = sum(1 for token in doc if token.text.lower() in FIRST_PERSON_PRONOUNS_T[lang])
    
    return (first_person_count / total_tokens) * 100

def get_first_person_turn(turn_df, turn_list, measures, lang = 'en'):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates measures related to the first person pronouns in each turn.
     Specifically, it calculates the percentage of first person pronouns in each turn,
     and the influence of sentiment on the use of first person pronouns.

    Parameters:
    ...........
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    turn_list: list
        List of transcribed text at the turn level.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    first_person_percentages = [calculate_first_person_percentage(turn, lang) for turn in turn_list]

    turn_df[measures["first_person_percentage"]] = first_person_percentages

    first_pos, first_neg = calculate_first_person_sentiment(turn_df, measures)

    turn_df[measures["first_person_sentiment_positive"]] = first_pos
    turn_df[measures["first_person_sentiment_negative"]] = first_neg

    if VADER_SENTIMENT_COLS["pos"] in turn_df.columns and VADER_SENTIMENT_COLS["neg"] in turn_df.columns:
        vader_pos, vader_neg = calculate_first_person_sentiment(
            turn_df,
            measures,
            pos_col=VADER_SENTIMENT_COLS["pos"],
            neg_col=VADER_SENTIMENT_COLS["neg"],
        )
        turn_df[FIRST_PERSON_VADER_COLS["positive"]] = vader_pos
        turn_df[FIRST_PERSON_VADER_COLS["negative"]] = vader_neg

    return turn_df

def get_first_person_summ(summ_df, turn_df, full_text, measures, lang='en'):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates measures related to the first person pronouns in the transcript.

    Parameters:
    ...........
    summ_df: pandas dataframe
        A dataframe containing summary information.
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    full_text: str
        The full transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    def _compute_first_person_summary(
        pos_col: str,
        neg_col: str,
        out_pos_col: str,
        out_neg_col: str,
        out_overall_col: str,
    ) -> None:
        if len(turn_df) > 0:
            if out_pos_col not in turn_df.columns or out_neg_col not in turn_df.columns:
                fp_pos, fp_neg = calculate_first_person_sentiment(
                    turn_df,
                    measures,
                    pos_col=pos_col,
                    neg_col=neg_col,
                )
                turn_df[out_pos_col] = fp_pos
                turn_df[out_neg_col] = fp_neg

            summ_df[out_pos_col] = turn_df[out_pos_col].mean(skipna=True)
            summ_df[out_neg_col] = turn_df[out_neg_col].mean(skipna=True)

            first_person_sentiment = []
            for _, row in turn_df.iterrows():
                pos_val = row.get(pos_col, np.nan)
                neg_val = row.get(neg_col, np.nan)
                if pd.isna(pos_val) or pd.isna(neg_val):
                    first_person_sentiment.append(np.nan)
                elif pos_val > neg_val:
                    first_person_sentiment.append(row.get(out_pos_col, np.nan))
                else:
                    first_person_sentiment.append(row.get(out_neg_col, np.nan))

            summ_df[out_overall_col] = np.nanmean(first_person_sentiment)
        else:
            fp_pos, fp_neg = calculate_first_person_sentiment(
                summ_df,
                measures,
                pos_col=pos_col,
                neg_col=neg_col,
            )
            summ_df[out_pos_col] = fp_pos
            summ_df[out_neg_col] = fp_neg

            pos_val = summ_df[pos_col].iloc[0] if pos_col in summ_df.columns else np.nan
            neg_val = summ_df[neg_col].iloc[0] if neg_col in summ_df.columns else np.nan
            if pd.isna(pos_val) or pd.isna(neg_val):
                summ_df[out_overall_col] = np.nan
            elif pos_val > neg_val:
                summ_df[out_overall_col] = summ_df[out_pos_col].iloc[0]
            else:
                summ_df[out_overall_col] = summ_df[out_neg_col].iloc[0]

    summ_df[measures["first_person_percentage"]] = calculate_first_person_percentage(full_text, lang=lang)
    try:
        _compute_first_person_summary(
            measures["pos"],
            measures["neg"],
            measures["first_person_sentiment_positive"],
            measures["first_person_sentiment_negative"],
            measures["first_person_sentiment_overall"],
        )

        has_vader_scores = (
            (VADER_SENTIMENT_COLS["pos"] in turn_df.columns and VADER_SENTIMENT_COLS["neg"] in turn_df.columns)
            or (VADER_SENTIMENT_COLS["pos"] in summ_df.columns and VADER_SENTIMENT_COLS["neg"] in summ_df.columns)
        )
        if has_vader_scores:
            _compute_first_person_summary(
                VADER_SENTIMENT_COLS["pos"],
                VADER_SENTIMENT_COLS["neg"],
                FIRST_PERSON_VADER_COLS["positive"],
                FIRST_PERSON_VADER_COLS["negative"],
                FIRST_PERSON_VADER_COLS["overall"],
            )

        return summ_df
    except Exception:
        print("exception")
        print(traceback.format_exc())


# def get_tag_l(full_text, lang='en'):
#     nlp = spacy.load("uk_core_news_sm") if (lang in ['uk', 'ua']) else spacy.load("en_core_web_sm")
#     if type(full_text) == list:
#         full_text = " ".join(full_text)
#     doc = nlp(full_text)
    
#     # Get the original tags and map them using our dictionary if available
#     pos_tags = [(token.text, TAG_DICT_T[lang].get(token.tag_, token.tag_)) for token in doc]
#     return pos_tags

def count_space_tokens(text, lang='en'):
    if lang in ['ua', 'uk']:
        nlp = spacy.load("uk_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    return sum(1 for token in doc if token.pos_ == "SPACE")


def get_tag_l(full_text, lang='en'):
    """
    Returns a list of tuples (text, pos, verb_tense).
    If given a list, produce exactly one tag per input element to keep
    alignment with word_df rows.
    """
    if lang in ['ua', 'uk']:
        nlp = spacy.load("uk_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")

    tags = []

    # Maintain 1:1 alignment when a list of words is provided
    if isinstance(full_text, list):
        for w in full_text:
            # Process each token individually to avoid spaCy retokenization expanding counts
            doc = nlp(w if isinstance(w, str) else str(w))
            # Pick first non-space token if available, else fall back to empty
            token = next((t for t in doc if t.pos_ != "SPACE"), None)
            if token is None:
                # No token produced (e.g., empty/space-only). Mark as Other/None keeping alignment
                pos = "Other"
                verb_tense = None
                text = w
            else:
                text = token.text
                if lang in ['ua', 'uk']:
                    pos = TAG_DICT_T[lang].get(token.pos_, token.pos_)
                    if token.pos_ in {"VERB", "AUX"}:
                        tense_vals = token.morph.get("Tense")
                        if tense_vals:
                            if "Past" in tense_vals:
                                verb_tense = "Past"
                            elif "Pres" in tense_vals:
                                verb_tense = "Present"
                            else:
                                verb_tense = "Other"
                        else:
                            verb_tense = "Other"
                    else:
                        verb_tense = None
                else:
                    pos = TAG_DICT_T[lang].get(token.tag_, token.tag_)
                    if token.tag_ in PRESENT:
                        verb_tense = "Present"
                    elif token.tag_ in PAST:
                        verb_tense = "Past"
                    else:
                        verb_tense = "Other"
            tags.append((text, pos, verb_tense))
        return tags

    # If given a single string, process as a whole and filter out SPACE tokens
    doc = nlp(full_text)
    for token in doc:
        if token.pos_ == "SPACE":
            continue
        if lang in ['ua', 'uk']:
            pos = TAG_DICT_T[lang].get(token.pos_, token.pos_)
            if token.pos_ in {"VERB", "AUX"}:
                tense_vals = token.morph.get("Tense")
                if tense_vals:
                    if "Past" in tense_vals:
                        verb_tense = "Past"
                    elif "Pres" in tense_vals:
                        verb_tense = "Present"
                    else:
                        verb_tense = "Other"
                else:
                    verb_tense = "Other"
            else:
                verb_tense = None
        else:
            pos = TAG_DICT_T[lang].get(token.tag_, token.tag_)
            if token.tag_ in PRESENT:
                verb_tense = "Present"
            elif token.tag_ in PAST:
                verb_tense = "Past"
            else:
                verb_tense = "Other"
        tags.append((token.text, pos, verb_tense))

    return tags

def get_pos_tag(df_list, text_list, measures, lang="en"):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the part-of-speech measures
        and adds them to the output dataframes

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    text_list: list
        List of transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    try:
        word_df, turn_df, summ_df = df_list
        word_list, turn_list, full_text = text_list

        word_df = get_tag(word_df, word_list, measures, lang=lang)

        if len(turn_list) > 0:
            turn_df = get_first_person_turn(turn_df, turn_list, measures, lang=lang)

        summ_df = get_first_person_summ(summ_df, turn_df, full_text, measures, lang)

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in pos tag feature calculation: {e}")
    finally:
        return df_list

def get_sentiment(df_list, text_list, measures, lang='en'):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates sentiment scores of the input text using
     multilingual XLM-R (main output columns) and VADER (extra *_vader columns).

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    text_list: list
        List of transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.

    ------------------------------------------------------------------------------------------------------
    """
    try:
        word_df, turn_df, summ_df = df_list
        _, turn_list, full_text = text_list
        lemmatizer = spacy.load("uk_core_news_sm") if lang in ['ua', 'uk'] else spacy.load('en_core_web_sm') # should be changed to normal model
        
        sentiment = get_multilingual_sentiment_analyzer()
        vader_sentiment = get_vader_sentiment_analyzer()
        cols = [measures["neg"], measures["neu"], measures["pos"], measures["compound"], measures["speech_mattr_5"], measures["speech_mattr_10"], measures["speech_mattr_25"], measures["speech_mattr_50"], measures["speech_mattr_100"]]
        vader_cols = [
            VADER_SENTIMENT_COLS["neg"],
            VADER_SENTIMENT_COLS["neu"],
            VADER_SENTIMENT_COLS["pos"],
            VADER_SENTIMENT_COLS["compound"],
        ]

        for idx, u in enumerate(turn_list):
            try:
                sentiment_dict = sentiment.polarity_scores(u)
                vader_dict = vader_sentiment.polarity_scores(u)
                mattrs = [get_mattr(u, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]
                turn_df.loc[idx, cols] = _sentiment_values(sentiment_dict) + mattrs
                turn_df.loc[idx, vader_cols] = _sentiment_values(vader_dict)

            except Exception as e:
                logger.info(f"Error in sentiment analysis: {e}")
                continue

        sentiment_dict = sentiment.polarity_scores(full_text)
        vader_dict = vader_sentiment.polarity_scores(full_text)
        mattrs = [get_mattr(full_text, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]

        summ_df.loc[0, cols] = _sentiment_values(sentiment_dict) + mattrs
        summ_df.loc[0, vader_cols] = _sentiment_values(vader_dict)
        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in sentiment feature calculation: {e}")
    finally:
        return df_list

def calculate_repetitions(words_texts, phrases_texts):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the percentage of repeated words and phrases in the input lists.

    Parameters:
    ...........
    words_texts: list
        List of transcribed text at the word level.
    phrases_texts: list
        List of transcribed text at the phrase level.

    Returns:
    ...........
    word_reps_perc: float
        The percentage of repeated words in the input lists.
    phrase_reps_perc: float
        The percentage of repeated phrases in the input lists.

    ------------------------------------------------------------------------------------------------------
    """
    def calculate_percentage_repetitions(text_list, window_size):
        """Helper function to calculate the percentage of repetitions in a sliding window."""
        if len(text_list) <= window_size:
            reps = len(text_list) - len(set(text_list))
            return 100 * reps / len(text_list) if len(text_list) > 0 else 0
        else:
            reps_list = [
                100 * (len(window) - len(set(window))) / len(window)
                for i in range(len(text_list) - window_size + 1)
                for window in [text_list[i:i + window_size]]
            ]
            return np.mean(reps_list)

    # Clean words and phrases: remove punctuation, convert to lowercase, and filter out empty strings
    words_texts = [word.translate(str.maketrans('', '', string.punctuation)).lower() for word in words_texts if word.strip()]
    phrases_texts = [phrase.translate(str.maketrans('', '', string.punctuation)).lower() for phrase in phrases_texts if phrase.strip()]

    # Calculate repetition percentages for words (sliding window of 10 words) and phrases (sliding window of 3 phrases)
    word_reps_perc = calculate_percentage_repetitions(words_texts, window_size=10)
    phrase_reps_perc = calculate_percentage_repetitions(phrases_texts, window_size=3) if phrases_texts else np.nan

    return word_reps_perc, phrase_reps_perc

def get_repetitions(df_list, utterances_speaker, utterances_speaker_filtered, measures):
    """
    This function calculates the percentage of repeated words and phrases in the input text
    and adds them to the output dataframes.

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    utterances_speaker_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker
        after filtering out turns with less than min_turn_length words.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    df_list: list
        List of updated pandas dataframes.
    """
    
    try:
        word_df, turn_df, summ_df = df_list

        # turn-level
        if not turn_df.empty:
            for i in range(len(utterances_speaker_filtered)):
                row = utterances_speaker_filtered.iloc[i]
                words_texts = row[measures['words_texts']]
                phrases_texts = row[measures['phrases_texts']]

                word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

                turn_df.loc[i, measures['word_repeat_percentage']] = word_reps_perc
                turn_df.loc[i, measures['phrase_repeat_percentage']] = phrase_reps_perc

            # Calculate summary-level statistics
            summ_df[measures['word_repeat_percentage']] = turn_df[measures['word_repeat_percentage']].mean(skipna=True)
            summ_df[measures['phrase_repeat_percentage']] = turn_df[measures['phrase_repeat_percentage']].mean(skipna=True)
        else:
            words_texts = [word for words in utterances_speaker[measures['words_texts']] for word in words]
            phrases_texts = [phrase for phrases in utterances_speaker[measures['phrases_texts']] for phrase in phrases]

            word_reps_perc, phrase_reps_perc = calculate_repetitions(words_texts, phrases_texts)

            summ_df[measures['word_repeat_percentage']] = word_reps_perc
            summ_df[measures['phrase_repeat_percentage']] = phrase_reps_perc

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in calculating repetitions: {e}")
    finally:
        return df_list
