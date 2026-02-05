# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import pandas as pd
import numpy as np
import string
import logging

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from lexicalrichness import LexicalRichness
import spacy
import traceback

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

FIRST_PERSON_PRONOUNS = ["i", "me", "my", "mine", "myself"]
FIRST_PERSON_PRONOUNS_T = {
    'en': {"i", "me", "my", "mine", "myself", "i'm", "i'll", "i'd", "i've"},
    # UA/UK: explicit first-person forms only (singular pronoun + singular possessive forms)
    'ua': {
        "я", "мене", "мені", "мною",
        "мій", "моя", "моє", "мої",
        "мого", "моєї", "моєму", "моїм", "моєю", "моїх", "моїми",
    },
    'uk': {
        "я", "мене", "мені", "мною",
        "мій", "моя", "моє", "мої",
        "мого", "моєї", "моєму", "моїм", "моєю", "моїх", "моїми",
    },
}
PRESENT = ["VBP", "VBZ"]
PAST = ["VBD", "VBN"]

from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, pipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import math
from typing import Dict, Tuple, Optional
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

class EngSentimentAnalyzer:
    """
    English sentiment analysis model: j-hartmann/sentiment-roberta-large-english-3-classes

    Provides:
      - raw_polarity_scores(text) -> {"negative","neutral","positive","compound"}  (compound=0.0 placeholder)
      - polarity_scores(text) -> {"neg","neu","pos","compound"}  (VADER-like)
      - vader_polarity_scores(text) -> alias for polarity_scores(text)
      - major_label(text) -> ("negative"|"neutral"|"positive", -1.0|0.0|1.0)
    """

    # This model already returns these string labels in the pipeline output.
    map_labels = {
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }

    int_label_map = {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0,
    }

    def __init__(
        self,
        device: int = -1,
        compound_scale: float = 1.0,
        compound_alpha: float = 15.0,  # VADER default alpha
        model_id: str = "j-hartmann/sentiment-roberta-large-english-3-classes",
    ):
        self.compound_scale = float(compound_scale)
        self.compound_alpha = float(compound_alpha)

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()

        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
            truncation=True,
        )

    def _vader_normalize(self, x: float) -> float:
        # VADER normalization: x / sqrt(x^2 + alpha)
        return x / math.sqrt(x * x + self.compound_alpha)

    def raw_polarity_scores(self, text: str) -> Dict[str, float]:
        """
        Returns the model's 3-class probabilities as:
        {"negative": p_neg, "neutral": p_neu, "positive": p_pos, "compound": 0.0}
        """
        results = self._pipe(text)  # list[list[{"label","score"}]]
        scores = {it["label"].lower(): float(it["score"]) for it in results[0]}

        neg = scores.get("negative", 0.0)
        neu = scores.get("neutral", 0.0)
        pos = scores.get("positive", 0.0)

        return {"negative": neg, "neutral": neu, "positive": pos, "compound": 0.0}

    # Backwards-compatible name vs your UA class
    def polarity_scores(self, text: str) -> Dict[str, float]:
        return self.raw_polarity_scores(text)

    def default_polarity_scores(self, text: str) -> Dict[str, float]:
        return self.raw_polarity_scores(text)

    def vader_polarity_scores(self, text: str) -> Dict[str, float]:
        """
        VADER-like output:
          - neg/neu/pos are proportions (sum ~ 1)
          - compound is a normalized (pos - neg) score in [-1, 1]
        """
        s = self.default_polarity_scores(text)
        pos = s["positive"]
        neg = s["negative"]
        neu = s["neutral"]

        denom = pos + neg + neu
        if denom > 0:
            pos_v = pos / denom
            neg_v = neg / denom
            neu_v = neu / denom
        else:
            pos_v, neg_v, neu_v = 0.0, 0.0, 1.0

        raw = self.compound_scale * (pos - neg)
        compound = float(self._vader_normalize(raw))

        return {"neg": float(neg_v), "neu": float(neu_v), "pos": float(pos_v), "compound": compound}

    # def vader_polarity_scores(self, text: str) -> Dict[str, float]:
    #     return self.polarity_scores(text)

    def major_label(self, text: str) -> Tuple[str, float]:
        """
        Returns the winning class label among {negative, neutral, positive}
        and its mapped numeric polarity (-1/0/1).
        """
        s = self.raw_polarity_scores(text)
        label = max(("negative", "neutral", "positive"), key=lambda k: s[k])
        return label, self.int_label_map[label]

class UkrSentimentAnalyzer:
    """
    Ukrainian sentiment analysis model: YShynkarov/ukr-roberta-cosmus-sentiment

    Provides:
      - polarity_scores(text) -> {"negative","neutral","positive","mixed"}
      - vader_polarity_scores(text) -> {"neg","neu","pos","compound"}  (VADER-like)
    """

    map_labels = {
        "LABEL_0": "mixed",
        "LABEL_1": "negative",
        "LABEL_2": "neutral",
        "LABEL_3": "positive",
    }

    int_label_map = {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0,
        "mixed": 0.0,
    }

    def __init__(
        self,
        device: int = -1,
        compound_scale: float = 1.0,
        compound_alpha: float = 15.0,   # VADER default alpha
        split_mixed: bool = True,
    ):
        self.compound_scale = float(compound_scale)
        self.compound_alpha = float(compound_alpha)
        self.split_mixed = bool(split_mixed)

        repo_id = "YShynkarov/ukr-roberta-cosmus-sentiment"
        safetensor = hf_hub_download(repo_id=repo_id, filename="ukrroberta_cosmus_sentiment.safetensors")

        config = RobertaConfig.from_pretrained("youscan/ukr-roberta-base", num_labels=4)
        tokenizer = RobertaTokenizer.from_pretrained("youscan/ukr-roberta-base")

        model = RobertaForSequenceClassification(config)
        state_dict = load_file(safetensor)
        model.load_state_dict(state_dict)
        model.eval()

        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            return_all_scores=True,
            truncation=True,
        )

    def old_polarity_scores(self, text: str) -> Dict[str, float]:
        results = self._pipe(text)  # list[list[{"label","score"}]]
        scores = {self.map_labels[it["label"]]: float(it["score"]) for it in results[0]}
        return {
            "negative": scores.get("negative", 0.0),
            "neutral": scores.get("neutral", 0.0),
            "positive": scores.get("positive", 0.0),
            "compound": scores.get("mixed", 0.0),
        }

    def _vader_normalize(self, x: float) -> float:
        # VADER normalization: x / sqrt(x^2 + alpha)
        return x / math.sqrt(x * x + self.compound_alpha)

    def default_polarity_scores(self, text: str) -> Dict[str, float]:
        return self.old_polarity_scores(text)

    def polarity_scores(self, text: str) -> Dict[str, float]:
        s = self.default_polarity_scores(text)
        pos = s["positive"]
        neg = s["negative"]
        neu = s["neutral"]
        mix = s["compound"]

        # VADER-like pos/neg/neu proportions
        if self.split_mixed:
            pos_v = pos + 0.5 * mix
            neg_v = neg + 0.5 * mix
        else:
            pos_v = pos
            neg_v = neg

        denom = pos_v + neg_v + neu
        if denom > 0:
            pos_v /= denom
            neg_v /= denom
            neu /= denom
        else:
            pos_v, neg_v, neu = 0.0, 0.0, 1.0

        raw = self.compound_scale * (pos - neg)
        compound = float(self._vader_normalize(raw))

        return {"neg": float(neg_v), "neu": float(neu), "pos": float(pos_v), "compound": compound}

    def major_label(self, text: str) -> Tuple[str, float]:
        scores = self.polarity_scores(text)
        best = max(scores, key=scores.get)
        return best, self.int_label_map[best]


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
    word_df[measures["first_person"]] = [
        True if word.lower() in FIRST_PERSON_PRONOUNS_T[lang] else np.nan
        for word, pos, _, _ in tag_list
    ]  # [word in FIRST_PERSON_PRONOUNS_T[lang] for word in word_list]
    # make non pronouns NaN (UA/UK: allow ADJ with Poss=Yes + Person=1)
    allowed_tags = {"Pronoun", "DET"}
    if lang in ['ua', 'uk']:
        allow_mask = [
            (pos in allowed_tags) or (pos == "Adjective" and poss_first_person)
            for _, pos, _, poss_first_person in tag_list
        ]
        word_df[measures["first_person"]] = word_df[measures["first_person"]].where(allow_mask, np.nan)
    else:
        word_df[measures["first_person"]] = word_df[measures["first_person"]].where(
            word_df[measures["part_of_speech"]].isin(allowed_tags), np.nan
        )
    # word_df[measures["first_person"]] = word_df[measures["first_person"]].where(word_df[measures["part_of_speech"]] == "Pronoun", np.nan)

    tag_list_verb = [
        verb_tense if pos == "Verb" else np.nan
        for _, pos, verb_tense, _ in tag_list
    ]  # ["Present" if tag[1] in PRESENT else "Past" if tag[1] in PAST else "Other" for tag in tag_list]
    word_df[measures["verb_tense"]] = tag_list_verb
    # make non verbs NaN
    word_df[measures["verb_tense"]] = word_df[measures["verb_tense"]].where(word_df[measures["part_of_speech"]] == "Verb", np.nan)

    return word_df

def calculate_first_person_sentiment(df, measures):
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
    
    res1 = []
    res2 = []
    for i in range(len(df)):
        perc = df.loc[i, measures["first_person_percentage"]]
        pos = df.loc[i, measures["pos"]]
        neg = df.loc[i, measures["neg"]]

        if perc is np.nan:
            res1.append(np.nan)
            res2.append(np.nan)
            continue

        res1.append((100-perc)*pos)
        res2.append(perc*neg)

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

    summ_df[measures["first_person_percentage"]] = calculate_first_person_percentage(full_text, lang=lang)
    try:
        if len(turn_df) > 0:
            summ_df[measures["first_person_sentiment_positive"]] = turn_df[measures["first_person_sentiment_positive"]].mean(skipna=True)
            summ_df[measures["first_person_sentiment_negative"]] = turn_df[measures["first_person_sentiment_negative"]].mean(skipna=True)

            first_person_sentiment = []
            for i in range(len(turn_df)):
                if turn_df.loc[i, measures["pos"]] > turn_df.loc[i, measures["neg"]]:
                    first_person_sentiment.append(turn_df.loc[i, measures["first_person_sentiment_positive"]])
                else:    
                    first_person_sentiment.append(turn_df.loc[i, measures["first_person_sentiment_negative"]])

            summ_df[measures["first_person_sentiment_overall"]] = np.nanmean(first_person_sentiment)
        else:
            first_pos, first_neg = calculate_first_person_sentiment(summ_df, measures)
            summ_df[measures["first_person_sentiment_positive"]] = first_pos
            summ_df[measures["first_person_sentiment_negative"]] = first_neg

            if summ_df[measures["pos"]].values[0] > summ_df[measures["neg"]].values[0]:
                summ_df[measures["first_person_sentiment_overall"]] = summ_df[measures["first_person_sentiment_positive"]].values[0]
            else:
                summ_df[measures["first_person_sentiment_overall"]] = summ_df[measures["first_person_sentiment_negative"]].values[0]

        return summ_df
    except: 
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
    Returns a list of tuples (text, pos, verb_tense, poss_first_person).
    If given a list, produce exactly one tag per input element to keep
    alignment with word_df rows.
    """
    if lang in ['ua', 'uk']:
        nlp = spacy.load("uk_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")

    def _is_possessive_first_person(token, lang_code):
        if token is None or lang_code not in ['ua', 'uk']:
            return False
        poss_vals = token.morph.get("Poss")
        person_vals = token.morph.get("Person")
        if not poss_vals or not person_vals:
            return False
        return ("Yes" in poss_vals) and ("1" in person_vals)

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
                poss_first_person = False
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
                poss_first_person = _is_possessive_first_person(token, lang)
            tags.append((text, pos, verb_tense, poss_first_person))
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
        poss_first_person = _is_possessive_first_person(token, lang)
        tags.append((token.text, pos, verb_tense, poss_first_person))

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

    This function calculates the sentiment scores of the input text using
     VADER, and adds them to the output dataframe summ_df.

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
        # SentimentIntensityAnalyzer()
        sentiment = EngSentimentAnalyzer(compound_scale=4.0, compound_alpha=15.0) if lang == 'en' else UkrSentimentAnalyzer(compound_scale=4.0, compound_alpha=15.0, split_mixed=True)
        cols = [measures["neg"], measures["neu"], measures["pos"], measures["compound"], measures["speech_mattr_5"], measures["speech_mattr_10"], measures["speech_mattr_25"], measures["speech_mattr_50"], measures["speech_mattr_100"]]

        for idx, u in enumerate(turn_list):
            try:
                sentiment_dict = sentiment.polarity_scores(u)
                mattrs = [get_mattr(u, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]
                turn_df.loc[idx, cols] = list(sentiment_dict.values()) + mattrs

            except Exception as e:
                logger.info(f"Error in sentiment analysis: {e}")
                continue

        sentiment_dict = sentiment.polarity_scores(full_text)
        mattrs = [get_mattr(full_text, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]

        summ_df.loc[0, cols] = list(sentiment_dict.values()) + mattrs
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
