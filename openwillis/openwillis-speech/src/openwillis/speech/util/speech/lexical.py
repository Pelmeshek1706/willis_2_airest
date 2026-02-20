# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import pandas as pd
import numpy as np
import string
import logging

import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # eng
from .ukrainian_vader import UkrainianSentimentIntensityAnalyzerImproved # ua/uk
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

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F


import math
from typing import Dict, Tuple, List, Any

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def _normalize_pipeline_scores(pipe_out: Any) -> List[Dict[str, Any]]:
    """
    Normalize HF text-classification outputs to a list of {"label","score"} dicts.
    Handles single vs batch shapes and return_all_scores variations.
    """
    if pipe_out is None:
        return []

    if isinstance(pipe_out, dict):
        if "label" in pipe_out and "score" in pipe_out:
            return [pipe_out]
        # If already a label->score mapping, convert it.
        return [
            {"label": k, "score": v}
            for k, v in pipe_out.items()
            if isinstance(v, (int, float))
        ]

    if isinstance(pipe_out, list):
        if not pipe_out:
            return []
        first = pipe_out[0]
        if isinstance(first, list):
            return _normalize_pipeline_scores(first)
        if isinstance(first, dict):
            return pipe_out

    return []


class EngSentimentAnalyzer:
    """
    English sentiment analysis model: j-hartmann/sentiment-roberta-large-english-3-classes

    Provides:
      - raw_polarity_scores(text) -> {"negative","neutral","positive","compound"}  (compound=0.0 placeholder)
      - polarity_scores(text) -> {"negative","neutral","positive","compound"}  (alias)
      - vader_polarity_scores(text) -> {"neg","neu","pos","compound"}  (VADER-like)
      - major_label(text) -> ("negative"|"neutral"|"positive", -1.0|0.0|1.0)

    NOTE: This implementation auto-handles long texts via sliding-window aggregation.
    """

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
        # long-text settings (can be overridden per-call too)
        max_length: int = 512,
        stride: int = 128,
        min_chunk_tokens: int = 16,
    ):
        self.compound_scale = float(compound_scale)
        self.compound_alpha = float(compound_alpha)

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()

        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=self._tokenizer,
            device=device,
            return_all_scores=True,
            truncation=False,  # keep False; we handle long texts ourselves
        )

        self._max_length = int(max_length)
        self._stride = int(stride)
        self._min_chunk_tokens = int(min_chunk_tokens)

    def _vader_normalize(self, x: float) -> float:
        return x / math.sqrt(x * x + self.compound_alpha)

    # -------------------------
    # Sliding-window long-text
    # -------------------------
    def _iter_token_windows(
        self,
        input_ids: List[int],
        window_tokens: int,
        stride: int,
        min_chunk_tokens: int,
    ) -> List[Tuple[int, int]]:
        """
        Returns [(start, end), ...] over token ids (no special tokens).
        window_tokens: max tokens per chunk (excluding special tokens).
        stride: overlap in tokens between consecutive chunks.
        """
        n = len(input_ids)
        if n == 0:
            return [(0, 0)]

        window_tokens = max(1, int(window_tokens))
        stride = max(0, int(stride))
        step = max(1, window_tokens - stride)

        spans: List[Tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + window_tokens, n)
            if (end - start) >= min_chunk_tokens or start == 0:
                spans.append((start, end))
            if end >= n:
                break
            start += step

        # Ensure at least one span
        if not spans:
            spans = [(0, min(window_tokens, n))]
        return spans

    def _half_overlap_weights(self, spans: List[Tuple[int, int]]) -> List[float]:
        """
        Weight each chunk by its non-overlapped contribution using half-overlap rule:
          weight = len(chunk) - 0.5*overlap_with_prev - 0.5*overlap_with_next
        This reduces double counting when stride>0.
        """
        if not spans:
            return []

        weights: List[float] = []
        for i, (s, e) in enumerate(spans):
            length = float(max(0, e - s))
            overlap_prev = 0.0
            overlap_next = 0.0

            if i > 0:
                ps, pe = spans[i - 1]
                overlap_prev = float(max(0, min(pe, e) - max(ps, s)))
            if i + 1 < len(spans):
                ns, ne = spans[i + 1]
                overlap_next = float(max(0, min(ne, e) - max(ns, s)))

            w = length - 0.5 * overlap_prev - 0.5 * overlap_next
            weights.append(max(0.0, w))

        # Fallback: if all weights are zero (can happen in degenerate cases), use uniform weights
        if sum(weights) <= 0.0:
            return [1.0] * len(spans)
        return weights

    def raw_polarity_scores_sliding(
        self,
        text: str,
        *,
        max_length: int | None = None,
        stride: int | None = None,
        min_chunk_tokens: int | None = None,
    ) -> Dict[str, float]:
        """
        Long-text handling: tokenize without special tokens, split into overlapping windows,
        run the classifier on each chunk, then aggregate probabilities with overlap-aware weights.
        """
        if text is None or not str(text).strip():
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "compound": 0.0}

        tok = self._tokenizer

        # Model max length is typically 512; we reserve space for special tokens added by the pipeline.
        max_len = int(max_length if max_length is not None else self._max_length)
        special = int(tok.num_special_tokens_to_add(pair=False))
        window_tokens = max(1, max_len - special)

        stride_val = int(stride if stride is not None else self._stride)
        min_tokens = int(min_chunk_tokens if min_chunk_tokens is not None else self._min_chunk_tokens)

        # Tokenize full text (no special tokens) so we can window precisely
        input_ids = tok.encode(text, add_special_tokens=False)

        # Build spans + decode each chunk back to text for the pipeline
        spans = self._iter_token_windows(
            input_ids=input_ids,
            window_tokens=window_tokens,
            stride=stride_val,
            min_chunk_tokens=min_tokens,
        )
        chunk_texts: List[str] = []
        for s, e in spans:
            chunk_ids = input_ids[s:e]
            chunk_texts.append(
                tok.decode(
                    chunk_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )

        # Batch inference (pipeline supports list input)
        results = self._pipe(chunk_texts)  # -> list[list[{"label","score"}]]

        # Aggregate with overlap-aware weights
        weights = self._half_overlap_weights(spans)
        wsum = float(sum(weights))

        neg_sum = 0.0
        neu_sum = 0.0
        pos_sum = 0.0

        for w, res in zip(weights, results):
            items = _normalize_pipeline_scores(res)
            scores = {
                str(it.get("label", "")).lower(): float(it["score"])
                for it in items
                if isinstance(it, dict) and "label" in it and "score" in it
            }
            neg_sum += w * scores.get("negative", 0.0)
            neu_sum += w * scores.get("neutral", 0.0)
            pos_sum += w * scores.get("positive", 0.0)

        if wsum <= 0.0:
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "compound": 0.0}

        neg = neg_sum / wsum
        neu = neu_sum / wsum
        pos = pos_sum / wsum

        return {"negative": float(neg), "neutral": float(neu), "positive": float(pos), "compound": 0.0}

    # -------------------------
    # Original API, now robust
    # -------------------------
    def raw_polarity_scores(self, text: str) -> Dict[str, float]:
        """
        Returns:
          {"negative": p_neg, "neutral": p_neu, "positive": p_pos, "compound": 0.0}

        If the text is longer than model limit (or throws the known RuntimeError),
        automatically falls back to sliding-window processing.
        """
        if text is None or not str(text).strip():
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "compound": 0.0}

        # Fast pre-check: if token length is too large, go sliding window directly
        tok = self._tokenizer
        max_len = int(self._max_length)
        special = int(tok.num_special_tokens_to_add(pair=False))
        window_tokens = max(1, max_len - special)

        try:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) > window_tokens:
                return self.raw_polarity_scores_sliding(text)

            results = self._pipe(text)  # list[list[{"label","score"}]]
            items = _normalize_pipeline_scores(results)
            scores = {
                str(it.get("label", "")).lower(): float(it["score"])
                for it in items
                if isinstance(it, dict) and "label" in it and "score" in it
            }

            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral", 0.0)
            pos = scores.get("positive", 0.0)

            return {"negative": neg, "neutral": neu, "positive": pos, "compound": 0.0}

        except RuntimeError as e:
            # Handles the typical "expanded size ... must match ... 514" error for long sequences
            return self.raw_polarity_scores_sliding(text)

    def polarity_scores(self, text: str) -> Dict[str, float]:
        return self.raw_polarity_scores(text)

    def default_polarity_scores(self, text: str) -> Dict[str, float]:
        return self.raw_polarity_scores(text)

    def vader_polarity_scores(self, text: str) -> Dict[str, float]:
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

    def major_label(self, text: str) -> Tuple[str, float]:
        s = self.raw_polarity_scores(text)
        label = max(("negative", "neutral", "positive"), key=lambda k: s[k])
        return label, self.int_label_map[label]

import math
from typing import Dict, Tuple, List

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    pipeline,
)


class UkrSentimentAnalyzer:
    """
    Ukrainian sentiment analysis model: YShynkarov/ukr-roberta-cosmus-sentiment

    Provides:
      - old_polarity_scores(text) -> {"negative","neutral","positive","compound"(=mixed)}
      - default_polarity_scores(text) -> same as old_polarity_scores (now long-text safe)
      - polarity_scores(text) -> {"neg","neu","pos","compound"}  (VADER-like)
      - major_label(text) -> ("negative"|"neutral"|"positive"|"mixed", -1|0|1)
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
        # long-text defaults
        max_length: int = 512,
        stride: int = 64,
        min_chunk_tokens: int = 16,
    ):
        self.compound_scale = float(compound_scale)
        self.compound_alpha = float(compound_alpha)
        self.split_mixed = bool(split_mixed)

        repo_id = "YShynkarov/ukr-roberta-cosmus-sentiment"
        safetensor = hf_hub_download(
            repo_id=repo_id,
            filename="ukrroberta_cosmus_sentiment.safetensors",
        )

        config = RobertaConfig.from_pretrained("youscan/ukr-roberta-base", num_labels=4)
        tokenizer = RobertaTokenizer.from_pretrained("youscan/ukr-roberta-base")

        model = RobertaForSequenceClassification(config)
        state_dict = load_file(safetensor)
        model.load_state_dict(state_dict)
        model.eval()

        self._tokenizer = tokenizer
        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,
            truncation=False,  # keep False; we handle long texts ourselves
        )

        self._max_length = int(max_length)
        self._stride = int(stride)
        self._min_chunk_tokens = int(min_chunk_tokens)

    def _vader_normalize(self, x: float) -> float:
        return x / math.sqrt(x * x + self.compound_alpha)

    # -------------------------
    # Sliding-window helpers
    # -------------------------
    def _iter_token_windows(
        self,
        input_ids: List[int],
        window_tokens: int,
        stride: int,
        min_chunk_tokens: int,
    ) -> List[Tuple[int, int]]:
        n = len(input_ids)
        if n == 0:
            return [(0, 0)]

        window_tokens = max(1, int(window_tokens))
        stride = max(0, int(stride))
        step = max(1, window_tokens - stride)

        spans: List[Tuple[int, int]] = []
        start = 0
        while start < n:
            end = min(start + window_tokens, n)
            if (end - start) >= min_chunk_tokens or start == 0:
                spans.append((start, end))
            if end >= n:
                break
            start += step

        if not spans:
            spans = [(0, min(window_tokens, n))]
        return spans

    def _half_overlap_weights(self, spans: List[Tuple[int, int]]) -> List[float]:
        if not spans:
            return []

        weights: List[float] = []
        for i, (s, e) in enumerate(spans):
            length = float(max(0, e - s))
            overlap_prev = 0.0
            overlap_next = 0.0

            if i > 0:
                ps, pe = spans[i - 1]
                overlap_prev = float(max(0, min(pe, e) - max(ps, s)))
            if i + 1 < len(spans):
                ns, ne = spans[i + 1]
                overlap_next = float(max(0, min(ne, e) - max(ns, s)))

            w = length - 0.5 * overlap_prev - 0.5 * overlap_next
            weights.append(max(0.0, w))

        if sum(weights) <= 0.0:
            return [1.0] * len(spans)
        return weights

    def polarity_scores_sliding(
        self,
        text: str,
        *,
        max_length: int | None = None,
        stride: int | None = None,
        min_chunk_tokens: int | None = None,
    ) -> Dict[str, float]:
        """
        Sliding-window version of old_polarity_scores (returns mixed under "compound").
        Aggregates per-chunk class probabilities with overlap-aware weights.
        """
        if text is None or not str(text).strip():
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "compound": 0.0}

        tok = self._tokenizer

        max_len = int(max_length if max_length is not None else self._max_length)
        special = int(tok.num_special_tokens_to_add(pair=False))
        window_tokens = max(1, max_len - special)

        stride_val = int(stride if stride is not None else self._stride)
        min_tokens = int(min_chunk_tokens if min_chunk_tokens is not None else self._min_chunk_tokens)

        input_ids = tok.encode(text, add_special_tokens=False)

        spans = self._iter_token_windows(
            input_ids=input_ids,
            window_tokens=window_tokens,
            stride=stride_val,
            min_chunk_tokens=min_tokens,
        )

        chunk_texts: List[str] = []
        for s, e in spans:
            chunk_ids = input_ids[s:e]
            chunk_texts.append(
                tok.decode(
                    chunk_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )

        results = self._pipe(chunk_texts)  # list[list[{"label","score"}]]
        weights = self._half_overlap_weights(spans)
        wsum = float(sum(weights))

        neg_sum = 0.0
        neu_sum = 0.0
        pos_sum = 0.0
        mix_sum = 0.0

        for w, res in zip(weights, results):
            items = _normalize_pipeline_scores(res)
            mapped = {}
            for it in items:
                if not isinstance(it, dict) or "label" not in it or "score" not in it:
                    continue
                label = str(it.get("label", ""))
                mapped_label = self.map_labels.get(label, label.lower())
                mapped[mapped_label] = float(it["score"])
            neg_sum += w * mapped.get("negative", 0.0)
            neu_sum += w * mapped.get("neutral", 0.0)
            pos_sum += w * mapped.get("positive", 0.0)
            mix_sum += w * mapped.get("mixed", 0.0)

        if wsum <= 0.0:
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "compound": 0.0}

        return {
            "negative": float(neg_sum / wsum),
            "neutral": float(neu_sum / wsum),
            "positive": float(pos_sum / wsum),
            "compound": float(mix_sum / wsum),  # keep same behavior: mixed stored in "compound"
        }

    # -------------------------
    # Original API, now robust
    # -------------------------
    def default_polarity_scores(self, text: str) -> Dict[str, float]:
        """
        Original single-pass behavior, but now auto-falls back to sliding window
        when text is too long or the model throws the long-seq RuntimeError.
        """
        if text is None or not str(text).strip():
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0, "compound": 0.0}

        tok = self._tokenizer
        max_len = int(self._max_length)
        special = int(tok.num_special_tokens_to_add(pair=False))
        window_tokens = max(1, max_len - special)

        try:
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) > window_tokens:
                return self.polarity_scores_sliding(text)

            results = self._pipe(text)  # list[list[{"label","score"}]]
            items = _normalize_pipeline_scores(results)
            scores = {}
            for it in items:
                if not isinstance(it, dict) or "label" not in it or "score" not in it:
                    continue
                label = str(it.get("label", ""))
                mapped_label = self.map_labels.get(label, label.lower())
                scores[mapped_label] = float(it["score"])
            return {
                "negative": scores.get("negative", 0.0),
                "neutral": scores.get("neutral", 0.0),
                "positive": scores.get("positive", 0.0),
                "compound": scores.get("mixed", 0.0),
            }
        except RuntimeError:
            return self.polarity_scores_sliding(text)

    def polarity_scores(self, text: str) -> Dict[str, float]:
        return self.default_polarity_scores(text)

    def vader_polarity_scores(self, text: str) -> Dict[str, float]:
        """
        VADER-like output using (possibly sliding-window aggregated) base probs.
        """
        s = self.default_polarity_scores(text)
        pos = s["positive"]
        neg = s["negative"]
        neu = s["neutral"]
        mix = s["compound"]  # mixed

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
        """
        Pick winning label among the *base* classes (negative/neutral/positive/mixed).
        Uses the underlying probability distribution (sliding-window safe).
        """
        base = self.default_polarity_scores(text)
        base4 = {
            "negative": base["negative"],
            "neutral": base["neutral"],
            "positive": base["positive"],
            "mixed": base["compound"],
        }
        label = max(base4, key=base4.get)
        return label, self.int_label_map[label]


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

    This function calculates two sentiment groups:
     1) sentiment_* via EngSentimentAnalyzer/UkrSentimentAnalyzer
     2) sentiment_vader_* via classic VADER (en) / Ukrainian VADER (ua|uk)
    and adds them to turn and summary dataframes.

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
        # Cache heavy analyzers across files in the same Python process.
        if not hasattr(get_sentiment, "_model_cache"):
            get_sentiment._model_cache = {}

        word_df, turn_df, summ_df = df_list
        _, turn_list, full_text = text_list
        lemmatizer = spacy.load("uk_core_news_sm") if lang in ['ua', 'uk'] else spacy.load('en_core_web_sm')

        lang_key = "uk" if lang in ["ua", "uk"] else "en"
        cache = get_sentiment._model_cache

        if lang_key == "en":
            if "vader_en" not in cache:
                cache["vader_en"] = SentimentIntensityAnalyzer()
            if "sentiment_en" not in cache:
                try:
                    cache["sentiment_en"] = EngSentimentAnalyzer()
                except Exception as model_exc:
                    logger.info(f"Falling back to English VADER for sentiment model: {model_exc}")
                    cache["sentiment_en"] = cache["vader_en"]
            sentiment_analyzer = cache["sentiment_en"]
            vader_analyzer = cache["vader_en"]
        else:
            if "vader_uk" not in cache:
                cache["vader_uk"] = UkrainianSentimentIntensityAnalyzerImproved()
            if "sentiment_uk" not in cache:
                try:
                    cache["sentiment_uk"] = UkrSentimentAnalyzer()
                except Exception as model_exc:
                    logger.info(f"Falling back to Ukrainian VADER for sentiment model: {model_exc}")
                    cache["sentiment_uk"] = cache["vader_uk"]
            sentiment_analyzer = cache["sentiment_uk"]
            vader_analyzer = cache["vader_uk"]

        sentiment_cols = [measures["neg"], measures["neu"], measures["pos"], measures["compound"]]
        vader_cols = [measures["neg_vader"], measures["neu_vader"], measures["pos_vader"], measures["compound_vader"]]
        mattr_cols = [measures["speech_mattr_5"], measures["speech_mattr_10"], measures["speech_mattr_25"], measures["speech_mattr_50"], measures["speech_mattr_100"]]
        all_cols = sentiment_cols + vader_cols + mattr_cols

        def _extract_sentiment_scores(score_dict):
            return [
                float(score_dict.get("negative", score_dict.get("neg", 0.0))),
                float(score_dict.get("neutral", score_dict.get("neu", 0.0))),
                float(score_dict.get("positive", score_dict.get("pos", 0.0))),
                float(score_dict.get("compound", 0.0)),
            ]

        def _extract_vader_scores(score_dict):
            return [
                float(score_dict.get("neg", score_dict.get("negative", 0.0))),
                float(score_dict.get("neu", score_dict.get("neutral", 0.0))),
                float(score_dict.get("pos", score_dict.get("positive", 0.0))),
                float(score_dict.get("compound", 0.0)),
            ]

        for idx, u in enumerate(turn_list):
            try:
                sentiment_dict = sentiment_analyzer.polarity_scores(u)
                vader_dict = vader_analyzer.polarity_scores(u)
                mattrs = [get_mattr(u, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]
                turn_df.loc[idx, all_cols] = _extract_sentiment_scores(sentiment_dict) + _extract_vader_scores(vader_dict) + mattrs

            except Exception as e:
                logger.info(f"Error in sentiment analysis: {e}")
                continue

        sentiment_dict = sentiment_analyzer.polarity_scores(full_text)
        vader_dict = vader_analyzer.polarity_scores(full_text)
        mattrs = [get_mattr(full_text, lemmatizer, window_size=ws) for ws in [5, 10, 25, 50, 100]]

        summ_df.loc[0, all_cols] = _extract_sentiment_scores(sentiment_dict) + _extract_vader_scores(vader_dict) + mattrs
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
