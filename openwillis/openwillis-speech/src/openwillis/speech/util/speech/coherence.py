# author:    Georgios Efstathiadis
# website:   http://www.bklynhlth.com

# import the required packages
import logging
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
login(token)



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
# Suppress warnings from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)


DEFAULT_EMBEDDING_MODEL_ID = "google/embeddinggemma-300m"
DEFAULT_PPL_MODEL_ID = "google/gemma-3-270m"
PPL_MAX_TOKENS = 2048
EMBEDDING_BATCH_SIZE = 256
WINDOW_BATCH_SIZE = 512
MIN_EMBEDDING_BATCH_SIZE = 8


def _select_torch_device(explicit: Optional[str] = None) -> torch.device:
    """Return an available torch device, preferring CUDA, then MPS, otherwise CPU."""
    if explicit:
        return torch.device(explicit)

    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _maybe_enable_tf32(device: torch.device) -> None:
    """Enable TF32 where it is supported to speed up matmul-heavy workloads."""
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


_MODEL_CACHE: Dict[str, Dict[str, object]] = {}


def _load_sentence_encoder(device: torch.device) -> SentenceTransformer:
    """Instantiate the sentence embedding model on the requested device."""
    target = device.type
    # SentenceTransformer expects a string device identifier
    try:
        return SentenceTransformer(DEFAULT_EMBEDDING_MODEL_ID, device=target)
    except Exception:
        if target != "cpu":
            # Fallback to CPU if the requested accelerator is unsupported
            return SentenceTransformer(DEFAULT_EMBEDDING_MODEL_ID, device="cpu")
        raise


def _resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        # Gemma kernels on MPS are unstable in float16/bfloat16; force float32
        return torch.float32
    return torch.float32


def _load_lm_bundle(device: torch.device) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Instantiate the Gemma language model and tokenizer on the requested device."""
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_PPL_MODEL_ID)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = _resolve_dtype(device)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_PPL_MODEL_ID,
            torch_dtype=dtype,
        )
    except (TypeError, ValueError):
        # Fallback for older transformer versions or unsupported dtypes
        model = AutoModelForCausalLM.from_pretrained(DEFAULT_PPL_MODEL_ID)

    if device.type == "mps":
        model = model.to(device, dtype=torch.float32)
    else:
        model = model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return tokenizer, model


def get_model_bundle(language: str, device_hint: Optional[str] = None) -> Dict[str, object]:
    """Lazy-load and cache the Gemma models required for coherence metrics."""
    cache_key = (language or "").lower() or "default"
    bundle = _MODEL_CACHE.get(cache_key)
    if bundle:
        return bundle

    device = _select_torch_device(device_hint)
    _maybe_enable_tf32(device)

    sentence_encoder = None
    try:
        sentence_encoder = _load_sentence_encoder(device)
    except Exception as exc:
        logger.warning("Failed to load sentence encoder on %s: %s", device, exc)

    try:
        tokenizer, lm_model = _load_lm_bundle(device)
    except Exception as exc:
        logger.warning("Failed to load Gemma LM on %s: %s", device, exc)
        tokenizer, lm_model = None, None

    bundle = {
        "device": device,
        "sentence_encoder": sentence_encoder,
        "tokenizer": tokenizer,
        "lm_model": lm_model,
        "model_max_length": int(getattr(lm_model.config, "max_position_embeddings", 32768)) if lm_model else None,
    }

    _MODEL_CACHE[cache_key] = bundle
    return bundle

def get_word_embeddings(word_list, sentence_encoder):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the word embeddings for the input text using the Gemma sentence encoder.

    Parameters:
    ...........
    word_list: list
        List of transcribed text at the word level.
    sentence_encoder: SentenceTransformer
        Sentence embedding model used for encoding.

    Returns:
    ...........
    word_embeddings: numpy array
        The calculated word embeddings.

    ------------------------------------------------------------------------------------------------------
    """
    if sentence_encoder is None or len(word_list) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    def _safe_encode(bs: int) -> np.ndarray:
        return _encode_in_chunks(sentence_encoder, word_list, max(bs, 1))

    try:
        return _safe_encode(EMBEDDING_BATCH_SIZE)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise RuntimeError(f"Error in embedding calculation: {exc}") from exc

        logger.warning(
            "Sentence encoder OOM on device %s; retrying on CPU with smaller batches.",
            getattr(sentence_encoder, "device", "unknown"),
        )

        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()  # type: ignore[attr-defined]
        except Exception:
            pass

        try:
            sentence_encoder.to("cpu")  # type: ignore[attr-defined]
        except Exception:
            pass

        reduced_bs = max(MIN_EMBEDDING_BATCH_SIZE, EMBEDDING_BATCH_SIZE // 4)
        return _safe_encode(reduced_bs)
    except Exception as exc:
        raise RuntimeError(f"Error in embedding calculation: {exc}") from exc

def get_word_coherence_utterance(row, sentence_encoder, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures for a single utterance.

    Parameters:
    ...........
    row: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    sentence_encoder: SentenceTransformer
        A sentence encoder model (Gemma embeddings).
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    word_coherence: list
        A list containing the calculated semantic similarity of each word to the immediately preceding word.
    word_coherence_5: list
        A list containing the calculated semantic similarity of each word in 5-words window.
    word_coherence_10: list
        A list containing the calculated semantic similarity of each word in 10-words window.
    word_word_variability: dict
        A dictionary containing the calculated word-to-word variability at k inter-word distances.

    ------------------------------------------------------------------------------------------------------
    """
    words_texts = row[measures['words_texts']]
    if len(words_texts) == 0:
        # return empty lists if no words in the utterance
        return [np.nan]*len(words_texts), [np.nan]*len(words_texts), [np.nan]*len(words_texts), {k: [np.nan]*len(words_texts) for k in range(2, 11)}

    word_embeddings = get_word_embeddings(words_texts, sentence_encoder)
    similarity_matrix = cosine_similarity(word_embeddings)

    # calculate semantic similarity of each word to the immediately preceding word
    if len(words_texts) > 1:
        word_coherence = [np.nan] + [similarity_matrix[j, j-1] for j in range(1, len(words_texts))]
    else:
        word_coherence = [np.nan]*len(words_texts)

    # calculate semantic similarity of each word in 5-words window
    if len(words_texts) > 5:
        word_coherence_5 = [np.nan]*2 + [np.mean(similarity_matrix[j-2:j+3, j]) for j in range(2, len(words_texts)-2)] + [np.nan]*2
    else:
        word_coherence_5 = [np.nan]*len(words_texts)

    # calculate semantic similarity of each word in 10-words window
    if len(words_texts) > 10:
        word_coherence_10 = [np.nan]*5 + [np.mean(similarity_matrix[j-5:j+6, j]) for j in range(5, len(words_texts)-5)] + [np.nan]*5
    else:
        word_coherence_10 = [np.nan]*len(words_texts)

    # calculate word-to-word variability at k inter-word distances (for k from 2 to 10)
    # indicating semantic similarity between each word and the next following word at k inter-word distance
    word_word_variability = {}
    for k in range(2, 11):
        if len(words_texts) > k:
            word_word_variability[k] = [similarity_matrix[j, j+k] for j in range(len(words_texts)-k)] + [np.nan]*k
        else:
            word_word_variability[k] = [np.nan]*len(words_texts)

    return word_coherence, word_coherence_5, word_coherence_10, word_word_variability    

def get_word_coherence_summary(word_df, summ_df, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function summarizes the word coherence measures at the summary level.

    Parameters:
    ...........
    word_df: pandas dataframe
        A dataframe containing word summary information.
    summ_df: pandas dataframe
        A dataframe containing summary information.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    summ_df: pandas dataframe
        The updated summ_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """

    summ_df[measures['word_coherence_mean']] = word_df[measures['word_coherence']].mean(skipna=True)
    summ_df[measures['word_coherence_var']] = word_df[measures['word_coherence']].var(skipna=True)
    summ_df[measures['word_coherence_5_mean']] = word_df[measures['word_coherence_5']].mean(skipna=True)
    summ_df[measures['word_coherence_5_var']] = word_df[measures['word_coherence_5']].var(skipna=True)
    summ_df[measures['word_coherence_10_mean']] = word_df[measures['word_coherence_10']].mean(skipna=True)
    summ_df[measures['word_coherence_10_var']] = word_df[measures['word_coherence_10']].var(skipna=True)
    for k in range(2, 11):
        summ_df[measures[f'word_coherence_variability_{k}_mean']] = word_df[measures[f'word_coherence_variability_{k}']].mean(skipna=True)
        summ_df[measures[f'word_coherence_variability_{k}_var']] = word_df[measures[f'word_coherence_variability_{k}']].var(skipna=True)

    return summ_df

def append_nan_values(coherence_lists, row_len):
    """
    ------------------------------------------------------------------------------------------------------

    Helper function for appending NaN values to the coherence lists.

    Parameters:
    ...........
    coherence_lists: dict
        A dictionary containing the coherence lists.
    row_len: int
        The length of the row.

    Returns:
    ...........
    coherence_lists: dict
        The updated coherence lists.

    ------------------------------------------------------------------------------------------------------
    """
    coherence_lists['word_coherence'] += [np.nan] * row_len
    coherence_lists['word_coherence_5'] += [np.nan] * row_len
    coherence_lists['word_coherence_10'] += [np.nan] * row_len
    for k in range(2, 11):
        coherence_lists['variability'][k] += [np.nan] * row_len

    return coherence_lists

def get_word_coherence(df_list, utterances_speaker, min_coherence_turn_length, language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates word coherence measures

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_speaker: pandas dataframe
        A dataframe containing the turns extracted from the JSON object for the specified speaker.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    language: str
        Language of the transcribed text.
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

        bundle = get_model_bundle(language)
        sentence_encoder: Optional[SentenceTransformer] = bundle.get("sentence_encoder")

        if sentence_encoder is None:
            logger.info(f"Sentence encoder not available for language {language}; skipping word coherence analysis.")
            return df_list

        # Initialize coherence lists
        coherence_lists = {
            'word_coherence': [],
            'word_coherence_5': [],
            'word_coherence_10': [],
            'variability': {k: [] for k in range(2, 11)}
        }

        # Process each utterance
        for _, row in utterances_speaker.iterrows():
            try:
                if len(row[measures['words_texts']]) < min_coherence_turn_length:
                    coherence_lists = append_nan_values(coherence_lists, len(row[measures['words_texts']]))
                    continue

                # Get word coherence for the utterance
                coherence, coherence_5, coherence_10, variability = get_word_coherence_utterance(row, sentence_encoder, measures)

                # Append results to lists
                coherence_lists['word_coherence'] += coherence
                coherence_lists['word_coherence_5'] += coherence_5
                coherence_lists['word_coherence_10'] += coherence_10
                for k in range(2, 11):
                    coherence_lists['variability'][k] += variability[k]

            except Exception as e:
                logger.info(f"Error in word coherence analysis for row: {e}")
                coherence_lists = append_nan_values(coherence_lists, len(row[measures['words_texts']]))

        # Update word_df with calculated coherence values
        word_df[measures['word_coherence']] = coherence_lists['word_coherence']
        word_df[measures['word_coherence_5']] = coherence_lists['word_coherence_5']
        word_df[measures['word_coherence_10']] = coherence_lists['word_coherence_10']
        for k in range(2, 11):
            word_df[measures[f'word_coherence_variability_{k}']] = coherence_lists['variability'][k]

        # Update the summary-level dataframe
        summ_df = get_word_coherence_summary(word_df, summ_df, measures)

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in word coherence analysis: {e}")
    finally:
        return df_list

def calculate_perplexity(
    text: str,
    model: Optional[AutoModelForCausalLM],
    tokenizer: Optional[AutoTokenizer],
    model_max_length: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the perplexity of the input text using the Gemma causal language model.

    Parameters:
    ...........
    text: str
        The input text to be analyzed.
    model: AutoModelForCausalLM
        A Gemma causal language model.
    tokenizer: AutoTokenizer
        The tokenizer paired with the causal language model.

    Returns:
    ...........
    Tuple of floats
        The calculated perplexity of the input text (global teacher-forced perplexity).
        The calculated windowed perplexity with window size 2.
        The calculated windowed perplexity with window size 5.
        The calculated windowed perplexity with window size 7.

    ------------------------------------------------------------------------------------------------------
    """
    if model is None or tokenizer is None:
        return (np.nan, np.nan, np.nan, np.nan)

    if not isinstance(text, str):
        return (np.nan, np.nan, np.nan, np.nan)

    clean_text = re.sub(r"\s+", " ", text.strip())
    if len(clean_text) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    tokens = tokenizer(clean_text, return_tensors="pt", add_special_tokens=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    if input_ids.size(1) < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    max_len_model = model_max_length or int(getattr(model.config, "max_position_embeddings", input_ids.size(1)))
    effective_len = min(max_len_model, PPL_MAX_TOKENS)
    if input_ids.size(1) > effective_len:
        input_ids = input_ids[:, :effective_len]
        attention_mask = attention_mask[:, :effective_len]

    def _teacher_forced_perplexity(ids: torch.Tensor, mask: torch.Tensor) -> float:
        with torch.inference_mode():
            outputs = model(input_ids=ids, attention_mask=mask)
            logits = outputs.logits[:, :-1, :].float()
        labels = ids[:, 1:]
        attn = mask[:, 1:].to(logits.dtype)
        attn_sum = attn.sum()
        if attn_sum <= 0:
            return float("nan")
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        nll = (nll * attn).sum() / attn_sum
        return float(torch.exp(nll).item())

    def _windowed_perplexity(ids: torch.Tensor, k: int) -> float:
        ids = ids[:, :min(ids.size(1), PPL_MAX_TOKENS)]
        seq_len = ids.size(1)
        if seq_len < 2:
            return float("nan")

        k_eff = min(k, max_len_model - 1)
        if k_eff <= 0:
            return float("nan")

        pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token is None:
            fallback_token = tokenizer.eos_token or tokenizer.pad_token or tokenizer.bos_token
            if fallback_token is not None:
                pad_token = tokenizer.convert_tokens_to_ids(fallback_token)
            else:
                pad_token = 0

        log_prob_chunks = []
        for start in range(1, seq_len, WINDOW_BATCH_SIZE):
            positions = list(range(start, min(seq_len, start + WINDOW_BATCH_SIZE)))
            if not positions:
                continue

            lens, next_tokens, sequences = [], [], []
            for pos in positions:
                window = min(k_eff, pos)
                lens.append(window)
                next_tokens.append(int(ids[0, pos].item()))
                sequences.append(ids[0, pos - window:pos])

            max_window = max(lens) if lens else 0
            if max_window == 0:
                continue

            batch = torch.full((len(sequences), max_window), pad_token, dtype=torch.long, device=device)
            for row_idx, seq in enumerate(sequences):
                if lens[row_idx] == 0:
                    continue
                batch[row_idx, -lens[row_idx]:] = seq

            attn_mask = (batch != pad_token).long()
            lens_tensor = torch.tensor(lens, device=device, dtype=torch.long)
            next_tensor = torch.tensor(next_tokens, device=device, dtype=torch.long)

            with torch.inference_mode():
                outputs = model(input_ids=batch, attention_mask=attn_mask)
                idx = lens_tensor - 1
                logits = outputs.logits[torch.arange(batch.size(0), device=device), idx].float()
                log_probs = torch.log_softmax(logits, dim=-1)
                log_prob_chunks.append(log_probs[torch.arange(batch.size(0), device=device), next_tensor])

        if not log_prob_chunks:
            return float("nan")

        stacked = torch.cat(log_prob_chunks).float()
        finite_mask = torch.isfinite(stacked)
        if not torch.any(finite_mask):
            return float("nan")

        stacked = stacked[finite_mask]
        return float(torch.exp(-stacked.mean()).item())

    global_ppl = _teacher_forced_perplexity(input_ids, attention_mask)
    ppl_2 = _windowed_perplexity(input_ids, 2)
    ppl_5 = _windowed_perplexity(input_ids, 5)
    ppl_7 = _windowed_perplexity(input_ids, 7)

    return global_ppl, ppl_2, ppl_5, ppl_7

def calculate_phrase_tangeniality(phrases_texts, utterance_text, sentence_encoder, lm_model, tokenizer):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates the semantic similarity of each phrase to the immediately preceding phrase,
    the semantic similarity of each phrase to the phrase 2 turns before, and the Gemma-based perplexity of the turn.

    Parameters:
    ...........
    phrases_texts: list
        List of transcribed text at the phrase level.
    utterance_text: str
        The full transcribed text.
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    lm_model: AutoModelForCausalLM
        A Gemma causal language model.
    tokenizer: AutoTokenizer
        A tokenizer paired with the causal language model.

    Returns:
    ...........
    sentence_tangeniality1: float
        The semantic similarity of each phrase to the immediately preceding phrase.
    sentence_tangeniality2: float
        The semantic similarity of each phrase to the phrase 2 turns before.
    perplexity: float
        The perplexity of the turn (global teacher-forced perplexity).
    perplexity_5: float
        The windowed perplexity of the turn with window size 2.
    perplexity_11: float
        The windowed perplexity of the turn with window size 5.
    perplexity_15: float
        The windowed perplexity of the turn with window size 7.

    ------------------------------------------------------------------------------------------------------
    """
    sentence_tangeniality1 = np.nan
    sentence_tangeniality2 = np.nan
    if sentence_encoder is not None and len(phrases_texts) > 0:
        phrase_embeddings = sentence_encoder.encode(phrases_texts)
        similarity_matrix = cosine_similarity(phrase_embeddings)

        # calculate semantic similarity of each phrase to the immediately preceding phrase
        if len(phrases_texts) > 1:
            sentence_tangeniality1 = np.mean([similarity_matrix[j-1, j] for j in range(1, len(phrases_texts))])

        # calculate semantic similarity of each phrase to the phrase 2 turns before
        if len(phrases_texts) > 2:
            sentence_tangeniality2 = np.mean([similarity_matrix[j-2, j] for j in range(2, len(phrases_texts))])

    perplexity, perplexity_5, perplexity_11, perplexity_15 = np.nan, np.nan, np.nan, np.nan
    if tokenizer is not None and lm_model is not None:
        max_len_model = int(getattr(lm_model.config, "max_position_embeddings", 32768))
        perplexity, perplexity_5, perplexity_11, perplexity_15 = calculate_perplexity(
            utterance_text,
            lm_model,
            tokenizer,
            model_max_length=max_len_model,
        )

    return sentence_tangeniality1, sentence_tangeniality2, perplexity, perplexity_5, perplexity_11, perplexity_15

def calculate_slope(y):
    """
    ------------------------------------------------------------------------------------------------------
    This function calculates the slope
     of the input list using linear regression

    Parameters:
    ...........
    y: list
        A list of values.

    Returns:
    ...........
    float
        The calculated slope of the input list.

    ------------------------------------------------------------------------------------------------------
    """
    # remove NaNs
    y = [val for val in y if not np.isnan(val)]

    x = range(len(y))
    try:
        slope, _ = np.polyfit(x, y, 1)
    except:
        slope = np.nan

    return slope

def init_model(language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function initializes the appropriate models and tokenizers based on language.

    Parameters:
    ...........
    language: str
        Language of the transcribed text.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    tokenizer: AutoTokenizer
        A tokenizer compatible with the Gemma causal language model.
    lm_model: AutoModelForCausalLM
        A Gemma causal language model.

    ------------------------------------------------------------------------------------------------------
    """
    sentence_encoder, tokenizer, lm_model = None, None, None

    bundle = get_model_bundle(language)
    sentence_encoder = bundle.get("sentence_encoder")
    tokenizer = bundle.get("tokenizer")
    lm_model = bundle.get("lm_model")

    if sentence_encoder is None and language not in measures.get("english_langs", set()):
        if language in measures.get("supported_langs_sentence_embeddings", set()):
            logger.info(f"Sentence encoder is unavailable for language {language}.")
        else:
            logger.info(f"Language {language} not supported for phrase coherence analysis")

    if (tokenizer is None or lm_model is None) and language not in measures.get("english_langs", set()):
        if language in measures.get("supported_langs_bert", set()):
            logger.info(f"Gemma language model unavailable for language {language}.")
        else:
            logger.info(f"Language {language} not supported for perplexity analysis")

    return sentence_encoder, tokenizer, lm_model

def calculate_turn_coherence(utterances_filtered, turn_df, min_coherence_turn_length, speaker_label, sentence_encoder, lm_model, tokenizer, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates turn coherence measures for the specified speaker.

    Parameters:
    ...........
    utterances_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
        after filtering out turns with less than min_turn_length words of the specified speaker.
    turn_df: pandas dataframe
        A dataframe containing turn summary information.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    speaker_label: str
        Speaker label
    sentence_encoder: SentenceTransformer
        A SentenceTransformer model.
    lm_model: AutoModelForCausalLM
        A Gemma causal language model.
    tokenizer: AutoTokenizer
        The tokenizer paired with the causal language model.
    measures: dict
        A dictionary containing the names of the columns in the output dataframes.

    Returns:
    ...........
    turn_df: pandas dataframe
        The updated turn_df dataframe.

    ------------------------------------------------------------------------------------------------------
    """
    # speaker_label = ""
    # semantic similarity between each pair of utterances
    utterances_texts = utterances_filtered[measures['utterance_text']].values.tolist()
    utterances_embeddings = sentence_encoder.encode(utterances_texts) if sentence_encoder else None
    similarity_matrix = cosine_similarity(utterances_embeddings) if sentence_encoder else None

    # Initialize coherence lists
    sentence_tangeniality1_list, sentence_tangeniality2_list = [], []
    perplexity_list, perplexity_5_list, perplexity_11_list, perplexity_15_list = [], [], [], []
    turn_to_turn_tangeniality_list = []
    for i, row in utterances_filtered.iterrows():
        current_speaker = row[measures['speaker_label']]
        current_speaker = speaker_label # 
        if current_speaker != speaker_label:
            continue
        elif len(row[measures['words_texts']]) < min_coherence_turn_length:
            sentence_tangeniality1_list.append(np.nan)
            sentence_tangeniality2_list.append(np.nan)
            perplexity_list.append(np.nan)
            perplexity_5_list.append(np.nan)
            perplexity_11_list.append(np.nan)
            perplexity_15_list.append(np.nan)
            turn_to_turn_tangeniality_list.append(np.nan)
            continue

        phrases_texts = row[measures['phrases_texts']]
        utterance_text = row[measures['utterance_text']]
        
        sentence_tangeniality1, sentence_tangeniality2, perplexity, perplexity_5, perplexity_11, perplexity_15 = calculate_phrase_tangeniality(
            phrases_texts, utterance_text, sentence_encoder, lm_model, tokenizer
        )
        
        sentence_tangeniality1_list.append(sentence_tangeniality1)
        sentence_tangeniality2_list.append(sentence_tangeniality2)
        perplexity_list.append(perplexity)
        perplexity_5_list.append(perplexity_5)
        perplexity_11_list.append(perplexity_11)
        perplexity_15_list.append(perplexity_15)

        if i == 0 or len(utterances_filtered.iloc[i - 1][measures['words_texts']]) < min_coherence_turn_length or not sentence_encoder:
            turn_to_turn_tangeniality_list.append(np.nan)
        else:
            turn_to_turn_tangeniality_list.append(similarity_matrix[i, i - 1])

    turn_df[measures['sentence_tangeniality1']] = sentence_tangeniality1_list
    turn_df[measures['sentence_tangeniality2']] = sentence_tangeniality2_list
    turn_df[measures['perplexity']] = perplexity_list
    turn_df[measures['perplexity_5']] = perplexity_5_list
    turn_df[measures['perplexity_11']] = perplexity_11_list
    turn_df[measures['perplexity_15']] = perplexity_15_list
    turn_df[measures['turn_to_turn_tangeniality']] = turn_to_turn_tangeniality_list

    return turn_df

def get_phrase_coherence(df_list, utterances_filtered, min_coherence_turn_length, speaker_label, language, measures):
    """
    ------------------------------------------------------------------------------------------------------

    This function calculates turn coherence measures

    Parameters:
    ...........
    df_list: list
        List of pandas dataframes.
    utterances_filtered: pandas dataframe
        A dataframe containing the turns extracted from the JSON object
        after filtering out turns with less than min_turn_length words of the specified speaker.
    min_coherence_turn_length: int
        Minimum number of words in a turn for word coherence analysis.
    speaker_label: str
        Speaker label
    language: str
        Language of the transcribed text.
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

        sentence_encoder, tokenizer, lm_model = init_model(language, measures)
        if not sentence_encoder and not tokenizer and not lm_model:
            logger.info(f"Language {language} not supported for phrase coherence nor perplexity analysis")
            return df_list

        # turn-level
        if len(turn_df) > 0:
            turn_df = calculate_turn_coherence(
                utterances_filtered,
                turn_df,
                min_coherence_turn_length,
                speaker_label,
                sentence_encoder,
                lm_model,
                tokenizer,
                measures,
            )

            for measure in ['sentence_tangeniality1', 'sentence_tangeniality2', 'perplexity', 'perplexity_5', 'perplexity_11', 'perplexity_15', 'turn_to_turn_tangeniality']:
                if turn_df[measures[measure]].isnull().all():
                    continue
                summ_df[measures[measure + '_mean']] = turn_df[measures[measure]].mean(skipna=True)
                summ_df[measures[measure + '_var']] = turn_df[measures[measure]].var(skipna=True)

            if not turn_df[measures['turn_to_turn_tangeniality']].isnull().all():
                summ_df[measures['turn_to_turn_tangeniality_slope']] = calculate_slope(turn_df[measures['turn_to_turn_tangeniality']])

        df_list = [word_df, turn_df, summ_df]
    except Exception as e:
        logger.info(f"Error in phrase coherence analysis: {e}")
    finally:
        return df_list
def _encode_in_chunks(
    encoder: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    """Encode texts in smaller batches to control peak memory usage."""
    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    outputs: List[np.ndarray] = []
    from tqdm import tqdm

    for start in tqdm(range(0, len(texts), batch_size), desc="Calculating embeddings", unit="batches"):
        chunk = texts[start:start + batch_size]
        if len(chunk) == 0:
            continue
        chunk_emb = encoder.encode(
            chunk,
            batch_size=min(batch_size, len(chunk)),
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        outputs.append(np.asarray(chunk_emb, dtype=np.float32))

    if not outputs:
        return np.zeros((0, encoder.get_sentence_embedding_dimension()), dtype=np.float32)

    return np.vstack(outputs)
