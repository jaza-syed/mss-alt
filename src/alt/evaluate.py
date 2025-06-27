# stdlib
import logging
import regex as re
from typing import Dict, List, Tuple, Any
from multiprocessing import Pool

# third-party
import numpy as np
import pandas as pd
import rapidfuzz
from tqdm import tqdm

# first-party
from . import metrics
from .alt_types import Result, EvalResult, Song, InferResult
from .tokenizer import LyricsTokenizer, tokens_as_words, Token, WORD, BACKING, NONLEXICAL

logger = logging.getLogger(__name__)


## evaluate.py should be eval song + dataframe conversion tools
## evaluate.py is a set of **evaluation utilities**
## eval_song takes infer_result and produces eval data
## evaluate.py contains utilities to convert eval data to dataframes for analysis
## render.py contains utilities to render eval data to html with jinja2
tokenizer = LyricsTokenizer()


# NOTE: Not actually used in alt_eval, just provided for guideline adherence
def normalize_lyrics(text: str) -> str:
    """Normalize lyrics to improve adherence to the Jam-ALT annotation guide.

    Unwanted end-of-line punctuation is removed and the first letter of each line is uppercased.

    This is a relatively crude normalization method that should generally improve the results at
    least for languages that use the Latin script. However, it may make the results worse if the
    line break predictions are not accurate.
    """
    # Remove unwanted trailing punctuation if not on a line on its own
    # TODO: Hack for broken tags
    text = re.sub(">,", ">", text)
    # TODO: Add > to allow tags at end to exist
    text = re.sub(r"(?<!^)[^\w\n!?'´‘’\"“”»>)]+ *$", "", text, flags=re.MULTILINE)
    # Uppercase the first letter of each line
    text = re.sub(
        r"^([^\w\n]*\w)", lambda m: m.group(1).upper(), text, flags=re.MULTILINE
    )
    return text


def eval_song_new(data: Result) -> EvalResult:
    hyps = [normalize_lyrics(sample.hyp) for sample in data.infer.asr_samples]
    refs = [normalize_lyrics(sample.ref) for sample in data.infer.asr_samples]

    song_metrics, wo = metrics.compute_metrics(refs, hyps, data.song.language)

    ref_tokens = [tokenizer(ref, data.song.language) for ref in refs]
    hyp_tokens = [tokenizer(hyp, data.song.language) for hyp in hyps]
    return EvalResult(
        metrics=song_metrics, wo=wo, ref_tokens=ref_tokens, hyp_tokens=hyp_tokens
    )


### DataFrame conversion stuff

# Convert from AlignmentChunk type to our metrics.py names
map_type = {
    "insert": "ins",
    "equal": "hit",
    "delete": "del",
    "substitute": "sub",
}


def get_chunk_edits(
    reference: List[Token], hypothesis: List[Token], chunk_type: str
) -> List[Tuple[str, str, str, set]]:
    """
    Extract edits from a chunk alignment.
    """
    edits = []
    if chunk_type == "delete":
        for token in reference:
            edits.append(("del", token.text, None, token.tags & {NONLEXICAL, BACKING}))
    elif chunk_type == "insert":
        for token in hypothesis:
            edits.append(("ins", None, token.text, set()))
    elif chunk_type in ["substitute", "equal"]:
        for token_ref, token_hyp in zip(reference, hypothesis):
            edits.append(
                (
                    "hit" if chunk_type == "equal" else "sub",
                    token_ref.text,
                    token_hyp.text,
                    token_ref.tags & {NONLEXICAL, BACKING},
                )
            )
    return edits


def word_dist(w1: str, w2: str) -> float:
    """
    Compute Levenshtein distance between two words.
    """
    if w1 is None or w2 is None:
        return np.nan
    w1 = w1.replace("'", "")
    w2 = w2.replace("'", "")
    return rapidfuzz.distance.Levenshtein.distance(w1, w2)


def song_to_rows(
    data: Result,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Evaluate a song and return dataframe rows
    This can handle inference at song or line level!
    Returns: samples ([song] or [line, line ...]), [chunks], [edits]
    """
    if not data.eval:
        eval_result = eval_song_new(data)
    else:
        eval_result = data.eval

    if not eval_result:
        return ([], [], [])

    uid = data.song.uid
    dataset = data.song.dataset_id
    language = data.song.language

    sample_rows = []
    edit_rows = []
    chunk_rows = []

    for sample_idx, (sample, hyp_tokens, ref_tokens, alignment) in enumerate(
        zip(
            data.infer.asr_samples,
            eval_result.hyp_tokens,
            eval_result.ref_tokens,
            eval_result.wo.alignments,
        )
    ):
        ref_tokens = tokens_as_words(ref_tokens)
        hyp_tokens = tokens_as_words(hyp_tokens)
        alignment = eval_result.wo.alignments[sample_idx]
        edit_counts, _error_counts = metrics.process_alignments(
            [ref_tokens], [hyp_tokens], [alignment]
        )
        e = edit_counts[WORD]
        en = edit_counts[NONLEXICAL]
        eb = edit_counts[BACKING]
        duration = sample.end - sample.start
        sample_row = {
            "wer": (e.S + e.D + e.I) / (e.H + e.S + e.D),
            "hit": e.H,
            "sub": e.S,
            "del": e.D,
            "ins": e.I,
            # en.x + eb.x <= 2*e.x for x in {H, S, D}
            "hit_nl": en.H,
            "sub_nl": en.S,
            "del_nl": en.D,
            "hit_bv": eb.H,
            "sub_bv": eb.S,
            "del_bv": eb.D,
            "len": (e.H + e.S + e.D),
            "sample_idx": sample_idx,
            "uid": uid,
            "dataset": dataset,
            "start": sample.start,
            "end": sample.end,
            "language": language,
            "ref": " ".join(token.text for token in ref_tokens),
            "hyp": " ".join(token.text for token in hyp_tokens),
            "duration": duration,
        }
        sample_rows.append(sample_row)
        for chunk_idx, chunk in enumerate(alignment):
            chunk_hyp = hyp_tokens[chunk.hyp_start_idx : chunk.hyp_end_idx]
            chunk_ref = ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx]
            chunk_row = {
                "typ": map_type[chunk.type],
                "ref": " ".join(token.text for token in chunk_ref),
                "hyp": " ".join(token.text for token in chunk_hyp),
                "len": max(len(chunk_hyp), len(chunk_ref)),
                "uid": uid,
                "dataset": dataset,
                "language": language,
                "sample_idx": sample_idx,
                "chunk_idx": chunk_idx,
                "nl": 0,
                "bv": 0,
            }
            for op, edit_ref, edit_hyp, tags in get_chunk_edits(
                chunk_ref, chunk_hyp, chunk.type
            ):
                if NONLEXICAL in tags:
                    chunk_row["nl"] += 1
                if BACKING in tags:
                    chunk_row["bv"] += 1
                edit_rows.append(
                    {
                        "typ": op,
                        "ref": edit_ref,
                        "hyp": edit_hyp,
                        "uid": uid,
                        "dataset": dataset,
                        "language": language,
                        "dist": word_dist(edit_ref, edit_hyp),
                        "nl": NONLEXICAL in tags,
                        "bv": BACKING in tags,
                    }
                )
            chunk_rows.append(chunk_row)
    return sample_rows, chunk_rows, edit_rows


# Utility wrapping song_to_rows to convert results of eval_song into DataFrames
def process_data(
    data: dict[tuple[str, ...], list[Result]],
    system_keys: tuple[Any, ...],
    num_workers: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process raw data into DataFrames for songs, chunks, and edits.
    data: Map from system->evaluation results
    """
    sample_rows, chunk_rows, edit_rows = [], [], []
    pool = Pool(num_workers)
    for system, sample_data in tqdm(list(data.items())):
        assert len(system) == len(system_keys)
        system_dict = dict(zip(system_keys, system))
        results = pool.map(song_to_rows, tqdm(sample_data))
        for samples, chunks, edits in results:
            sample_rows.extend({**sample, **system_dict} for sample in samples)
            chunk_rows.extend({**chunk, **system_dict} for chunk in chunks)
            edit_rows.extend({**edit, **system_dict} for edit in edits)
    pool.close()
    pool.join()
    return pd.DataFrame(sample_rows), pd.DataFrame(chunk_rows), pd.DataFrame(edit_rows)
