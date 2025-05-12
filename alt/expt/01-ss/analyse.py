# stdlib
from multiprocessing import Pool
from typing import Tuple, List, Dict

# third-party
import pandas as pd
import numpy as np
import rapidfuzz
import jiwer
from tqdm import tqdm

# first-party
from alt import util, evaluate
from alt.alt_types import SongInfo
from alt import metrics
from alt.tokenizer import Token, WORD, BACKING, NONLEXICAL

# Tokenizer instance
tokenizer = metrics.LyricsTokenizer()

map_type = {
    "insert": "ins",
    "equal": "hit",
    "delete": "del",
    "substitute": "sub",
}


def eval_song(name: str, uid: str) -> Tuple[SongInfo, jiwer.WordOutput]:
    """
    Evaluate a single song and return SongInfo and WordOutput.
    """
    songinfo = util.read_pz(f"../../build/{name}/infer/{uid}.pz")
    eval_data, wo = evaluate.eval_song(songinfo)
    songinfo.evaluate = eval_data
    return songinfo, wo


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
    songinfo: SongInfo, wo: jiwer.WordOutput
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Convert a song's evaluation data into rows for DataFrames.
    This can handle inference at song or line level!
    Returns: samples ([song] or [line, line ...]), [chunks], [edits]
    """
    num_samples = len(wo.alignments)
    eval_data = songinfo.evaluate
    assert eval_data is not None
    assert len(eval_data.refs) == num_samples
    assert len(eval_data.hyps) == num_samples
    assert songinfo.infer is not None

    if songinfo.infer.timings:
        assert len(songinfo.infer.timings) == num_samples
        timings = [
            {"start": line.start, "end": line.end} for line in songinfo.infer.timings
        ]
    else:
        timings = [{"start": 0, "end": songinfo.extract.duration}]

    assert songinfo.evaluate is not None

    uid = songinfo.extract.uid
    dataset = songinfo.extract.dataset_id
    language = songinfo.extract.language

    ref_tokens_all = [
        metrics.tokens_as_words(tokenizer(ref, language=songinfo.extract.language))
        for ref in songinfo.evaluate.refs
    ]
    hyp_tokens_all = [
        metrics.tokens_as_words(tokenizer(hyp, language=songinfo.extract.language))
        for hyp in songinfo.evaluate.hyps
    ]
    sample_rows = []
    edit_rows = []
    chunk_rows = []
    for sample_idx, (hyp_tokens, ref_tokens, alignment, timing) in enumerate(
        zip(hyp_tokens_all, ref_tokens_all, wo.alignments, timings)
    ):
        edit_counts, _error_counts = metrics.process_alignments(
            [ref_tokens], [hyp_tokens], [alignment]
        )
        e = edit_counts[WORD]
        en = edit_counts[NONLEXICAL]
        eb = edit_counts[BACKING]
        duration = (
            timing["end"] - timing["start"] if timing else songinfo.extract.duration
        )
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
            "start": timing["start"],
            "end": timing["end"],
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


def process_data(
    data: Dict[Tuple[str, str, str], List[Tuple[SongInfo, jiwer.WordOutput]]],
    system_keys: Tuple[str, str, str],
    num_workers: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process raw data into DataFrames for songs, chunks, and edits.
    data: Map from system->evaluation results
    """
    sample_rows, chunk_rows, edit_rows = [], [], []
    pool = Pool(num_workers)
    for system, sample_data in tqdm(list(data.items())):
        system_dict = dict(zip(system_keys, system))
        results = pool.starmap(song_to_rows, sample_data)
        for samples, chunks, edits in results:
            sample_rows.extend({**sample, **system_dict} for sample in samples)
            chunk_rows.extend({**chunk, **system_dict} for chunk in chunks)
            edit_rows.extend({**edit, **system_dict} for edit in edits)
    pool.close()
    pool.join()
    return pd.DataFrame(sample_rows), pd.DataFrame(chunk_rows), pd.DataFrame(edit_rows)
