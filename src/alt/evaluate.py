# stdlib
from collections import defaultdict
import logging
from pathlib import Path
import os
import regex as re
from typing import Dict, Iterable, List, Tuple

# third-party
from . import metrics
from .tokenizer import LyricsTokenizer, tokens_as_words
import jiwer
from tqdm import tqdm

# first-party
from . import util, render
from .alt_types import EvalConfig, EvalData, SongInfo, EvalSummary, Segment
from .render import ErrorRates

logger = logging.getLogger(__name__)


# Utility function to split a range (as start and end points)
# and returns a list of ranges (as lists [start, end) )
def split_range(start: int, end: int, splits: Iterable[int]) -> list[list[int]]:
    tmp = []
    out = []
    for idx in range(start, end):
        tmp.append(idx)
        if idx in splits:
            out.append(tmp)
            tmp = []
    if len(tmp) != 0:
        out.append(tmp)
    return out


# takes lyrics and returns a map of *token* indices to number of newlines at that index
def get_newline_idxs(raw_lyrics: str, language: str) -> Dict[int, int]:
    word_idx = -1
    newline_idxs = dict()
    tokenizer = metrics.LyricsTokenizer()
    for line in raw_lyrics.strip().split("\n"):
        if line.strip() != "":  # non-empty line
            word_idx += len(
                metrics.tokens_as_words(
                    tokenizer(normalize_lyrics(line), language=language)
                )
            )
        if word_idx == -1:
            continue
        if word_idx not in newline_idxs:
            newline_idxs[word_idx] = 1
        else:
            newline_idxs[word_idx] = newline_idxs[word_idx] + 1
    return newline_idxs


# Split alignment chunks into lines, giving a list of list of alignment chunks
# We do this because whisper output just gives one big line.
def split_chunks(
    chunks_in: List[jiwer.AlignmentChunk], newline_idxs: Dict[int, int]
) -> List[List[jiwer.AlignmentChunk]]:
    chunks_split = []  # list of list of chunks (lines of chunks)
    chunks_line = []  # temp variable for current line
    for chunk in chunks_in:
        # find newlines in the chunk
        chunk_newline_idxs = []
        for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
            if idx in newline_idxs:
                chunk_newline_idxs.append(idx)
        # no newlines (type irrelevant)
        if len(chunk_newline_idxs) == 0:
            # logger.debug("No newlines")
            chunks_line.append(chunk)
        # single newline at end (type irrelevant)
        elif (
            len(chunk_newline_idxs) == 1
            and (chunk.ref_end_idx - 1) == chunk_newline_idxs[0]
        ):
            # logger.debug("Single newline at end of chunk")
            chunks_line.append(chunk)
            chunks_split.append(chunks_line)
            chunks_line = []
        # multiple newlines or single newline in middle
        # don't handle substitutions and deletions
        elif chunk.type in ["substitute", "insert"]:
            # logger.debug("Substitution or insertion with newlines: pass through")
            chunks_line.append(chunk)
        # multiple newlines or single newline in middle, insertion or equal
        else:
            # split range into new chunks
            # logger.debug("Multiple newlines in equality or deletion")
            split_ref_idxs = split_range(
                chunk.ref_start_idx, chunk.ref_end_idx, chunk_newline_idxs
            )
            # logger.debug(f"Old chunk: {chunk}")
            for split_ref in split_ref_idxs:
                ref_start_idx = split_ref[0]
                ref_end_idx = ref_start_idx + len(split_ref)
                if chunk.type == "equal":
                    hyp_start_idx = chunk.hyp_start_idx + (
                        ref_start_idx - chunk.ref_start_idx
                    )
                    hyp_end_idx = hyp_start_idx + len(split_ref)
                    new_chunk = jiwer.AlignmentChunk(
                        type="equal",
                        ref_start_idx=ref_start_idx,
                        ref_end_idx=ref_end_idx,
                        hyp_start_idx=hyp_start_idx,
                        hyp_end_idx=hyp_end_idx,
                    )
                    # logger.debug(f"new chunk: {new_chunk}")
                else:  # delete
                    new_chunk = jiwer.AlignmentChunk(
                        type="delete",
                        ref_start_idx=ref_start_idx,
                        ref_end_idx=ref_end_idx,
                        hyp_start_idx=chunk.hyp_start_idx,
                        hyp_end_idx=chunk.hyp_end_idx,
                    )
                    # logger.debug(f"new chunk: {new_chunk}")
                chunks_line.append(new_chunk)
                # if we are on last chunk and it does not end in a newline,
                # we want to keep the line open
                # if we are not on last chunk OR we are on last chunk and it
                # last chunk does end in a newline, close the line
                if (ref_end_idx != chunk.ref_end_idx) or (
                    chunk.ref_end_idx - 1
                ) == chunk_newline_idxs[-1]:
                    chunks_split.append(chunks_line[:])
                    chunks_line = []
    if len(chunks_line) != 0:  # final clearup
        chunks_split.append(chunks_line)
    return chunks_split


def convert_jiwer(out: jiwer.WordOutput) -> Dict:
    return dict(
        wer=out.wer,
        mer=out.mer,
        wil=out.wil,
        wip=out.wip,
        hits=out.hits,
        substitutions=out.substitutions,
        insertions=out.insertions,
        deletions=out.deletions,
    )


# Returns a mapping (as list) hyp_index(int)->timestamp(float)
def get_hyp_timestamps(segments: list[Segment], language="en") -> list[float]:
    hyp_timestamps = []
    tokenizer = metrics.LyricsTokenizer()
    for segment in segments:
        num_words = len(
            metrics.tokens_as_words(tokenizer(segment.text, language=language))
        )
        for idx in range(num_words):
            start, end = segment.start, segment.end
            hyp_timestamps.append(start + (idx / num_words) * (end - start))
    return hyp_timestamps


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


# function that takes alignments + songinfo and returns list of list of chunks with timestamps
def get_html_verses(
    songinfo: SongInfo,
    wo: jiwer.WordOutput,
) -> List[List[List[render.HtmlAlignmentChunk]]]:
    assert songinfo.infer is not None
    assert songinfo.infer.whisper_result is not None
    chunks_split_l: list[list[list[jiwer.AlignmentChunk]]]
    hyp_timestamps_l: list[list[float]]
    lang = songinfo.extract.language
    if not songinfo.infer.timings:
        newline_idxs = get_newline_idxs(songinfo.extract.text, language=lang)
        chunks_split_l = [split_chunks(wo.alignments[0], newline_idxs)]
        hyp_timestamps_l = [
            get_hyp_timestamps(
                songinfo.infer.whisper_result,
                language=lang,
            )
        ]
    else:
        chunks_split_l = []
        hyp_timestamps_l = []
        assert songinfo.infer.timings
        for alignments, ref, hyp in zip(
            wo.alignments, songinfo.infer.timings, songinfo.infer.whisper_result
        ):
            newline_idxs = get_newline_idxs(
                ref.text, language=lang
            )  # both lines and verses have attr text
            chunks_split_l.append(split_chunks(alignments, newline_idxs))
            hyp_timestamps_l.append(get_hyp_timestamps([hyp], language=lang))  # type: ignore
    html_verses = []
    for refs, hyps, chunks_split, hyp_timestamps in zip(
        wo.references, wo.hypotheses, chunks_split_l, hyp_timestamps_l
    ):
        html_lines = []
        for line in chunks_split:
            html_line = []
            for chunk in line:
                if chunk.type == "delete":
                    ref_text = " ".join(refs[chunk.ref_start_idx : chunk.ref_end_idx])
                    hyp_text = "-" * len(ref_text)
                elif chunk.type == "insert":
                    hyp_text = " ".join(hyps[chunk.hyp_start_idx : chunk.hyp_end_idx])
                    ref_text = "+" * len(hyp_text)
                elif chunk.type == "equal":
                    ref_text = " ".join(refs[chunk.ref_start_idx : chunk.ref_end_idx])
                    hyp_text = ref_text
                elif chunk.type == "substitute":
                    ref_text = " ".join(refs[chunk.ref_start_idx : chunk.ref_end_idx])
                    hyp_text = " ".join(hyps[chunk.hyp_start_idx : chunk.hyp_end_idx])
                if len(hyp_timestamps) > 0:  # just in case
                    # also just in case
                    timestamp = max(
                        0.1,
                        hyp_timestamps[min(chunk.hyp_start_idx, len(hyp_timestamps) - 1)],
                    )
                else:
                    timestamp = 0
                html_chunk = render.HtmlAlignmentChunk(
                    chunk_type=chunk.type,
                    ref_text=ref_text,
                    hyp_text=hyp_text,
                    timestamp=timestamp,
                )
                html_line.append(html_chunk)
            html_lines.append(html_line)
        html_verses.append(html_lines)
    return html_verses


# takes list of songinfo, returns renderable summary
def get_html_summary(results: list[SongInfo]) -> render.HtmlSummary:
    song_summaries = []

    for songinfo in results:
        assert songinfo.evaluate is not None
        song_summary = render.HtmlSongSummary(
            song_id=songinfo.extract.song_id,
            uid=songinfo.extract.uid,
            language=songinfo.extract.language,
            wer=songinfo.evaluate.metrics["WER"],
            wer_near=songinfo.evaluate.metrics["WER_near"],
            wil=songinfo.evaluate.metrics["WIL"],
            hits=songinfo.evaluate.metrics["hits"],
            subs=songinfo.evaluate.metrics["substitutions"],
            dels=songinfo.evaluate.metrics["deletions"],
            ins=songinfo.evaluate.metrics["insertions"],
            nears=songinfo.evaluate.metrics["nears"],
            num_ref_tokens=songinfo.evaluate.metrics["total_len"],
        )
        song_summaries.append(song_summary)

    def mean_metric(metric: str, results: list[SongInfo]):
        return sum(songinfo.evaluate.metrics[metric] for songinfo in results) / len(  # type: ignore
            results
        )

    def mean_error_rates(results: list[SongInfo]):
        refs, hyps, langs = [], [], []
        for result in results:
            assert result.evaluate is not None
            assert len(result.evaluate.refs) == len(result.evaluate.hyps)
            refs.extend(result.evaluate.refs)
            hyps.extend(result.evaluate.hyps)
            langs.extend([result.extract.language] * len(result.evaluate.refs))
        total_metrics, _ = metrics.compute_metrics(refs, hyps, langs)
        return render.ErrorRates(
            wer=mean_metric("WER", results), wer_total=total_metrics["WER"]
        )

    lang_songinfo = defaultdict(list)
    for songinfo in results:
        lang_songinfo[songinfo.extract.language].append(songinfo)
    lang_error_rates = {
        lang: mean_error_rates(lang_results)
        for lang, lang_results in lang_songinfo.items()
    }

    ds_songinfo = defaultdict(list)
    for songinfo in results:
        ds_songinfo[songinfo.extract.dataset_id].append(songinfo)
    ds_error_rates = {
        ds: mean_error_rates(ds_results) for ds, ds_results in ds_songinfo.items()
    }

    lang_ds_songinfo: dict[str, dict[str, list[SongInfo]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for songinfo in results:
        lang_ds_songinfo[songinfo.extract.dataset_id][songinfo.extract.language].append(
            songinfo
        )
    lang_ds_error_rates: dict[str, dict[str, ErrorRates]] = {}
    for ds, lang_results in lang_ds_songinfo.items():
        lang_ds_error_rates[ds] = {}
        for lang, lang_ds_results in lang_results.items():
            lang_ds_error_rates[ds][lang] = mean_error_rates(lang_ds_results)

    total_rates = mean_error_rates(results)

    summary = render.HtmlSummary(
        title="",
        description="",
        lang_error_rates=lang_error_rates,
        ds_error_rates=ds_error_rates,
        lang_ds_error_rates=lang_ds_error_rates,
        wer_total=total_rates.wer_total,
        mean_wer=mean_metric("WER", results),
        mean_wil=mean_metric("WIL", results),
        mean_wer_near=mean_metric("WER_near", results),
        song_summaries=song_summaries,
    )
    return summary


tokenizer = LyricsTokenizer()


def eval_song(songinfo: SongInfo) -> Tuple[EvalData, jiwer.WordOutput]:
    assert songinfo.infer is not None
    if songinfo.infer.timings:
        hyps = [
            normalize_lyrics(segment.text) for segment in songinfo.infer.whisper_result
        ]
        refs = [normalize_lyrics(verse.text) for verse in songinfo.infer.timings]
        tokenized_refs = [
            tokens_as_words(tokenizer(ref, songinfo.extract.language)) for ref in refs
        ]
        if any(len(tokens) == 0 for tokens in tokenized_refs):
            logger.warning(f"Empty ref: {songinfo.extract.uid}")
        refs, hyps = map(
            list,
            zip(
                *(
                    (ref, hyp)
                    for ref, hyp, tokens_ref in zip(refs, hyps, tokenized_refs)
                    if tokens_ref
                )
            ),
        )
        song_metrics, wo = metrics.compute_metrics(refs, hyps, songinfo.extract.language)
    # elif songinfo.infer.lines:
    #     hyps = [
    #         normalize_lyrics(segment.text) for segment in songinfo.infer.whisper_result
    #     ]
    #     refs = [normalize_lyrics(line.text) for line in songinfo.infer.lines]
    #     song_metrics, wo = metrics.compute_metrics(refs, hyps, songinfo.extract.language)

    else:
        hyp_text = "\n".join(segment.text for segment in songinfo.infer.whisper_result)
        hyps = [normalize_lyrics(hyp_text)]
        refs = [normalize_lyrics(songinfo.extract.text)]
        song_metrics, wo = metrics.compute_metrics(
            refs,
            hyps,
            languages=[songinfo.extract.language],
        )

    return EvalData(metrics=song_metrics, refs=refs, hyps=hyps), wo


def run_eval(infer_dir: Path, eval_dir: Path, cfg: EvalConfig):
    """
    Executes an evaluation process on the inference results and generates a report.
    Args:
        infer_dir (str): The directory path containing the inference results to be evaluated.
        eval_dir (str): The directory path where the evaluation results and reports will be saved.
        title (str): The title for the evaluation report, used for labeling and identification.
    Returns: None
    """
    logger.info(f"Running evaluation: {eval_dir}")
    eval_results = []
    summary = util.read_summary(infer_dir / "summary.json")
    assert summary.infer is not None

    for uid in tqdm(summary.infer.uids):
        songinfo: SongInfo = util.read_pz(infer_dir / (uid + ".pz"))
        # Run evaluation
        assert songinfo.infer is not None
        assert songinfo.infer.whisper_result is not None
        eval_data, wo = eval_song(songinfo)
        # update songinfo and write out JSON result
        songinfo.evaluate = eval_data
        eval_results.append(songinfo)
        util.write_pz(eval_dir / (uid + ".pz"), songinfo)
        # Render html for song
        if cfg.render:
            html_lines = get_html_verses(songinfo, wo)
            render.render_song_html(songinfo, html_lines, str(eval_dir))
            util.write_json_dataclass(songinfo, eval_dir / (uid + ".json"))

    # Render HTML
    # TODO: When analysis notebooks no longer use HTML summary, we can put this in render block
    html_summary = get_html_summary(eval_results)
    html_summary.title = os.path.basename(os.path.dirname(eval_dir))
    html_summary.description = cfg.description
    if cfg.render:
        render.render_summary_html(str(eval_dir / "summary.html"), html_summary)
        util.write_json_dataclass(html_summary, eval_dir / "summary_eval.json")
    util.write_pz(eval_dir / "summary_eval.pz", html_summary)

    # Write JSON summary
    summary.evaluate = EvalSummary(uids=summary.infer.uids, info=dict(), cfg=cfg)
    util.write_json_dataclass(summary, eval_dir / "summary.json")


def run_eval_named(
    pipeline_name: str,
    cfg: EvalConfig,
    infer_name: str | None = None,
    force=False,
):
    eval_dir = Path("build") / pipeline_name / "eval"
    os.makedirs(eval_dir, exist_ok=True)

    infer_name = infer_name if infer_name is not None else pipeline_name
    infer_dir = Path("build") / infer_name / "infer"

    summary_file = eval_dir / "summary.json"
    if not force and os.path.exists(summary_file):
        logger.info(
            f"Summary file already exists at {summary_file}. Skipping evaluation."
        )
        return

    run_eval(cfg=cfg, eval_dir=eval_dir, infer_dir=infer_dir)
