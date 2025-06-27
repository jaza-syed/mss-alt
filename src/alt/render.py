# stdlib
import argparse
from dataclasses import dataclass
import errno
from itertools import accumulate
import os
import logging
from typing import Iterable
from pathlib import Path
from multiprocessing import Pool

# third-party
import numpy as np
import matplotlib.pyplot as plt
from jiwer import AlignmentChunk
import pandas as pd
from tqdm import tqdm

# first-party
from . import metrics, util, html_templates
from .alt_types import VadData, Result, AsrSegment, InferLevel
from .tokenizer import Token, LINE, WORD
from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


@dataclass
class HtmlAlignmentChunk:
    chunk_type: str  # E/S/I/D
    ref_text: str  # ++ for insertion
    hyp_text: str  # -- for deletiouun
    timestamp: float


@dataclass
class HtmlSongSummary:
    song_id: str
    uid: str
    language: str
    wer: float
    wil: float
    wer_near: float
    hits: int
    subs: int
    ins: int
    dels: int
    nears: int
    num_ref_tokens: int


@dataclass
class HtmlSummary:
    title: str
    description: str
    songs: pd.DataFrame


def split_range(start: int, end: int, splits: Iterable[int]) -> list[tuple[int, int]]:
    """Return a list of (start, end) pairs split at each index in splits."""
    splits = sorted(set(splits))
    result = []
    last = start
    for s in splits:
        if start <= s < end:
            result.append((last, s + 1))
            last = s + 1
    if last < end:
        result.append((last, end))
    return result


def split_chunks(
    chunks_in: list[AlignmentChunk], newline_idxs: set[int]
) -> list[list[AlignmentChunk]]:
    lines = []
    current_line = []
    for chunk in chunks_in:
        # Only split if chunk is not an insertion (insertions have no newlines)
        if chunk.type in ("equal", "delete", "substitute"):
            # Find all newlines in this chunk's reference span
            chunk_newlines = [
                idx
                for idx in range(chunk.ref_start_idx, chunk.ref_end_idx)
                if idx in newline_idxs
            ]
            if chunk_newlines:
                # Split the chunk at each newline
                for ref_start, ref_end in split_range(
                    chunk.ref_start_idx, chunk.ref_end_idx, chunk_newlines
                ):
                    # Calculate hyp indices for each split
                    if chunk.type == "equal":
                        hyp_start = chunk.hyp_start_idx + (
                            ref_start - chunk.ref_start_idx
                        )
                        hyp_end = hyp_start + (ref_end - ref_start)
                    elif chunk.type == "substitute":
                        # Substitutions are always 1:1
                        hyp_start = chunk.hyp_start_idx + (
                            ref_start - chunk.ref_start_idx
                        )
                        hyp_end = hyp_start + (ref_end - ref_start)
                    else:  # delete
                        hyp_start = chunk.hyp_start_idx
                        hyp_end = chunk.hyp_end_idx
                    new_chunk = AlignmentChunk(
                        type=chunk.type,
                        ref_start_idx=ref_start,
                        ref_end_idx=ref_end,
                        hyp_start_idx=hyp_start,
                        hyp_end_idx=hyp_end,
                    )
                    current_line.append(new_chunk)
                    # If this split ends at a newline, start a new line
                    if (ref_end - 1) in newline_idxs:
                        lines.append(current_line)
                        current_line = []
            else:
                current_line.append(chunk)
        else:
            # Insertions: just add to current line
            current_line.append(chunk)
    if current_line:
        lines.append(current_line)
    return lines


# Returns a mapping (as list) hyp_index(int)->timestamp(float)
def get_hyp_timestamps(segments: list[AsrSegment], language="en") -> list[float]:
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


# Get indices of the word before a newline in the word-token list
def word_newline_indices(tokens: list[Token]) -> np.ndarray:
    # Filter out consecutive newlines
    filtered_tokens = []
    prev_was_newline = False
    for token in tokens:
        if LINE in token.tags:
            if not prev_was_newline:
                filtered_tokens.append(token)
            prev_was_newline = True
        else:
            filtered_tokens.append(token)
            prev_was_newline = False

    word_indices = np.fromiter(
        accumulate(
            filtered_tokens,
            lambda idx, tok: idx + 1 if WORD in tok.tags else idx,
            initial=-1,
        ),
        dtype=int,
    )
    newlines = np.fromiter(
        (idx for idx, token in enumerate(filtered_tokens) if LINE in token.tags),
        dtype=int,
    )
    return word_indices[newlines]


# function that takes alignments + songinfo and returns list of list of chunks with timestamps
def get_html_samples(
    data: Result,
) -> list[list[list[HtmlAlignmentChunk]]]:
    chunks_split_l: list[list[list[AlignmentChunk]]]
    hyp_timestamps_l: list[list[float]]

    assert data.eval is not None
    eval_result = data.eval

    # in theory we should have been able to do this with asr_samples, but the information
    # is only encoded implicitly
    # the difference is line-level evaluations have a 1-1 segment to sampple mapping
    # whereas for songs there are 1 sample and many segments, so we have to do the join
    if data.infer.infer_level == InferLevel.SONG:
        newline_idxs = word_newline_indices(eval_result.ref_tokens[0])
        chunks_split_l = [split_chunks(eval_result.wo.alignments[0], set(newline_idxs))]
        hyp_timestamps_l = [
            get_hyp_timestamps(data.infer.asr_output, language=data.song.language)
        ]
    else:
        chunks_split_l = []
        hyp_timestamps_l = []
        # short-form has one segment per sample
        for alignments, ref_tokens, segment in zip(
            eval_result.wo.alignments, eval_result.ref_tokens, data.infer.asr_output
        ):
            newline_idxs = word_newline_indices(ref_tokens)
            chunks_split_l.append(split_chunks(alignments, set(newline_idxs)))
            hyp_timestamps_l.append(
                get_hyp_timestamps([segment], language=data.song.language)
            )  # type: ignore

    html_samples = []
    for refs, hyps, chunks_split, hyp_timestamps in zip(
        eval_result.wo.references,
        eval_result.wo.hypotheses,
        chunks_split_l,
        hyp_timestamps_l,
    ):
        html_lines = []
        for line in chunks_split:
            html_line = []
            ref_text = ""
            hyp_text = ""
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
                html_chunk = HtmlAlignmentChunk(
                    chunk_type=chunk.type,
                    ref_text=ref_text,
                    hyp_text=hyp_text,
                    timestamp=timestamp,
                )
                html_line.append(html_chunk)
            html_lines.append(html_line)
        html_samples.append(html_lines)
    return html_samples


def total_metrics(df: pd.DataFrame) -> dict[str, float]:
    totals = df[["subs", "dels", "inss", "hits"]].sum()
    ref_len = totals["subs"] + totals["dels"] + totals["hits"]
    return {
        "wer": (totals["subs"] + totals["dels"] + totals["inss"]) / ref_len,
        "dr": totals["dels"] / ref_len,
        "sr": totals["subs"] / ref_len,
        "ir": totals["inss"] / ref_len,
    }


# write summary table out to a html file
def render_summary_html(
    output_fn: Path, results: list[Result], title: str, description: str = ""
):
    song_rows = []
    for res in results:
        assert res.eval is not None
        song_rows.append(
            {
                "uid": res.song.uid,
                "wer": res.eval.metrics["WER"],
                "subs": res.eval.metrics["substitutions"],
                "dels": res.eval.metrics["deletions"],
                "inss": res.eval.metrics["insertions"],
                "hits": res.eval.metrics["hits"],
                "lang": res.song.language,
                "dataset": res.song.dataset_id,
            }
        )
    df_songs = pd.DataFrame(song_rows).sort_values("wer")
    ds_metrics = {
        dataset: total_metrics(df) for dataset, df in df_songs.groupby("dataset")
    }
    summary_metrics = total_metrics(df_songs)
    metric_order = ["wer", "sr", "dr", "ir"]
    num_columns = len(metric_order) + 1

    # Render template
    context = {
        "title": title,
        "description": description,
        "ds_metrics": ds_metrics,
        "summary_metrics": summary_metrics,
        "num_columns": num_columns,
        "metric_order": metric_order,
        "df_songs": df_songs,
    }

    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader("."),  # Not used, but required
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.from_string(html_templates.SUMMARY_HTML_TEMPLATE)

    html = template.render(**context)

    # Write to file
    with open(output_fn, "w") as fout:
        fout.write(html)


def vad_plot(fn: Path, vad_result: VadData) -> None:
    fig, ax = plt.subplots()
    active = np.zeros(len(vad_result.scores))
    for segment in vad_result.segments:
        start_frame = segment["start"] // vad_result.window_size_samples
        end_frame = segment["end"] // vad_result.window_size_samples
        active[start_frame:end_frame] = 1
    time = (
        np.arange(len(active)) * vad_result.window_size_samples / vad_result.sampling_rate
    )
    ax.plot(time, vad_result.scores, label="RMS Vocals")
    ax.plot(time, active, label="Segments")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(fn)
    plt.close(fig)


def symlink_force(target, link_name, *args, **kwargs) -> None:
    try:
        os.symlink(target, link_name, *args, **kwargs)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def render_song_html(
    songdata: Result,
    eval_dir: Path,
) -> None:
    """
    Generates an HTML file that visualizes the transcription results with audio playback functionality.
    The output path is `eval_dir / songdata.song.uid + '.html'`
    """
    html_samples = get_html_samples(songdata)

    song = songdata.song
    infer = songdata.infer

    html_fn = eval_dir / (song.uid + ".html")
    html_audio_fn = eval_dir / (song.uid + ".mp3")
    symlink_force(song.audio_path.absolute(), html_audio_fn)
    if song.audio_path != infer.audio_path:
        html_vocals_fn = eval_dir / f"{song.uid}_target.mp3"
        symlink_force(infer.audio_path.absolute(), html_vocals_fn)
    else:
        html_vocals_fn = None

    raw_lyrics_fn = os.path.join(eval_dir, song.uid + ".lyrics.txt")
    with open(raw_lyrics_fn, "w") as fout:
        fout.write(song.text)

    vad_plot_fn = None
    if infer.vad_data is not None:
        vad_plot_fn = song.uid + "_vad.svg"
        vad_plot(eval_dir / vad_plot_fn, vad_result=infer.vad_data)

    util.model_dump_json(eval_dir / f"{song.uid}.json", songdata)

    # Prepare data for template
    context = {
        "song": song,
        "infer": infer,
        "html_audio_fn": html_audio_fn,
        "html_vocals_fn": html_vocals_fn if html_vocals_fn else None,
        "raw_lyrics_fn": raw_lyrics_fn,
        "vad_plot_fn": vad_plot_fn,
        "html_samples": html_samples,
    }
    env = Environment(
        loader=FileSystemLoader("."),  # Not used, but required
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.from_string(html_templates.SONG_HTML_TEMPLATE)
    html = template.render(**context)

    with open(html_fn, "w") as fout:
        fout.write(html)


def render_songs(
    output_dir: Path,
    songs: list[Result],
    title: str,
    description: str = "",
    num_workers: int = 20,
) -> None:
    logger.info(f"Rendering HTML {output_dir}")
    render_summary_html(
        output_dir / "summary.html", songs, title=title, description=description
    )

    if num_workers > 1:
        with Pool(num_workers) as pool:
            pool.starmap(
                render_song_html,
                tqdm(((song, output_dir) for song in songs), total=len(songs)),
            )
    else:
        for song in tqdm(songs):
            render_song_html(song, output_dir)
