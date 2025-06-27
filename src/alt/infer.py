# stdlib
import logging
import os
import concurrent.futures
import copy
from typing import Tuple, TYPE_CHECKING
from pathlib import Path
from multiprocessing import Pool

# third-party
import librosa
import numpy as np
import ctranslate2
from tqdm import tqdm
import pandas as pd

# first-party
from . import util
from .evaluate import eval_song_new, song_to_rows
from .alt_types import (
    Song,
    VadOptions,
    AsrTask,
    AsrSample,
    VadData,
    InferResult,
    InferConfig,
    Result,
    InferLevel,
    LongFormAlgo,
)
from .asr_pool import AsrPool

logger = logging.getLogger(__name__)


# Based on get_speech_timestamps() in vad.py
# https://github.com/SYSTRAN/faster-whisper/blob/11fd8ab3011dedd05d83e215a90cda63da7de653/faster_whisper/vad.py#L45
def get_speech_timestamps_rms(
    audio: np.ndarray,
    vad_options: VadOptions,
    window_size_samples=512,
    sampling_rate: int = 16000,
) -> Tuple[list[dict], np.ndarray]:
    """This method is used for splitting long audio of separated vocals

    Args:
      audio: One dimensional float array of separated vocals
      vad_options: Binarization options (thresholds, min/max durations).
      sampling rate: Sampling rate of the audio.
      window_size_samples: hop length of rms feature (frame length is fixed)

    Returns:
      List of dicts containing begin and end samples of each speech chunk.
    """

    onset = vad_options.onset
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    speech_pad_ms = vad_options.speech_pad_ms
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    rms = np.mean(
        librosa.feature.rms(y=audio, frame_length=2048, hop_length=window_size_samples),
        axis=0,
    )
    probs = rms / np.max(rms)

    triggered = False
    speeches = []
    current_speech = {}
    offset = vad_options.offset

    # to save potential segment end (and tolerate some silence)
    temp_end = 0
    # to save potential segment limits in case of maximum segment size reached
    prev_end = next_start = 0

    # Create speech segments using onset and offset thresholds
    # Merging to avoid minimum silence duration
    for i, speech_prob in enumerate(probs):
        if (speech_prob >= onset) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= onset) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        # Switch off if current segment longer than max_speech_duration_s
        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                # previously reached silence (< neg_thres) and is still not speech (< thres)
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                # No prior silence available: perform min-cut on the second half of the segment
                start_frame = current_speech["start"] // window_size_samples
                end_frame = i + 1  # include the current frame in the segment
                segment_scores = probs[start_frame:end_frame]
                # Only consider the second half of the current segment
                second_half_start = len(segment_scores) // 2
                search_segment = segment_scores[second_half_start:]
                # Find the minimum value in the second half
                min_val = search_segment.min()
                min_indices = np.where(search_segment == min_val)[0]
                # If there are multiple equal minimums, pick the last one
                min_index_in_half = min_indices[-1]
                # Adjust the index to match the full segment's index
                chosen_frame = start_frame + second_half_start + min_index_in_half
                # Compute the corresponding sample index for the new cut point
                min_cut_sample = window_size_samples * chosen_frame
                # End the current segment at the min-cut
                current_speech["end"] = min_cut_sample
                speeches.append(current_speech)
                # Start a new segment from the min-cut sample
                current_speech = {"start": min_cut_sample}
                triggered = True
                prev_end = next_start = temp_end = 0
                continue

        # switch off if prob goes below offset and
        # and has been for longer than min_silence_duration_ms
        if (speech_prob < offset) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            # condition to avoid cutting in very short silence
            if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech["end"] = temp_end
                # Add the chunk if it's longer than min_speech_duration_ms
                if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                continue

    # special handling for last chunk
    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    # Add speech_pad_ms to start of first chunk and end of last chunk
    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    return speeches, probs


def merge_segments(
    segments_list: list[dict], max_length_s: int, sampling_rate: int = 16000
) -> list[dict]:
    """
    Merges a list of segments into larger segments based on a maximum length constraint.

    Args:
        segments_list (list of dict): A list of segments where each segment is a dictionary
                                      with "start" and "end" keys representing the start and
                                      end **sample indices** of the segment.
        max_length_s (int): The maximum length (in seconds) for the merged segments.
        sampling_rate (int, optional): The sampling rate in Hz. Default is 16000.

    Returns:
        list of dict: A list of merged segments where each segment is a dictionary with
                      "start", "end", and "segments" keys. The "segments" key contains
                      a list of tuples representing the start and end times of the original
                      segments that were merged.
    """
    if not segments_list:
        return []

    curr_end = 0
    seg_idxs = []
    merged_segments = []
    chunk_length = max_length_s * sampling_rate

    curr_start = segments_list[0]["start"]

    for seg in segments_list:
        if seg["end"] - curr_start > chunk_length and curr_end - curr_start > 0:
            merged_segments.append(
                {
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                }
            )
            curr_start = seg["start"]
            seg_idxs = []
        curr_end = seg["end"]
        seg_idxs.append((seg["start"], seg["end"]))
    # add final
    merged_segments.append(
        {
            "start": curr_start,
            "end": curr_end,
            "segments": seg_idxs,
        }
    )
    return merged_segments


def get_audio_path(song: Song, audio: str) -> Path:
    if audio == "original":
        audio_path = song.audio_path
    elif audio == "vocal_stem":
        if song.stem_dir is None:
            raise ValueError("No stem directory")
        audio_path = song.stem_dir / "vocals.wav"
    else:
        audio_path = util.separation_dir() / audio / song.uid / "vocals.mp3"
    return audio_path


def get_transcription_task(song: Song, cfg: InferConfig) -> InferResult:
    """
    Transcription task - passable to the pool
    Infer Result - intermediate result with actual model outputs missing, which will be provided by pool
    """
    # Find audio file
    target_audio_path = get_audio_path(song, cfg.audio)
    if not target_audio_path.exists():
        raise ValueError(f"Target audio path does not exist: {target_audio_path}")

    # Find language if wanted
    language = song.language if cfg.add_lang else None
    transcribe_args = copy.deepcopy(cfg.transcribe_args)
    # NOTE: clip_timestamps has special handling within whisper_pool
    if cfg.model_type == "whisper":
        transcribe_args["language"] = language
    else:
        raise ValueError("Only model type 'whisper' supported")

    # Assemble transcription task, depending on inference granularity level
    # If not inferring at "song" level, we need to keep track of the reference
    # transcript for each segment we are passing

    if cfg.level == "song":
        # Determine clip timestamps
        if cfg.algo == LongFormAlgo.RMSVAD:
            assert cfg.vad_audio is not None
            assert cfg.vad_options is not None
            vad_audio_path = get_audio_path(song, cfg.vad_audio)
            assert vad_audio_path.exists()
            vocals_audio, _sr = librosa.load(vad_audio_path, sr=16000)

            speech_segments, rms = get_speech_timestamps_rms(
                vocals_audio,
                vad_options=cfg.vad_options,
                sampling_rate=16000,
                window_size_samples=512,
            )
            line_timestamps = merge_segments(
                speech_segments, max_length_s=30, sampling_rate=16000
            )
            # use default integer type to write out json
            for segment in line_timestamps:
                segment["start"] = int(segment["start"])
                segment["end"] = int(segment["end"])
                segment["segments"] = [
                    (int(seg[0]), int(seg[1])) for seg in segment["segments"]
                ]

            vad_data = VadData(
                scores=rms.tolist(),
                segments=copy.deepcopy(line_timestamps),
                sampling_rate=16000,
                window_size_samples=512,
                audio_path=vad_audio_path,
            )
            for segment in line_timestamps:
                segment.pop("segments")
            # Sequential inference pipeline expects flat list [s1,e1,s2,e2...]
            if not cfg.batched:
                line_timestamps = [
                    timestamp
                    for segment in line_timestamps
                    for timestamp in (
                        segment["start"] / 16000,
                        segment["end"] / 16000,
                    )
                ]
            # for timestamp in line_timestamps:
            #     logging.info(f"Timestamp: {timestamp}")
        else:
            line_timestamps = None if cfg.batched else "0"
            vad_data = None
        transcribe_args["clip_timestamps"] = line_timestamps
        task = AsrTask(
            uid=song.uid,
            audio_path=target_audio_path,
            transcribe_kwargs=dict(
                **transcribe_args,
            ),
        )
        return InferResult(audio_path=target_audio_path, asr_task=task, vad_data=vad_data)
    else:
        if cfg.level == InferLevel.GROUPED_LINE:
            timings = song.grouped_lines
        elif cfg.level == InferLevel.MERGED_LINE:
            timings = song.merged_lines
        else:
            raise ValueError(f"Unsupported infer level: {cfg.level}")
        # Construct transcription task
        transcribe_args["clip_timestamps"] = [
            dict(start=int(sample.start * 16000), end=int(sample.end * 16000))
            for sample in timings
        ]
        task = AsrTask(
            uid=song.uid,
            audio_path=target_audio_path,
            transcribe_kwargs=dict(**transcribe_args),
        )
        return InferResult(audio_path=target_audio_path, asr_task=task)


def run_infer(
    songs: list[Song],
    cfg: InferConfig,
) -> list[Result]:
    # Config validation
    if not (cfg.level == InferLevel.SONG and cfg.algo == LongFormAlgo.NATIVE):
        logger.warning(
            "Only long-form inference with native algo requires non-batched processing. Switching to batched"
        )
        cfg.batched = True
    logging.info(f"Running inference at {cfg.level} level")

    # Filtering
    songs_filtered: dict[str, Song] = {}
    for song in songs:
        if cfg.datasets and song.dataset_id not in cfg.datasets:
            continue
        if cfg.splits and song.split not in cfg.splits:
            continue
        if cfg.langs and song.language not in cfg.langs:
            continue
        songs_filtered[song.uid] = song
    logger.info(f"Filtered out {len(songs) - len(songs_filtered)}/{len(songs)} songs")

    # Song -> InferResult (half-constructed)
    logger.info("Constructing inference tasks")
    with Pool(cfg.num_workers) as pool:
        infer_results = pool.starmap(
            get_transcription_task, tqdm([(s, cfg) for s in songs_filtered.values()])
        )
        infer_results = {res.asr_task.uid: res for res in infer_results}

    # Actually run ASR
    logger.info("Constructing ASR pool")
    cache_dir = None
    if cfg.cache_dir:
        cache_dir = util.project_root() / "cache" / cfg.cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    pool = AsrPool(
        model_name=cfg.model_name,
        model_type=cfg.model_type,
        batched=cfg.batched,
        cache_dir=cache_dir,
    )
    asr_results = pool.transcribe_batch(
        [result.asr_task for result in infer_results.values()]
    )

    # Piece together results
    outputs: list[Result] = []
    for uid, asr_result in asr_results.items():
        infer_result = infer_results[uid]
        infer_result.asr_output = asr_result

        if cfg.level == InferLevel.SONG:
            song = songs_filtered[uid]
            infer_result.asr_samples = [
                AsrSample(
                    ref=song.text,
                    hyp="\n".join(seg.text for seg in asr_result),
                    start=0,
                    end=song.duration_secs,
                )
            ]
        elif cfg.level in (InferLevel.MERGED_LINE, InferLevel.GROUPED_LINE):
            lines = (
                songs_filtered[uid].merged_lines
                if cfg.level == InferLevel.MERGED_LINE
                else songs_filtered[uid].grouped_lines
            )
            infer_result.asr_samples = [
                AsrSample(ref=line.text, hyp=segment.text, start=line.start, end=line.end)
                for line, segment in zip(lines, asr_result)
            ]
        else:
            raise ValueError(f"Unsupported infer level: {cfg.level} for uid: {uid}")
        outputs.append(Result(song=songs_filtered[uid], infer=infer_result))
    return outputs


def get_metrics(df: pd.DataFrame) -> dict:
    metrics = df[["hit", "ins", "sub", "del", "duration"]]
    totals = metrics.sum()
    totals["err"] = totals["ins"] + totals["sub"] + totals["del"]
    totals["len"] = totals["sub"] + totals["del"] + totals["hit"]
    totals["wer"] = totals["err"] / totals["len"]
    return {
        "ins": totals["ins"] / totals["len"],
        "sub": totals["sub"] / totals["len"],
        "del": totals["del"] / totals["len"],
        "hit": totals["hit"] / totals["len"],
        "len": int(totals["len"]),
        "wer": totals["wer"],
        "dur": totals["duration"],
    }


def summarize(df: pd.DataFrame) -> dict:
    res = get_metrics(df)
    res["datasets"] = {}
    for dataset, df_dataset in df.groupby("dataset"):
        res["datasets"][dataset] = get_metrics(df_dataset)
    return res


def run_infer_named(
    name: str,
    songs_path: Path,
    cfg: InferConfig,
    force: bool = True,
    random_seed=42,
    debug: bool = False,
) -> Path:
    infer_dir = util.infer_dir() / name
    os.makedirs(infer_dir, exist_ok=True)

    songs = util.read_datafile(songs_path, Song)
    if debug:
        logging.info(f"Debug mode: Limited to {5} songs")
        songs = songs[:5]

    output_path = infer_dir / "output.gz"
    if not force and os.path.exists(output_path):
        logger.info(
            f"Inference output already exists at {output_path}. Skipping inference."
        )
        infer_data = util.read_datafile(output_path, Result)
    else:
        logger.info(f"Running inference: {infer_dir}")
        logger.info(f"Random seed: {random_seed}")
        ctranslate2.set_random_seed(random_seed)
        infer_data = run_infer(songs, cfg)

    # Run evaluation + write out data
    logger.info(f"Evaluating with {cfg.num_workers} workers")
    with Pool(cfg.num_workers) as pool:
        eval_data = pool.map(eval_song_new, tqdm(infer_data))
        for data, song_eval in zip(infer_data, eval_data):
            data.eval = song_eval
    with util.DataFile(output_path, Result, "w") as f:
        f.write_all(iter(infer_data))

    # Convert evaluations to dataframes
    logger.info("Writing out dataframes")
    samples, chunks, edits = [], [], []
    with Pool(cfg.num_workers) as pool:
        for sample_rows, chunk_rows, edit_rows in pool.map(
            song_to_rows, tqdm(infer_data)
        ):
            samples.extend(sample_rows)
            chunks.extend(chunk_rows)
            edits.extend(edit_rows)
    dataframes = (pd.DataFrame(samples), pd.DataFrame(chunks), pd.DataFrame(edits))
    util.write_pz(infer_dir / "dataframes.pz", dataframes)

    # Get summary metrics
    # Headline WER, sub/del/ins rates, duration
    logger.info("Writing out summary")
    samples, chunks, edits = [], [], []
    metrics = summarize(dataframes[0])
    util.model_dump_json(infer_dir / "metrics.json", metrics)

    if debug:
        logging.info("Debug mode: write out all JSON files")
        for song in infer_data:
            util.model_dump_json(infer_dir / (song.song.uid + ".json"), song)

    return infer_dir


def run_infer_named_process(*args, **kwargs) -> Path:
    """
    Run the `run_infer_named` function in a separate process using a ProcessPoolExecutor.

    This function submits the `run_infer_named` function to a process pool executor with the provided arguments
    and keyword arguments, and waits for the result. Any exceptions raised by `run_infer_named` will be rethrown.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f = executor.submit(run_infer_named, *args, **kwargs)
        return f.result()  # rethrow any exceptions
