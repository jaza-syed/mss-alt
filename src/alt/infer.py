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
from tqdm import tqdm
import ctranslate2

# first-party
from . import util
from .alt_types import (
    VadOptions,
    InferConfig,
    InferData,
    VadResult,
    SongInfo,
    InferSummary,
    TranscriptionTask,
)


# AsrPool import espnet.bin.s2t_inference which is very slow, this allows us
# to only load it when we need it while maintaining type checking
if TYPE_CHECKING:
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


def get_transcription_task(
    songinfo: SongInfo, cfg: InferConfig
) -> Tuple[TranscriptionTask, InferData] | None:
    uid = songinfo.extract.uid
    # Find audio file
    if cfg.infer_target == "original":
        audio_fn = songinfo.extract.audio_fn
    elif cfg.infer_target == "separated_vocals":
        assert songinfo.preproc is not None
        if songinfo.preproc.vocals_fn is None:
            raise ValueError(f"No vocals_fn: {uid}")
        assert songinfo.preproc.vocals_fn is not None
        audio_fn = songinfo.preproc.vocals_fn
    elif cfg.infer_target == "stem":
        if "stem_dir" not in songinfo.extract.extra:
            raise ValueError(f"No stem dir: {uid}")
        audio_fn = Path(songinfo.extract.extra["stem_dir"]) / "vocals.wav"
    else:
        raise NotImplementedError(f"Unsupported infer_target: {cfg.infer_target}")
    infer_data = InferData(audio_fn=audio_fn, infer_target=cfg.infer_target)

    # Find language if wanted
    language = songinfo.extract.language if cfg.add_lang else None
    transcribe_args = copy.deepcopy(cfg.transcribe_args)
    # NOTE: clip_timestamps has special handling within whisper_pool
    if cfg.model_type == "whisper":
        transcribe_args["language"] = language
    else:
        raise ValueError("Only model type 'whisper' supported")

    # Assemble transcription task, depending on inference granularity level
    # If not inferring at "song" level, we need to keep track of the reference
    # transcript for each segment we are passing
    from .asr_pool import TranscriptionTask

    if cfg.infer_level == "song":
        # Determine clip timestamps
        if cfg.vocal_rms_vad:
            assert songinfo.preproc is not None
            assert songinfo.preproc.vocals_fn is not None
            vocals_audio, _sr = librosa.load(songinfo.preproc.vocals_fn, sr=16000)
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
            segments = copy.deepcopy(line_timestamps)
            for segment in segments:
                segment["start"] = int(segment["start"])
                segment["end"] = int(segment["end"])
                segment["segments"] = [
                    (int(seg[0]), int(seg[1])) for seg in segment["segments"]
                ]

            infer_data.vad_result = VadResult(
                scores=rms,
                segments=segments,
                sampling_rate=16000,
                window_size_samples=512,
            )
            # Sequential inference pipeline expects flat list [s1,e1,s2,e2...]
            if cfg.model_type == "whisper" and not cfg.batched:
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
            line_timestamps = (
                "0" if (cfg.model_type == "whisper" and not cfg.batched) else None
            )
        transcribe_args["clip_timestamps"] = line_timestamps
        return TranscriptionTask(
            uid=songinfo.extract.uid,
            audio_path=audio_fn,
            transcribe_kwargs=dict(
                **transcribe_args,
            ),
        ), infer_data
    else:
        if cfg.infer_level == "grouped_line":
            timings = songinfo.extract.grouped_lines
            # logging.info(f"Evaluating at grouped line level: {uid}")
        elif cfg.infer_level == "merged_line":
            timings = songinfo.extract.merged_lines
            # logging.info(f"Evaluating at merged line level: {uid}")
        else:
            raise ValueError(f"Unsupported infer level: {cfg.infer_level}")
        if not timings:
            logger.error(f"Infer level '{cfg.infer_level}' but no timings: {uid}")
            return None
        # Construct transcription task
        transcribe_args["clip_timestamps"] = [
            dict(start=int(sample.start * 16000), end=int(sample.end * 16000))
            for sample in timings
        ]
        return TranscriptionTask(
            uid=songinfo.extract.uid,
            audio_path=audio_fn,
            transcribe_kwargs=dict(**transcribe_args),
            timings=timings,
        ), infer_data


# preproc_dir: path to directory contain json files created by `extract.py`
# outpath: path to directory containing updated json files with evaluation results in key eval
def run_infer(
    preproc_dir: Path,
    infer_dir: Path,
    cfg: InferConfig,
    cached: bool = False,
    seed: int = 42,
):
    logger.info(f"Running inference: {infer_dir}")
    # Config validation
    if cfg.infer_level in ["grouped_line", "merged_line"] and not cfg.batched:
        cfg.batched = True
        logger.warning("Line level inference require batched processing, switching.")
    if cfg.vocal_rms_vad and not cfg.batched:
        cfg.batched = True
        logger.warning("vocal_rms_vad used, switch to batched processing")
    if cfg.model_type not in ["whisper"]:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")

    # Read / filter songinfos
    summary = util.read_summary(preproc_dir / "summary.json")
    assert summary.preproc is not None
    songinfos: dict[str, SongInfo] = dict()
    tasks = []
    for uid in summary.preproc.uids:
        songinfo: SongInfo = util.read_pz(preproc_dir / (uid + ".pz"))
        # filters
        if cfg.datasets and songinfo.extract.dataset_id not in cfg.datasets:
            continue
        if cfg.splits and songinfo.extract.split not in cfg.splits:
            continue
        if cfg.langs and songinfo.extract.language not in cfg.langs:
            continue
        songinfos[songinfo.extract.uid] = songinfo

    # Construct tasks
    logger.info("Inference: constructing tasks")
    with Pool(int(os.getenv("NUM_WORKERS", 10))) as pool:
        task_data = pool.starmap(
            get_transcription_task, tqdm([(s, cfg) for s in songinfos.values()])
        )
    task_data = [td for td in task_data if td is not None]
    for task, data in task_data:
        tasks.append(task)
        songinfos[task.uid].infer = data

    # Actually run transcription. Local import is used to prevent expensive imports
    # of espnet.bin.s2t_inference until required to reduce startup time.
    logger.info("Inference: constructing pool")
    logger.info(f"Setting ctranslate seed: {seed}")
    ctranslate2.set_random_seed(seed)
    from .asr_pool import AsrPool

    pool = AsrPool(
        num_gpus=None,
        model_type=cfg.model_type,
        model_name=cfg.model,
        batched=cfg.batched,
        cached=cached,
        infer_dir=infer_dir,
    )
    logger.info(f"Extra transcribe arguments: {cfg.transcribe_args}")
    results = pool.transcribe_batch(tasks)

    # Collect and write out results
    for uid, (result, task) in results.items():
        songinfo = songinfos[uid]
        assert songinfo.infer is not None
        songinfo.infer.whisper_result = result
        songinfo.infer.timings = task.timings
        util.write_json_dataclass(songinfo, infer_dir / (uid + ".json"))
        util.write_pz(infer_dir / (uid + ".pz"), songinfo)
    summary.infer = InferSummary(uids=list(results.keys()), info=dict(), cfg=cfg)
    util.write_json_dataclass(summary, infer_dir / "summary.json")


def run_infer_named(
    pipeline_name: str,
    cfg: InferConfig,
    preproc_name: str | None = None,
    force: bool = False,
    cached: bool = False,
    seed=42,
):
    """
    Run inference using a named pipeline.

    This function sets up the necessary directories for inference and preprocessing,
    and then calls the `run_infer` function with the provided configuration.

    Args:
        pipeline_name (str): The name of the pipeline to use for inference.
        cfg (InferConfig): The configuration object for inference.
        preproc_name (str | None, optional): The name of the preprocessing pipeline.
            If None, the `pipeline_name` will be used. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.

    Returns:
        None
    """
    infer_dir = Path("build") / pipeline_name / "infer"
    os.makedirs(infer_dir, exist_ok=True)

    preproc_name = preproc_name if preproc_name is not None else pipeline_name
    preproc_dir = Path("build") / preproc_name / "preproc"

    summary_file = infer_dir / "summary.json"
    if not force and os.path.exists(summary_file):
        logger.info(f"Summary file already exists at {summary_file}. Skipping inference.")
        return

    run_infer(
        cfg=cfg, infer_dir=infer_dir, preproc_dir=preproc_dir, cached=cached, seed=seed
    )


def run_infer_named_process(*args, **kwargs):
    """
    Run the `run_infer_named` function in a separate process using a ProcessPoolExecutor.

    This function submits the `run_infer_named` function to a process pool executor with the provided arguments
    and keyword arguments, and waits for the result. Any exceptions raised by `run_infer_named` will be rethrown.

    Args:
        *args: Variable length argument list to pass to `run_infer_named`.
        **kwargs: Arbitrary keyword arguments to pass to `run_infer_named`.

    Returns:
        None

    Raises:
        Any exception raised by `run_infer_named`.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f = executor.submit(run_infer_named, *args, **kwargs)
        f.result()  # rethrow any exceptions
