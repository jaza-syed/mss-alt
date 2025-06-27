# STDLIB
import logging
import os
from pathlib import Path
from collections import defaultdict
from typing import Tuple
from multiprocessing import Pool

# THIRDPARTY
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import datasets
from datasets import load_dataset, IterableDatasetDict
from librosa import get_duration
from DALI import Annotations

# FIRSTPARTY
from . import util
from .alt_types import (
    Song,
    TimedText,
    ExtractSummary,
)
from . import merge
from .metrics import tokens_as_words
from .tokenizer import LyricsTokenizer

logger = logging.getLogger(__name__)

DATASET_JAM_ALT = "jam-alt"
DATASET_MUSDB_ALT = "musdb-alt"

g_tokenizer = LyricsTokenizer()

#### JAM-ALT


def to_dict(text: TimedText) -> dict:
    return dict(start=text.start, end=text.end, text=text.text)


def merge_lines(lines: list[TimedText], *args, **kwargs) -> list[TimedText]:
    merged_lines = merge.merge_lines([to_dict(line) for line in lines], *args, **kwargs)
    return [TimedText(**line) for line in merged_lines]


def group_lines(lines: list[TimedText], *args, **kwargs) -> list[TimedText]:
    grouped_lines = merge.group_lines([to_dict(line) for line in lines], *args, **kwargs)
    return [TimedText(**line) for line in grouped_lines]


def use_tagged_line(line: dict):
    line["text"] = line.pop("text_tagged")
    return line


def extract_jam_alt(cfg: dict, dataset_id: str) -> list[Song]:
    """
    Extracts the jam-alt dataset and processes it according to the provided configuration.

    Args:
        cfg (dict): Configuration dictionary containing:
            - "revision": the dataset revision.
            - "languages": (optional) list of languages to include.

    Returns:
        dict[str, SongInfo]: Dictionary mapping unique song IDs to SongInfo objects.
    """
    logger.info("Extracting jam-alt dataset")
    logger.info("Loading jam-alt from huggingface datasets")
    dataset: IterableDatasetDict = load_dataset(
        "jamendolyrics/jam-alt", revision=cfg["revision"], trust_remote_code=True
    )["test"]  # type: ignore
    dataset.cast_column("audio", datasets.Audio(decode=False))
    logger.info("Loaded jam-alt")

    infos: list[Song] = []
    for row in tqdm(dataset):  # type: ignore
        if cfg.get("languages") is not None and row["language"] not in cfg["languages"]:
            continue
        song_id = row["name"]

        # Read short-form timings
        duration_secs = get_duration(path=row["audio"]["path"])
        lines = [
            TimedText(**use_tagged_line(line))
            for line in row["lines"]
            if line["end"] - line["start"] <= 30
        ]
        merged_lines = merge_lines(lines, audio_duration=duration_secs, song_id=song_id)
        grouped_lines = group_lines(merged_lines)

        # Construct songinfo
        infos.append(
            Song(
                uid=f"{song_id}_{dataset_id}",
                dataset_id=dataset_id,
                audio_path=row["audio"]["path"],
                song_id=song_id,
                duration_secs=duration_secs,
                language=row["language"],
                text=row["text_tagged"],
                merged_lines=merged_lines,
                grouped_lines=grouped_lines,
                split="test",
            )
        )
    return infos


### MUSDB-ALT
def extract_musdb_alt(cfg: dict, dataset_id: str) -> list[Song]:
    """
    Extracts the musdb-alt dataset and processes it according to the provided configuration.
    Args:
        cfg (dict): Configuration dictionary containing:
            - "revision": the dataset revision.
            - "path": the path to the musdb audio
            - "languages": (optional) list of languages to include.

    Returns:
        dict[str, SongInfo]: Dictionary mapping unique song IDs to SongInfo objects.
    """
    logger.info("Extracting MUSDB-ALT")
    musdb_dir: Path = util.data_root() / cfg["audio_path"]
    dataset: IterableDatasetDict = load_dataset(
        "jazasyed/musdb-alt",
        revision=cfg["revision"],
        trust_remote_code=True,
        tagged=True,
    )  # type: ignore
    infos: list[Song] = []
    for row in tqdm(dataset["test"]):
        song_id = row["name"].replace(" ", "_")
        # Get audio path
        audio_dir: Path = musdb_dir / "test" / row["name"]
        audio_path = audio_dir / "mixture.wav"
        duration_secs = get_duration(path=audio_path)

        # Read short-form timings
        lines = [line for line in row["lines"] if line["end"] - line["start"] <= 30]
        for line in lines:
            if not line["text_tagged"].endswith("\n"):
                line["text_tagged"] += "\n"
        lines = [TimedText(**use_tagged_line(line)) for line in lines]
        merged_lines = merge_lines(lines, audio_duration=duration_secs, song_id=song_id)
        grouped_lines = group_lines(merged_lines)

        # Construct songinfo
        infos.append(
            Song(
                uid=f"{song_id}_{dataset_id}",
                dataset_id=dataset_id,
                audio_path=audio_path,
                duration_secs=duration_secs,
                song_id=song_id,
                language="en",
                text=row["text_tagged"],
                merged_lines=merged_lines,
                grouped_lines=grouped_lines,
                split="test",
                stem_dir=audio_dir,
            )
        )
    return infos


dali_language_codes = {
    "english": "en",
    "french": "fr",
    "german": "de",
    "spanish": "es",
}


def has_words(text: str, language: str) -> bool:
    return len(tokens_as_words(g_tokenizer(text, language))) != 0


### DALI
def dali_data_to_songinfo(
    dali_dir: Path, dali_id: str, dataset_id: str, durations: dict[str, float]
) -> Song | None:
    """
    Converts DALI dataset annotations to a Song object.

    Args:
        data (Annotations): The annotations from the DALI dataset.
        audio_fn (Path): The file path to the audio file.
        dataset_id (str): The identifier for the dataset.
        durations (dict): Dictionary mapping dali_id to duration_secs.

    Returns:
        Song: The converted song information including lyrics and timings.
    """
    audio_path = dali_dir / "audio" / (dali_id + ".mp3")
    annot_path = dali_dir / "annot_tismir" / (dali_id + ".gz")
    data: Annotations = util.read_pz(annot_path)
    dali_id = data.info["id"]
    dali_language = data.info["metadata"]["language"]
    if dali_language not in dali_language_codes:
        return None
    language = dali_language_codes[dali_language]

    # assemble paragraphs
    paragraphs = defaultdict(list[str])
    for line in data.annotations["annot"]["lines"]:
        paragraphs[line["index"]].append(line["text"])
    paragraphs = sorted(list(paragraphs.items()), key=lambda pg: pg[0])
    lyrics = "\n\n".join(["\n".join(lines[1]) for lines in paragraphs])

    # timings
    line_timings = [
        TimedText(
            start=line["time"][0],
            end=line["time"][1],
            text=line["text"].strip() + "\n",
            extra={"verse_index": line["index"]},
        )
        for line in data.annotations["annot"]["lines"]
        if has_words(line["text"].strip(), language)
        and line["time"][1] - line["time"][0] < 30
    ]
    grouped_lines = group_lines(line_timings)

    # correct languages
    if dali_id == "3f95c86b880d483d905f20ec35e336fa":
        language = "de"
    if dali_id == "91885c30b8bb4847ac220e4ed36e96c3":
        language = "de"
    if dali_id == "efbc7553da754f28ba888c089216a731":
        language = "de"
    if dali_id == "f4f69b23ab9f4add9274fd712c75ff4":
        language = "fr"

    # duration_secs: must use durations dict
    if dali_id not in durations:
        raise ValueError(f"Duration not found for DALI id: {dali_id}")
    duration_secs = durations[dali_id]
    song = Song(
        dataset_id=dataset_id,
        uid=f"{dali_id}_{dataset_id}",
        song_id=dali_id,
        language=language,
        audio_path=audio_path,
        duration_secs=duration_secs,
        split="train",
        merged_lines=line_timings,
        grouped_lines=grouped_lines,
        text=lyrics,
    )
    return song


def extract_dali_v2(cfg: dict, dataset_id: str, num_workers: int = 10) -> list[Song]:
    """
    Extracts DALI V2 dataset annotations and audio information, filters based
    on exclusions and languages, and writes the extracted data to JSON files.

    Args:
        cfg (dict): Dictionary with keys:
            - "path": Path to the DALI dataset directory.
            - "languages": List of languages to include.
    Returns:
        list[Song]: A list of extracted Song objects.
    """
    # Set up
    dali_dir: Path = cfg["path"]
    data_root = util.data_root()
    counts = {"missing_audio": 0, "excluded": 0}
    exclusions = util.model_read_json(
        util.project_root() / "data" / "dali_v2_exclude_ids.json", dict[str, list[str]]
    )
    exclusions = {uid for sublist in exclusions.values() for uid in sublist}
    # Load durations (must exist)
    durations = util.model_read_json(
        util.project_root() / "data/dali_v2_durations.json", dict[str, float]
    )

    logger.info(f"Reading DALI V2 data from {dali_dir}/*.gz")
    # Filter
    valid_dali_ids: list[str] = []
    for gzfn in list((data_root / dali_dir).glob("annot_tismir/*.gz")):
        # Check for audio / exclusions
        dali_id = os.path.basename(gzfn)[:-3]
        if dali_id in exclusions:
            counts["excluded"] += 1
            continue
        audio_fn = dali_dir / "audio" / (dali_id + ".mp3")
        if not os.path.exists(audio_fn):
            logger.debug(f"Audio not found: {dali_id}")
            counts["missing_audio"] += 1
            continue
        valid_dali_ids.append(dali_id)

    # Process
    pool = Pool(num_workers)
    infos = pool.starmap(
        dali_data_to_songinfo,
        tqdm(
            [(dali_dir, dali_id, dataset_id, durations) for dali_id in valid_dali_ids],
            total=len(valid_dali_ids),
        ),
    )
    infos = list(info for info in infos if info is not None)

    # filter out
    languages = cfg.get("languages", ["en", "es", "de", "fr"])
    infos = [info for info in infos if info is not None and info.language in languages]

    # print summary
    total = counts["excluded"] + counts["missing_audio"] + len(infos)
    logger.info(f"Missing audio: {counts['missing_audio']}/{total}")
    logger.info(f"Excluded: {counts['excluded']}/{total}")
    return infos


def run_extract(
    cfg: dict[str, dict], extract_dir: Path, num_workers: int = 10
) -> Tuple[Path, Path]:
    """
    Run the extraction process based on the provided configuration.

    Args:
        cfg (ExtractConfig): Configuration for the extraction process.
        extract_dir (Path): Directory where the extracted data will be stored.

    Returns:
        None
    """
    logger.info(f"Running extract: {extract_dir}")
    infos: list[Song] = []
    for dataset_id, dataset_cfg in cfg.items():
        if dataset_cfg["dataset_type"] == "jam-alt":
            infos.extend(extract_jam_alt(cfg=dataset_cfg, dataset_id=dataset_id))
        if dataset_cfg["dataset_type"] == "musdb-alt":
            infos.extend(extract_musdb_alt(cfg=dataset_cfg, dataset_id=dataset_id))
        if dataset_cfg["dataset_type"] == "dali_v2":
            infos.extend(
                extract_dali_v2(
                    cfg=dataset_cfg, dataset_id=dataset_id, num_workers=num_workers
                )
            )

    # Write main output
    output_path = extract_dir / "output.gz"
    with util.DataFile(output_path, Song, mode="w") as f:
        for songinfo in infos:
            f.write(songinfo)

    # Write summary
    summary_fn = extract_dir / "summary.json"
    summary = ExtractSummary(uids=[info.uid for info in infos])
    util.model_dump_json(summary_fn, summary)

    return output_path, summary_fn


def run_extract_named(
    name: str,
    cfg: dict[str, dict],
    force: bool = False,
    num_workers: int = 10,
) -> Tuple[Path, Path]:
    """
    Run the extraction process for the given pipeline name and configuration.

    Args:
        pipeline_name (str): The name of the pipeline.
        cfg (ExtractConfig): The configuration for extraction.
        force (bool): If False and the summary file already exists, skip processing.
                      Default is False.
    """
    extract_dir = util.extract_dir() / name
    os.makedirs(extract_dir, exist_ok=True)

    summary_file = extract_dir / "summary.json"
    if not force and os.path.exists(summary_file):
        logger.info(
            f"Summary file already exists at {summary_file}. Skipping extraction."
        )
        return (extract_dir / "output.gz", extract_dir / "summary.json")

    with logging_redirect_tqdm():
        return run_extract(cfg=cfg, extract_dir=extract_dir, num_workers=num_workers)
