# stdlib
import logging
import os
from pathlib import Path

# third-party
from tqdm import tqdm
from datasets import load_dataset
import soundfile as sf

# first-party
from . import util
from .merge import group_lines, merge_lines
from .tokenizer import LyricsTokenizer
from .alt_types import (
    ExtractConfig,
    SongData,
    Timing,
    SongInfo,
    JamAltConfig,
    MusdbAltConfig,
    Summary,
    ExtractSummary,
)

logger = logging.getLogger(__name__)

DATASET_JAM_ALT = "jam-alt"
DATASET_MUSDB_ALT = "musdb-alt"

g_tokenizer = LyricsTokenizer()


def use_tagged_line(line: dict):
    line["text"] = line.pop("text_tagged")
    return line


def extract_jam_alt(cfg: JamAltConfig) -> dict[str, SongInfo]:
    """
    Extracts the jam-alt dataset and processes it according to the provided configuration.

    Args:
        extract_dir (Path): The directory where the extracted data will be saved.
        cfg (JamAltConfig): Configuration object containing parameters for extraction.

    Returns:
        list[str]: A list of unique identifiers for the extracted songs.

    """
    logger.info("Extracting jam-alt dataset")
    logger.info("Loading jam-alt from huggingface datasets")
    dataset = load_dataset(
        "audioshake/jam-alt", revision=cfg.revision, trust_remote_code=True
    )["test"]  # type: ignore
    logger.info("Loaded jam-alt")
    # Iterate over dataset
    row: dict  # not actually true, but gets us rid of the type errors?
    infos = {}
    for row in tqdm(dataset):  # type: ignore
        if cfg.languages is not None and row["language"] not in cfg.languages:
            continue
        song_id = row["name"]

        # Read short-form timings
        audio_duration = row["audio"]["array"].shape[0] / row["audio"]["sampling_rate"]
        lines = [
            Timing(**use_tagged_line(line))
            for line in row["lines"]
            if line["end"] - line["start"] <= 30
        ]
        merged_lines = merge_lines(lines, audio_duration=audio_duration, song_id=song_id)
        grouped_lines = group_lines(merged_lines)

        # Construct songinfo
        songinfo = SongInfo(
            extract=SongData(
                uid=f"{song_id}_{DATASET_JAM_ALT}",
                dataset_id=DATASET_JAM_ALT,
                audio_fn=row["audio"]["path"],
                song_id=song_id,
                duration=audio_duration,
                language=row["language"],
                text=row["text_tagged"],
                merged_lines=[Timing(**line) for line in merged_lines],
                grouped_lines=[Timing(**line) for line in grouped_lines],
                split="test",
            )
        )
        infos[songinfo.extract.uid] = songinfo

    return infos


def extract_musdb_alt(cfg: MusdbAltConfig) -> dict[str, SongInfo]:
    logger.info("Extracting MUSDB-ALT")
    musdb_dir = util.data_root() / cfg.musdb_dir
    dataset = load_dataset(
        "jazasyed/musdb-alt", revision=cfg.revision, trust_remote_code=True, tagged=True
    )  # type: ignore
    infos = {}
    for split in ["test"]:
        for row in tqdm(dataset[split]):
            song_id = row["name"].replace(" ", "_")
            # Get audio path
            audio_dir = musdb_dir / split / row["name"]
            audio_fn = audio_dir / "mixture.wav"
            audio, sr = sf.read(audio_fn, always_2d=True)

            # Read short-form timings
            lines = [line for line in row["lines"] if line["end"] - line["start"] <= 30]
            for line in lines:
                if not line["text"].endswith("\n"):
                    line["text"] += "\n"
            merged_lines = merge_lines(
                lines, audio_duration=len(audio) / sr, song_id=song_id
            )
            grouped_lines = group_lines(merged_lines)

            # Construct songinfo
            songinfo = SongInfo(
                extract=SongData(
                    uid=f"{song_id}_{DATASET_MUSDB_ALT}",
                    dataset_id=DATASET_MUSDB_ALT,
                    audio_fn=audio_dir / "mixture.wav",
                    duration=len(audio) / sr,
                    song_id=song_id,
                    language="en",
                    text=row["text"],
                    merged_lines=[Timing(**line) for line in merged_lines],
                    grouped_lines=[Timing(**line) for line in grouped_lines],
                    split="test",
                    extra=dict(stem_dir=str(audio_dir)),
                )
            )
            infos[songinfo.extract.uid] = songinfo
    return infos


def run_extract(cfg: ExtractConfig, extract_dir: Path):
    """
    Run the extraction process based on the provided configuration.

    Args:
        cfg (ExtractConfig): Configuration for the extraction process.
        extract_dir (Path): Directory where the extracted data will be stored.

    Returns:
        None
    """
    logger.info(f"Running extract: {extract_dir}")
    infos = {}
    if cfg.jam_alt is not None:
        infos |= extract_jam_alt(cfg=cfg.jam_alt)
    if cfg.musdb_alt is not None:
        infos |= extract_musdb_alt(
            cfg=cfg.musdb_alt,
        )

    # Write results
    for uid, songinfo in infos.items():
        util.write_json_dataclass(songinfo, extract_dir / (uid + ".json"))
        util.write_pz(extract_dir / (uid + ".pz"), songinfo)
    summary = Summary(extract=ExtractSummary(uids=infos.keys(), cfg=cfg, info={}))
    util.write_json_dataclass(summary, extract_dir / "summary.json")


def run_extract_named(pipeline_name: str, cfg: ExtractConfig, force: bool = False):
    """
    Run the extraction process for the given pipeline name and configuration.

    Args:
        pipeline_name (str): The name of the pipeline.
        cfg (ExtractConfig): The configuration for extraction.
        force (bool): If False and the summary file already exists, skip processing.
                      Default is False.
    """
    extract_dir = Path("build") / pipeline_name / "extract"
    os.makedirs(extract_dir, exist_ok=True)

    summary_file = extract_dir / "summary.json"
    if not force and os.path.exists(summary_file):
        logger.info(
            f"Summary file already exists at {summary_file}. Skipping extraction."
        )
        return

    run_extract(cfg=cfg, extract_dir=extract_dir)
