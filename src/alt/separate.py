# stdlib
import concurrent.futures
import copy
import logging
import os
from pathlib import Path

# third-party
from tqdm import tqdm

# first-party
from . import util
from .alt_types import (
    SeparationTask,
    SeparateConfig,
    SeparateSummary,
    Song,
)
from .demucs_pool import DemucsPool

logger = logging.getLogger(__name__)


def run_separate(
    songs: list[Song],  # list of songs (uid + audio fn are the most important)
    output_path: Path,  # subdirectory to output stems
    config: SeparateConfig,
) -> Path:
    """
    Separate songs
    Result: in format separation_dir()/{output_path}/{uid}/{stem_name}.mp3
    """
    logger.info(f"Running preprocessing: {output_path}")
    if config.cached:
        logger.info(
            "Not separating any songs with existing completion files (cached=True)"
        )

    tasks: list[SeparationTask] = []
    for song in songs:
        # actually run preprocessing + store metadata
        tasks.append(
            SeparationTask(
                uid=song.uid,
                audio_path=song.audio_path,
            )
        )

    # run separation in batch and update songinfos with results
    if config.model_type == "demucs":
        apply_args = copy.deepcopy(config.apply_args)
        segment = apply_args.pop("segment", None)
        pool = DemucsPool(
            output_dir=util.separation_dir() / output_path,
            model_name=config.model_name,
            overwrite=not config.cached,
            segment=segment,
            stems=config.stems,
            apply_args=apply_args,
        )
        pool.separate_batch(tasks)

    # write summary
    logger.info("Completed separation")
    summary = SeparateSummary(
        tasks=tasks,
        config=config,
        path=output_path,
    )
    summary_path = util.separation_dir() / output_path / "summary.json"
    util.model_dump_json(summary_path, summary)
    return summary_path
