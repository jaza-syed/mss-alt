# stdlib
import concurrent.futures
import logging
import os
from pathlib import Path

# third-party
from tqdm import tqdm

# first-party
from . import util
from .alt_types import PreprocConfig, PreprocData, SongInfo, PreprocSummary
from .demucs_pool import DemucsPool, SeparationTask

logger = logging.getLogger(__name__)


def run_preproc(
    extract_dir: Path,
    preproc_dir: Path,
    cfg: PreprocConfig,
    cached: bool = True,
    cpu=False,
    debug=False,
):
    """
    Run the preprocessing pipeline for audio data.

    Args:
        extract_dir (Path): Directory containing the extracted audio and metadata files.
        preproc_dir (Path): Directory where the preprocessed data will be stored.
        cfg (PreprocConfig): Configuration object for preprocessing settings.
        cached (bool, optional): If True, overwrite existing separated audio. Defaults to True.

    Returns:
        None
    """
    logger.info(f"Running preprocessing: {preproc_dir}")
    if cached:
        logger.info("Using existing separated audio (cached=True)")
    summary = util.read_summary(extract_dir / "summary.json")
    tasks: list[SeparationTask] = []
    songinfos: dict[str, SongInfo] = dict()
    # Construct separation tasks and make id->songinfo mapping
    # Note that a task is still submitted even if separation is complete, so
    # other code in this file still runs! This means we can rerun preproc.py
    # if extract directory updated and pick up info without having to rerun
    # separation.
    extract_uids = summary.extract.uids
    if debug:
        extract_uids = extract_uids[::20]
    for uid in tqdm(extract_uids):
        songinfo: SongInfo = util.read_pz(extract_dir / (uid + ".pz"))
        songinfos[songinfo.extract.uid] = songinfo
        songinfo.preproc = None
        # actually run preprocessing + store metadata
        if cfg.separate:
            tasks.append(
                SeparationTask(
                    uid=songinfo.extract.uid,
                    audio_path=songinfo.extract.audio_fn,
                    output_dir=os.path.join(preproc_dir, songinfo.extract.uid),
                    apply_kwargs=cfg.demucs_args,
                )
            )

    # run separation in batch and update songinfos with results
    if cfg.separate:
        pool = DemucsPool(cpu=cpu, model_name=cfg.demucs_model_name, overwrite=not cached)
        results = pool.separate_batch(tasks)
        for uid, result in results.items():
            songinfos[uid].preproc = PreprocData(
                vocals_fn=result["vocals"], other_fn=result.get("other", Path())
            )
        pool.cleanup()

    # write out preproc fns
    uids = []
    for uid, songinfo in songinfos.items():
        util.write_pz(preproc_dir / (uid + ".pz"), songinfo)
        util.write_json_dataclass(songinfo, preproc_dir / (uid + ".json"))
        uids.append(uid)

    # write summary
    summary.preproc = PreprocSummary(uids=uids, info=dict(), cfg=cfg)
    util.write_json_dataclass(summary, preproc_dir / "summary.json")
    logger.info("Completed preprocessing")


def run_preproc_named(
    name: str, cfg: PreprocConfig, extract_name: str | None = None, force=False, **kwargs
):
    """
    Run preprocessing using a named pipeline.

    This function sets up the necessary directories for preprocessing,
    and then calls the `run_preproc` function with the provided configuration.

    Args:
        pipeline_name (str): The name of the pipeline to use for preprocessing.
        cfg (PreprocConfig): The configuration object for preprocessing.
        extract_name (str | None, optional): The name of the extraction pipeline.
            If None, the `pipeline_name` will be used. Defaults to None.
        force (bool, optional): Whether to skip if existing summaryexists.
        Cached (bool, optional): wwhether to use cached audio
    """
    preproc_dir = Path("build", name, "preproc")
    os.makedirs(preproc_dir, exist_ok=True)

    extract_name = extract_name if extract_name is not None else name
    extract_dir = Path("build") / extract_name / "extract"

    summary_file = preproc_dir / "summary.json"
    if not force and os.path.exists(summary_file):
        logger.info(
            f"Summary file already exists at {summary_file}. Skipping preprocessing."
        )
        return

    run_preproc(cfg=cfg, preproc_dir=preproc_dir, extract_dir=extract_dir, **kwargs)


def run_preproc_named_process(*args, **kwargs):
    """
    Runs the `run_preproc_named` function in a separate process using a ProcessPoolExecutor.

    Args:
        *args: Variable length argument list to be passed to `run_preproc_named`.
        **kwargs: Arbitrary keyword arguments to be passed to `run_preproc_named`.

    Raises:
        Any exception raised by `run_preproc_named` will be rethrown.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f = executor.submit(run_preproc_named, *args, **kwargs)
        f.result()  # rethrow any exceptions
