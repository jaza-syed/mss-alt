# STDLIB
import argparse
import itertools
import logging
from dataclasses import replace

# FIRSTPARTY
from alt.alt_types import (
    VadOptions,
    InferConfig,
    ModelType,
    LongFormAlgo,
    InferLevel,
)
from alt.infer import run_infer_named_process
from alt.util import extract_dir, default_log_config, infer_dir

logger = logging.getLogger(__name__)


def infer_short(num_iterations: int = 5, force: bool = True, debug: bool = False):
    base_config = InferConfig(
        model_type=ModelType.WHISPER,
        model_name="large-v2",
        batched=True,
        add_lang=True,
        transcribe_args={"beam_size": 5, "best_of": 5},
    )
    audios: list[str] = [
        "vocal_stem",
        "original",
        "mdx",
        "mdx_extra",
    ]
    eval_types: list[InferLevel] = [
        InferLevel.MERGED_LINE,
        InferLevel.GROUPED_LINE,
    ]
    songs_path = extract_dir() / "01-extract" / "output.gz"
    for audio, eval_type in itertools.product(audios, eval_types):
        for idx in range(num_iterations):
            config = replace(base_config)
            config.level = eval_type
            config.audio = audio
            if audio == "vocal_stem":
                config.datasets = ["musdb-alt"]
            name = f"01-ss-short-{audio}-{eval_type}-{idx}"
            logger.info(f"Short-form inference: {name}")
            # use _process version to ensure effect of random seed
            run_infer_named_process(
                name,
                songs_path=songs_path,
                cfg=config,
                force=force,
                random_seed=idx,
                debug=debug,
            )


def infer_long(num_iterations: int = 5, force: bool = True, debug: bool = False):
    vad_options = VadOptions(
        onset=0.1,
        offset=0.1,
        min_speech_duration_ms=0,
        max_speech_duration_s=30,
        min_silence_duration_ms=1000,
        speech_pad_ms=200,
    )
    base_config = InferConfig(
        level=InferLevel.SONG,
        model_type=ModelType.WHISPER,
        model_name="large-v2",
        add_lang=True,
        transcribe_args={
            "beam_size": 5,
            "best_of": 5,
            "condition_on_previous_text": False,
        },
        # Vad options only used if algorithm set to RMSVAD, so we can set them here
        vad_options=vad_options,
        vad_audio="mdx_extra",
        batched=False,
    )
    # Segmentation algorithm
    algos = [
        LongFormAlgo.NATIVE,
        LongFormAlgo.RMSVAD,
    ]
    # targets
    audios = [
        "original",
        "mdx_extra",
        "vocal_stem",
    ]

    songs_path = extract_dir() / "01-extract" / "output.gz"
    for algo, audio in itertools.product(algos, audios):
        config = replace(base_config)
        config.audio = audio
        config.algo = algo
        if audio == "vocal_stem":
            config.datasets = ["musdb-alt"]
        if algo == LongFormAlgo.RMSVAD:
            config.batched = True
        for idx in range(num_iterations):
            name = f"01-ss-long-{audio}-{algo}-{idx}"
            logger.info(f"Long-form inference: {name}")
            # use _process version to ensure effect of random seed
            run_infer_named_process(
                name,
                songs_path=songs_path,
                cfg=config,
                random_seed=idx,
                force=force,
                debug=debug,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run audio processing pipeline.")
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage for processing."
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of iterations for inference.",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Use cached results if available (sets force to False).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug mode",
    )
    args = parser.parse_args()
    logger.info(
        f"Starting audio processing pipeline. Running {args.num_iterations} iterations for inference."
    )

    default_log_config()
    infer_dir().mkdir(exist_ok=True)
    infer_short(
        num_iterations=args.num_iterations, force=not args.cached, debug=args.debug
    )
    infer_long(
        num_iterations=args.num_iterations, force=not args.cached, debug=args.debug
    )
