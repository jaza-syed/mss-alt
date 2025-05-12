# stdlib
from collections import OrderedDict
import socket
import logging
from multiprocessing import Pool

# first-party
from alt.extract import run_extract_named
from alt.preproc import run_preproc_named_process, run_preproc_named
from alt.infer import run_infer_named_process, run_infer_named, LEVEL_SONG
from alt.evaluate import run_eval_named
from alt.alt_types import (
    VadOptions,
    InferConfig,
    ExtractConfig,
    JamAltConfig,
    MusdbAltConfig,
    PreprocConfig,
)
from alt import util

langs = ["en", "es", "fr", "de"]

extract = ExtractConfig(
    # Test
    # jamendolyrics=JamendoLyricsConfig(languages=langs),
    jam_alt=JamAltConfig(
        languages=langs,
        revision="e149fa0f",
    ),
    # dali_gt=DaliConfig(languages=langs, path=Path("DALI_v1.0")),
    musdb_alt=MusdbAltConfig(),
)

vad_options = VadOptions(
    onset=0.1,
    offset=0.1,
    min_speech_duration_ms=0,
    max_speech_duration_s=30,
    min_silence_duration_ms=1000,
    speech_pad_ms=200,
)

base_infer = {
    "add_lang": True,
    "infer_level": LEVEL_SONG,
}
# models
models = OrderedDict(
    [
        (
            "whisper",
            {
                "model_type": "whisper",
                "model": "large-v2",
                "transcribe_args": {"condition_on_previous_text": False},
            },
        ),
        # (
        #     "owsm",
        #     {
        #         "model_type": "espnet-s2t",
        #         "model": "espnet/owsm_v3.1_ebf",
        #         "model_init_args": {"maxlenratio": 0.25},
        #         "transcribe_args": {"condition_on_prev_text": False},
        #     },
        # ),
    ]
)
# Segmentation algorithm
algos = OrderedDict(
    [("base", {}), ("rmsvad", {"vocal_rms_vad": True, "vad_options": vad_options})]
)
# Targets
targets = OrderedDict(
    [
        ("original", {"infer_target": "original"}),
        ("separated", {"infer_target": "separated_vocals"}),
        ("stem", {"infer_target": "stem", "datasets": ["musdb-alt"]}),
    ]
)

demucs_model_names = [
    "mdx_extra",
    # "mdx",
]

# FINAL: Running RMSVAD *5

# Inference
if __name__ == "__main__":
    util.default_log_config()
    names = set()
    extract_name = "04-extract"
    run_extract_named(extract_name, extract, force=False)
    for demucs_model in demucs_model_names:
        preproc_name = f"04-extract-{demucs_model}"
        preproc_cfg = PreprocConfig(separate=True, demucs_model_name=demucs_model)
        run_preproc_named_process(
            preproc_name,
            cfg=preproc_cfg,
            extract_name=extract_name,
            force=False,
            cached=True,
        )
        for model, m_args in models.items():
            for target, t_args in targets.items():
                for algo, a_args in algos.items():
                    infer_cfg = InferConfig(**(base_infer | m_args | t_args | a_args))  # type: ignore
                    # "base" does not use separated vox unless they are the target
                    if target != "separated" and algo == "base":
                        name = f"04-long-{model}-{target}-base"
                    else:
                        name = f"04-long-{demucs_model}-{model}-{target}-{algo}"
                    logging.info(f"Long-form experiments, inference: {name}")
                    for idx in range(5):
                        name_idx = f"{name}-{idx}"
                        run_infer_named_process(
                            name_idx,
                            infer_cfg,
                            preproc_name=preproc_name,
                            force=name_idx not in names,
                            seed=idx,
                        )
                        names.add(name_idx)
