# stdlib
import logging
from collections import OrderedDict
from multiprocessing import Pool

# first party
from alt.infer import run_infer_named_process

from alt.alt_types import InferConfig
from alt import util
from alt.extract import ExtractConfig, JamAltConfig, MusdbAltConfig
from alt.preproc import PreprocConfig, run_preproc_named_process
from alt.extract import run_extract_named

base_infer = {"add_lang": True}

langs = ["en", "es", "fr", "de"]

models = OrderedDict(
    [
        ("whisper", {"model_type": "whisper", "model": "large-v2"}),
    ]
)

targets = OrderedDict(
    [
        ("original", {"infer_target": "original"}),
        ("separated", {"infer_target": "separated_vocals"}),
        ("stem", {"infer_target": "stem", "datasets": ["musdb-alt"]}),
    ]
)

levels = OrderedDict(
    [
        ("verse", {"infer_level": "verse"}),
        ("line", {"infer_level": "line"}),
    ]
)

demucs_model_names = [
    "mdx_extra",
    "mdx",
    # "htdemucs_ft",
]


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

# Inference
if __name__ == "__main__":
    util.default_log_config()
    names = set()
    extract_name = "04-extract"
    run_extract_named(extract_name, extract, force=False)
    # analysis is done a model/audio/level. "audio" is a compbination of demucs_model and target here
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
                for level, l_args in levels.items():
                    infer_cfg = InferConfig(**(base_infer | m_args | t_args | l_args))  # type: ignore
                    if target != "separated":
                        name = f"02-short-{model}-{target}-{level}"
                    else:
                        name = f"02-short-{demucs_model}-{model}-{target}-{level}"
                    logging.info(f"Short form experiments, inference: {name}")
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
