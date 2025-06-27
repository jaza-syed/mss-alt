# import argparse
# from collections import OrderedDict
# from pathlib import Path
# import itertools
# import logging

# from alt.alt_types import (
#     VadOptions,
#     InferConfig,
# )
# from alt.extract import run_extract_named
# from alt.infer import run_infer_named
# from alt.util import default_log_config, extract_dir, infer_dir

# logger = logging.getLogger(__name__)


# def infer_short(num_iterations: int = 5, force: bool = True):
#     demucs_model_names = [
#         "mdx",
#         "mdx_extra",
#     ]
#     base_infer = {"add_lang": True}
#     models = OrderedDict(
#         [
#             ("whisper", {"model_type": "whisper", "model": "large-v2"}),
#         ]
#     )
#     audios = OrderedDict(
#         [
#             ("original", {"infer_target": "original"}),
#             ("separated", {"infer_target": "separated_vocals"}),
#             ("stem", {"infer_target": "stem", "datasets": ["musdb-alt"]}),
#         ]
#     )
#     eval_types = OrderedDict(
#         [
#             ("verse", {"infer_level": "grouped_line"}),
#             ("line", {"infer_level": "merged_line"}),
#         ]
#     )
#     names = set()
#     for demucs_model, (model, m_args), (audio, a_args), (
#         eval_type,
#         e_args,
#     ) in itertools.product(
#         demucs_model_names, models.items(), audios.items(), eval_types.items()
#     ):
#         preproc_name = f"01-preproc-{demucs_model}"
#         infer_cfg = InferConfig(**(base_infer | m_args | a_args | e_args))  # type: ignore
#         if audio != "separated":
#             name = f"01-short-{model}-{audio}-{eval_type}"
#         else:
#             name = f"01-short-{demucs_model}-{model}-{audio}-{eval_type}"
#         for idx in range(num_iterations):
#             name_idx = f"{name}-{idx}"
#             logger.info(f"Short-form inference: {name_idx}")
#             run_infer_named(
#                 name_idx,
#                 songs_path=extract_dir() / "01_data_all" / "output.gz"
#                 infer_cfg,
#                 preproc_name=preproc_name,
#                 force=name_idx not in names and force,
#                 seed=idx,
#             )
#             names.add(name_idx)


# def infer_long(num_iterations: int = 5, force: bool = True):
#     demucs_model_names = [
#         "mdx_extra",
#     ]
#     vad_options = VadOptions(
#         onset=0.1,
#         offset=0.1,
#         min_speech_duration_ms=0,
#         max_speech_duration_s=30,
#         min_silence_duration_ms=1000,
#         speech_pad_ms=200,
#     )
#     models = OrderedDict(
#         [
#             (
#                 "whisper",
#                 {
#                     "model_type": "whisper",
#                     "model": "large-v2",
#                     "transcribe_args": {"condition_on_previous_text": False},
#                 },
#             )
#         ]
#     )
#     # Segmentation algorithm
#     algos = OrderedDict(
#         [
#             ("rmsvad", {"vocal_rms_vad": True, "vad_options": vad_options}),
#             ("base", {}),
#         ]
#     )
#     # targets
#     audios = OrderedDict(
#         [
#             ("original", {"infer_target": "original"}),
#             ("separated", {"infer_target": "separated_vocals"}),
#             ("stem", {"infer_target": "stem", "datasets": ["musdb-alt"]}),
#         ]
#     )

#     base_infer = {
#         "add_lang": True,
#         "infer_level": "song",
#     }

#     names = set()
#     for demucs_model, (model, model_args), (audio, audio_args), (
#         algo,
#         algo_args,
#     ) in itertools.product(
#         demucs_model_names, models.items(), audios.items(), algos.items()
#     ):
#         infer_cfg = InferConfig(**(base_infer | model_args | audio_args | algo_args))  # type: ignore
#         if audio != "separated" and algo == "base":
#             name = f"01-long-{model}-{audio}-base"
#         else:
#             name = f"01-long-{demucs_model}-{model}-{audio}-{algo}"
#         preproc_name = f"01-preproc-{demucs_model}"
#         for idx in range(num_iterations):
#             name_idx = f"{name}-{idx}"
#             logger.info(f"Long-form inference: {name_idx}")
#             run_infer_named(
#                 name_idx,
#                 songs_path=util.ex
#                 preproc_name=preproc_name,
#                 force=name_idx not in names and force,
#                 seed=idx,
#             )
#             names.add(name_idx)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run audio processing pipeline.")
#     parser.add_argument(
#         "--cpu", action="store_true", help="Force CPU usage for processing."
#     )
#     parser.add_argument(
#         "--num_iterations",
#         type=int,
#         default=5,
#         help="Number of iterations for inference.",
#     )
#     args = parser.parse_args()
#     logger.info(
#         f"Starting audio processing pipeline. Running {args.num_iterations} iterations for inference."
#     )

#     default_log_config()
#     prepare(cpu=args.cpu, force=True)
#     infer_short(num_iterations=args.num_iterations, force=True)
#     infer_long(num_iterations=args.num_iterations, force=True)
