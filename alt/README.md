# Source Separation for Lyrics Transcription
This is the repo assosciated with the [paper](https://arxiv.org/TODO) "Exploiting Music Source Separation for 
Automatic Lyrics Transcription with Whisper" published at the ICME 2025 workshop "Artificial Intelligence for Music"
(https://ai4musicians.org/2025icme.html)

## Setup
This repo uses [pixi](https://pixi.sh/latest/) for environment and python
dependency management. The environment is declared in `pyproject.toml`.
To create the environment and install dependencies for development
``` sh
export PIXI_FROZEN=true
pixi install
pixi shell  # shell to run pixi in
pre-commit install  # install pre-commit hooks
```
We recommend having `PIXI_FROZEN=true` set by default so that the environment
/ lock file is not updated every time you start a shell. You can unset it if you do want
to update dependencies.

## Datasets
The main datasets used in this repo are available on huggingface:
- `jam-alt` - https://huggingface.co/datasets/audioshake/jam-alt - tag `v1.4.0`
- `musdb-alt` - https://huggingface.co/datasets/audioshake/musdb-alt - tag `v1.0.0`
There must be a copy of MUSDB available. By default, this is expected to be in
the folder specified by the environment variable `DATA_ROOT`, in a subdirectory `musdb18hq`,
with the specified folder structure:
- `musdb18hq` - MUSDB18-HQ https://zenodo.org/records/3338373
  - `test/<song name>/<stem name>.wav`
  - `train/<song name>/<stem name>.wav`

## Running experiments
Before running experimental scripts, use the environment variables
  `CUDA_VISIBLE_DEVICES` to set the available GPUs and `NUM_WORKERS` to set the 
  number of CPU threads desired.

To run the experiments from our paper, run the following command in the project root directory
```
pixi run python expt/01-ss/ss.py
```
Then the run the notebooks `long.pynb` and `short.ipynb` in the directory `expt/01-ss` .

## Structure
The repo contains a python module `alt` in `src/alt` which provides functions
used to run ALT evaluation pipelines. The pipelines are divided into four stages
- `extract` : Copy per-song metadata into a standardised format across different datasets
- `preproc` : Run any preprocessing e.g. source separation, voice activity detection
- `infer` : Run lyrics transcription algorithm
- `evaluate` : Evaluate results from infer stage against the ground truth
The repo contains additional supporting modules:
- `alt_types` : dataclasses used throughout the module
- `asr_pool` : pool class for running ASR on multiple GPUs
- `demucs_pool` : pool class for running MSS on multiple GPUs
- `render` : rendering evaluation results to HTML
- `util` : filesystem and environment utilities
- `merge` : functions to produce merged lines and groupes from line-level timings
The following modules are adapted from the python package [alt-eval](https://github.com/audioshake/alt-eval)
- `tokenizer`: lyric tokenizer
- `metrics.py`: utilities to compute error metrics
- `normalization.py`: lyric normalizer

The folder `expt/` contains experimental code and analysis notebooks.
The folder `rsc/` contains some audio files for general use.
The folder `test/` contains tests for `alt`.