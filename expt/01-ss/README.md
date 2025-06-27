# Source separation experiments

This folder contains scripts and an Ipython notebook to reproduce the results from the paper
[Exploiting Music Source Separation for Automatic Lyrics Transcription with Whisper](https://arxiv.org/abs/2506.15514).

The environment variables `DATA_ROOT` and `OUTPUT_DIR` must be set as described in the top-level [README](../../README.md). Use 
`CUDA_VISIBLE_DEVICES` to control the GPUs used for separation and inference.
To run the experiments run the following commands in this directory:
``` sh
./download_musdb.sh  # download musdb to $DATA_ROOT/musdb18hq
pixi run -e cuda python ss_prepare.py  # preprocess data and separate
pixi run -e cuda python ss_infer.py --num-iterations=5  # run inference
```
then use `ss_plots.ipynb` to obtain tables and figures. `ss_plots.ipynb` saves figures in `$OUTPUT_DIR/share/01-ss`. 
With a Xeon Gold 5318Y CPU and 2 A5000 GPUs, the experiments take us roughly 6 hours to run for `--num_iterations=5`.
We use roughly 6GB VRAM on each GPU.