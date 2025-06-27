## Extract song information and separate vocals
from alt.extract import run_extract_named
from alt.separate import run_separate
from alt.util import separation_dir, default_log_config, read_datafile, extract_dir
from alt.alt_types import Song, SeparateConfig

datasets = {
    "jam-alt": {
        "dataset_type": "jam-alt",
        "revision": "0e15962",
        "languages": ["en", "fr", "de", "es"],
    },
    "musdb-alt": {
        "dataset_type": "musdb-alt",
        "revision": "v1.1.0",
        "audio_path": "musdb18hq",
    },
}

demucs_model_names = [
    "mdx",
    "mdx_extra",
]

if __name__ == "__main__":
    default_log_config()
    # songs_path is extract_dir() / "01-extract" / "output.gz"
    extract_dir().mkdir(exist_ok=True)
    songs_path, _summary_fn = run_extract_named("01-extract", cfg=datasets, force=False)
    songs = read_datafile(songs_path, Song)
    for demucs_model in demucs_model_names:
        config = SeparateConfig(model_name=demucs_model, model_type="demucs", cached=True)
        output_path = separation_dir() / demucs_model
        output_path.mkdir(exist_ok=True)
        run_separate(songs, output_path, config=config)
