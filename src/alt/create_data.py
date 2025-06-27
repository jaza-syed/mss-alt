# STDLIB
import logging
import unicodedata
from multiprocessing import Pool
from typing import Iterable
import re

# THIRDPARTY
from pydantic.dataclasses import dataclass

# FIRSTPARTY
from .alt_types import Song
from . import util

logger = logging.getLogger(__name__)


@dataclass
class Record:
    """
    A single training instance for Whisper.
    `text` can include timestamps in the format of <|0.00|>.
    """

    audio_path: str
    text: str  # text including timestamps -- grouped lines could have timestamps!
    start: float  # timestamp in seconds
    end: float  # timestamp in seconds
    dataset: str
    language: str = "en"


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"</?nl>", "", text)  # Remove <nl> and </nl>
    text = re.sub(r"\([^)]*\)", "", text)  # Remove text inside parentheses and the parens
    text = re.sub(r"\s+,", ",", text)  # Remove space before commas
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
    return text.strip()  # Trim leading/trailing spaces


def to_records(song: Song, separated: str | None = None) -> list[Record]:
    """
    This function converts a song to a Record for finetuning with whisper
    """
    records = []
    for line in song.grouped_lines:
        if separated:
            audio_path = util.separation_dir() / separated / song.uid / "vocals.mp3"
        else:
            audio_path = song.audio_path
        assert audio_path.is_file()

        # Unlikely to be necessary, but just in case
        text = clean_text(line.text)
        if not text:
            continue

        record = Record(
            audio_path=str(audio_path),
            text=text,
            start=line.start,
            end=line.end,
            language=song.language,
            dataset=song.dataset_id,
        )
        if line.end > line.start:
            records.append(record)
    return records


def songs_to_records(
    songs: list[Song], separated: str | None = None, num_workers: int = 1
) -> list[Record]:
    song_records: Iterable[Iterable[Record]]
    exclude = set(
        util.model_read_json(
            util.project_root() / "data" / "dali_v2_torchaudio_noncompat.json", list[str]
        )
    )
    if num_workers > 1:
        with Pool(num_workers) as pool:
            song_records = pool.starmap(
                to_records,
                ((song, separated) for song in songs if song.song_id not in exclude),
            )
    else:
        song_records = (
            to_records(song, separated) for song in songs if song.song_id not in exclude
        )
    return [rec for records in song_records for rec in records]
