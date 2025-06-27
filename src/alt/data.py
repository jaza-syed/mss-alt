# STDLIB
from typing import Any
from pathlib import Path
from dataclasses import asdict
import logging

# THIRDPARTY
from pydantic import Field
from pydantic.dataclasses import dataclass
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchaudio import AudioMetaData
from whisper.tokenizer import Tokenizer, get_tokenizer
from whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim

# FIRSTPARTY
from . import util
from .alt_types import Song
from .create_data import Record, songs_to_records


logger = logging.getLogger(__name__)

TrainingDataType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class AudioDataset(Dataset):
    def __init__(
        self,
        records: list[Record],
        tokenizer: Tokenizer,
        sample_rate: int = 16000,
        n_mels: int = 80,
    ):
        self.records: list[Record] = records
        self.tokenizer: Tokenizer = tokenizer
        self.sample_rate: int = sample_rate
        self.n_mels: int = n_mels
        self.audio_info: dict[str, AudioMetaData] = dict()

    def __len__(self) -> int:
        return len(self.records)

    def get_decoder_io(self, text: str, language: str) -> tuple[list[int], list[int]]:
        text_tokens = self.tokenizer.encode(text=text)
        if not text.strip():
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.no_speech,
                self.tokenizer.no_timestamps,
            ]
        else:
            special_tokens = [
                self.tokenizer.sot,
                self.tokenizer.special_tokens[f"<|{language}|>"],
                self.tokenizer.transcribe,
                self.tokenizer.no_timestamps,
            ]
        decoder_output = special_tokens[1:] + text_tokens + [self.tokenizer.eot]
        decoder_input = special_tokens + text_tokens
        return decoder_input, decoder_output

    def _calculate_mel(self, audio_path: str, start: float, end: float) -> torch.Tensor:
        # Get audio info
        if audio_path not in self.audio_info:
            self.audio_info[audio_path] = torchaudio.info(audio_path)
        info: AudioMetaData = self.audio_info[audio_path]

        # Calculate frame offsets
        original_sr = info.sample_rate
        start_frame = int(start * original_sr)
        num_frames = int((end - start) * original_sr)
        # Load only the required segment and resample if needed
        # load also has to call info to process, but caching is still useful
        # to avoid calling it twice to get the frame offsets before calling load
        waveform, sr = torchaudio.load(
            audio_path, frame_offset=start_frame, num_frames=num_frames
        )
        if waveform.shape[0] > 1:  # mono
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.sample_rate and sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform: torch.Tensor = resampler(waveform)

        mel = log_mel_spectrogram(waveform, self.n_mels)
        mel = pad_or_trim(mel, N_FRAMES)
        # mel = mel.half()
        return mel.squeeze(0)  # type: ignore

    def __getitem__(self, index: int) -> TrainingDataType:
        record = self.records[index]
        mel = self._calculate_mel(record.audio_path, record.start, record.end)
        decoder_input, decoder_output = self.get_decoder_io(record.text, record.language)
        y_in, y_out = torch.tensor(decoder_input), torch.tensor(decoder_output)
        return mel, y_in, y_out

    def collate_fn(self, data) -> Any:
        mel, y_in, y_out = zip(*data)
        mel = pad_sequence(mel, batch_first=True, padding_value=0)  # type: ignore
        mel = mel.half()
        y_in = pad_sequence(y_in, batch_first=True, padding_value=0)  # type: ignore
        y_out = pad_sequence(y_out, batch_first=True, padding_value=-100)  # type: ignore
        return mel, y_in, y_out


@dataclass
class DataFilter:
    datasets: list[str] = Field(default_factory=list)
    langs: list[str] = Field(default_factory=list)


@dataclass
class WhisperDataConfig:
    audio: str = "original"
    train_filter: DataFilter = Field(default_factory=DataFilter)
    val_filter: DataFilter = Field(default_factory=DataFilter)
    batch_size: int = 4
    num_workers: int = 4


def filter_songs(records: list[Record], filter: DataFilter) -> list[Record]:
    filtered = []
    for record in records:
        if filter.datasets and record.dataset not in filter.datasets:
            continue
        if filter.langs and record.language not in filter.langs:
            continue
        filtered.append(record)
    return filtered


class WhisperDataModule(LightningDataModule):
    def __init__(self, songs_path: Path, config: WhisperDataConfig, multilingual: bool):
        super().__init__()
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.songs_path = songs_path
        self.train_filter = config.train_filter
        self.val_filter = config.val_filter
        self.audio = config.audio
        self.multilingual = multilingual

        self.save_hyperparameters(asdict(config))

    def setup(self, stage: str):
        tokenizer = get_tokenizer(multilingual=self.multilingual, task="transcribe")
        songs = util.read_datafile(self.songs_path, Song)
        separated = None if self.audio == "original" else self.audio
        records = songs_to_records(
            songs, separated=separated, num_workers=self.num_workers
        )

        if stage == "fit":
            self.train_dataset = AudioDataset(
                records=filter_songs(records, self.train_filter),
                tokenizer=tokenizer,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = AudioDataset(
                records=filter_songs(records, self.val_filter),
                tokenizer=tokenizer,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn,  # Use custom collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate_fn,  # Use custom collate_fn
        )
