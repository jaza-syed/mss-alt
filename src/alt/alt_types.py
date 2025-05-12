# stdlib
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
from typing import Any
from pathlib import Path

# third-party
import numpy as np


@dataclass_json
@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass_json
@dataclass
class VadOptions:
    onset: float = 0.5
    offset: float = onset - 0.15
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = 30
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400


### Extraction
@dataclass_json
@dataclass
class Timing:
    text: str
    start: float
    end: float
    extra: dict = field(default_factory=dict)


@dataclass_json
@dataclass
class SongData:
    uid: str
    dataset_id: str
    song_id: str
    text: str
    duration: float  # seconds
    language: str
    split: str
    audio_fn: Path = field(metadata=config(encoder=str, decoder=Path))
    merged_lines: list[Timing] = field(default_factory=list)
    grouped_lines: list[Timing] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass_json
@dataclass
class JamAltConfig:
    languages: list[str]
    revision: str = "v1.1.0"


@dataclass_json
@dataclass
class MusdbAltConfig:
    musdb_dir: Path = field(
        metadata=config(encoder=str, decoder=Path),
    )
    revision: str = "0.2.0"


@dataclass_json
@dataclass
class ExtractConfig:
    jam_alt: JamAltConfig | None = None
    musdb_alt: MusdbAltConfig | None = None


### Preprocessing
@dataclass_json
@dataclass
class PreprocData:
    vocals_fn: Path = field(metadata=config(decoder=Path, encoder=str))
    other_fn: Path = field(metadata=config(decoder=Path, encoder=str))


@dataclass_json
@dataclass
class PreprocConfig:
    separate: bool = False
    demucs_model_name: str = "mdx_extra"
    demucs_args: dict = field(default_factory=dict)


### Inference
@dataclass_json
@dataclass
class VadResult:
    scores: np.ndarray = field(
        metadata=config(decoder=np.asarray, encoder=lambda arr: arr.tolist())
    )  # vad results by frame
    segments: list[dict]  # keys start and end, values are sample indices
    sampling_rate: int
    window_size_samples: int


@dataclass_json
@dataclass
class InferData:
    infer_target: str
    audio_fn: Path = field(metadata=config(decoder=Path, encoder=str))
    whisper_result: list[Segment] = field(default_factory=list)
    timings: list[Timing] = field(default_factory=list)
    vad_result: VadResult | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class TranscriptionTask:
    """Container for a transcription task with its arguments"""

    uid: str
    audio_path: Path
    transcribe_kwargs: dict[str, Any]
    task_id: int | None = None
    timings: list[Timing] = field(default_factory=list)


@dataclass_json
@dataclass
class InferConfig:
    model_type: str = "whisper"
    # Device to run inference on
    device: str = "cuda"
    # What level of timestamps to inform inference
    # - "song" means no timestamps
    # - "line" means using line-level timestamps
    # - "verse" means using verse-level timestamps
    infer_level: str = "song"
    # which audio file to run inference on
    # - "original" means the original audio
    # - "separated_vocals" means the separated vocals from demcus
    # - "stem_vocals" means the vocal stems
    infer_target: str = "original"
    # use RMS of separated vocals to infer clip timestamps
    vocal_rms_vad: bool = False
    vad_options: VadOptions = field(default_factory=VadOptions)
    # provide text language at inference time
    add_lang: bool = True
    # Use batched inference
    batched: bool = False
    # model name
    model: str = "large-v2"
    # args for Speech2Text.decode or model.transcribe
    transcribe_args: dict = field(default_factory=dict)
    # args for WhisperModel or Speech2Text.from_pretrained
    model_init_args: dict = field(default_factory=dict)
    # filters - if non-empty, select fromt hese
    datasets: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    langs: list[str] = field(default_factory=list)


### Evaluation
@dataclass_json
@dataclass
class EvalData:
    metrics: dict[str, Any]
    refs: list[str]
    hyps: list[str]


@dataclass_json
@dataclass
class EvalConfig:
    description: str
    nooverlap: bool = False
    render: bool = True


### Container
@dataclass_json
@dataclass
class SongInfo:
    extract: SongData
    preproc: PreprocData | None = None
    infer: InferData | None = None
    evaluate: EvalData | None = None


@dataclass_json
@dataclass
class PipelineConfig:
    name: str
    extract: ExtractConfig
    preproc: PreprocConfig
    infer: InferConfig
    evaluate: EvalConfig


@dataclass_json
@dataclass
class ExtractSummary:
    uids: list[str]
    info: dict[str, dict]  # dataset->information
    cfg: ExtractConfig


@dataclass_json
@dataclass
class InferSummary:
    uids: list[str]
    info: dict  # any extra information
    cfg: InferConfig


@dataclass_json
@dataclass
class EvalSummary:
    uids: list[str]
    info: dict  # any extra information
    cfg: EvalConfig


@dataclass_json
@dataclass
class PreprocSummary:
    uids: list[str]
    info: dict  # any extra information
    cfg: PreprocConfig


@dataclass_json
@dataclass
class Summary:
    extract: ExtractSummary
    preproc: PreprocSummary | None = None
    infer: InferSummary | None = None
    evaluate: EvalSummary | None = None
