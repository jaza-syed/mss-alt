# stdlib
from enum import Enum
from typing import Any
from pathlib import Path
from dataclasses import field

# third-party
import jiwer
from pydantic.dataclasses import dataclass
from pydantic import Field


####### Extract types
# Class to represent line annotations
@dataclass
class TimedText:
    text: str
    start: float
    end: float
    extra: dict = field(default_factory=dict)


# Class to represent information about a song
@dataclass
class Song:
    uid: str  # {dataset_id}_{song_id}
    dataset_id: str
    song_id: str
    text: str  # long-form lyric transcript
    split: str  # train / test
    duration_secs: float
    language: str  # ISO639 language code
    audio_path: Path
    merged_lines: list[TimedText] = field(default_factory=list)
    grouped_lines: list[TimedText] = field(default_factory=list)
    stem_dir: Path | None = None


@dataclass
class ExtractSummary:
    uids: list[str] = field(default_factory=list)


####### Separate types
@dataclass
class SeparateConfig:
    model_name: str = "mdx_extra"
    model_type: str = "demucs"
    apply_args: dict = field(default_factory=dict)  # args to pass to demucs
    stems: tuple[str] = ("vocals",)
    cached: bool = True  # do not separate if completion file exists


@dataclass
class SeparationTask:
    """Container for a separation task with its arguments"""

    uid: str
    audio_path: Path
    task_id: int | None = None


@dataclass
class SeparateSummary:
    tasks: list[SeparationTask]
    config: SeparateConfig
    path: Path


####### Inference types
class InferLevel(str, Enum):
    SONG = "song"
    MERGED_LINE = "merged_line"
    GROUPED_LINE = "grouped_line"


class ModelType(str, Enum):
    WHISPER = "whisper"


class LongFormAlgo(str, Enum):
    NATIVE = "native"
    RMSVAD = "rmsvad"


@dataclass
class VadOptions:
    # Default values used in ICME Workshop Paper
    onset: float = 0.1
    offset: float = 0.1
    min_speech_duration_ms: float = 0
    max_speech_duration_s: float = 30
    min_silence_duration_ms: float = 1000
    speech_pad_ms: float = 200


@dataclass
class InferConfig:
    # Inference type
    level: InferLevel = InferLevel.SONG
    audio: str = "original"  # "original", "stem", or separation name
    algo: LongFormAlgo = LongFormAlgo.NATIVE
    # Model params
    model_type: ModelType = ModelType.WHISPER
    model_name: str = "large-v2"
    batched: bool = True
    add_lang: bool = True
    transcribe_args: dict = field(default_factory=dict)
    model_init_args: dict = field(default_factory=dict)
    vad_audio: str | None = None
    vad_options: VadOptions | None = None  # must be non-None if algo == RMSVAD
    # Filters
    datasets: list[str] = field(default_factory=list)
    splits: list[str] = field(default_factory=list)
    langs: list[str] = field(default_factory=list)
    # Other
    num_workers: int = 10
    cache_dir: Path | None = None


# Class to represent a timed section of ASR output
# Similar to TimedText, but a different class for clarity / future adjustment
@dataclass
class AsrSegment:
    text: str
    start: float
    end: float
    word_timestamps: dict | None = None


# Class to represent a single ref/hyp pair with **reference** timing information if available
# To work wth ref/hyp in the context of alignments, we need to do
# tokenized_{ref|hyp} = metrics.tokens_as_word(tokenizer({ref|hyp}, language))
# NOTE: ref / hyp are not normalized or tokenized
@dataclass
class AsrSample:
    ref: str
    hyp: str
    start: float
    end: float


# Results of running RMS-VAD
@dataclass
class VadData:
    audio_path: Path
    scores: list[float]
    segments: list[dict]
    sampling_rate: int
    window_size_samples: int


@dataclass
class AsrTask:
    """Container for a transcription task with its arguments"""

    uid: str
    audio_path: Path
    transcribe_kwargs: dict[str, Any]
    task_id: int | None = None


@dataclass
class InferResult:
    audio_path: Path  # path to the audio the model was evaluated on
    asr_task: AsrTask  # record of the task, for reference
    asr_output: list[AsrSegment] = Field(
        default_factory=list  # , validation_alias="asr_result"
    )  # Output of ASR model
    asr_samples: list[AsrSample] = field(
        default_factory=list
    )  # non-normalised pairs of ASR model outputs and references
    infer_level: InferLevel = InferLevel.SONG
    vad_data: VadData | None = None


@dataclass
class Token:
    """A "rich" token (with a set of associated tags)."""

    text: str
    tags: set = field(default_factory=set)


### Evaluation
@dataclass
class EvalResult:
    metrics: dict[str, Any]
    # jiwer "words" are the result of the **text** of tokens_as_words(tokenizer(str))
    # the output of tokenizer is **rich** tokens, but jiwer is just using plain tokens
    wo: jiwer.WordOutput
    ref_tokens: list[list[Token]]  # rich tokens (including non-word tokens)
    hyp_tokens: list[list[Token]]  # rich tokens (including non-word tokens)


@dataclass
class Result:
    song: Song
    infer: InferResult  # = Field(validation_alias="result")
    eval: EvalResult | None = None


@dataclass
class RenderArgs:
    title: str
    description: str = ""
