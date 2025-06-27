# STDLIB
from dataclasses import asdict, replace
from functools import partial
import logging
from uu import decode

# THIRDPARTY
import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
import whisper
from whisper import Whisper
from whisper.tokenizer import get_tokenizer
from whisper.decoding import DecodingTask, DecodingOptions, DecodingResult
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from pydantic.dataclasses import dataclass
from transformers.optimization import get_linear_schedule_with_warmup
from torcheval.metrics.functional import word_error_rate

# FIRSTPARTY
from .data import TrainingDataType

checkpoint_func = partial(checkpoint, use_reentrant=False)

logger = logging.getLogger(__name__)


def transcribe(model: Whisper, mel: torch.Tensor, options: DecodingOptions) -> str:
    compression_ratio_threshold = 2.4
    logprob_threshold = -1.0
    no_speech_threshold = 0.6
    temperatures = [0, 0.2, 0.4, 0.6, 0.8]
    options = replace(options)

    decode_result: DecodingResult | None = None
    for t in temperatures:
        if t > 0:
            # disable beam_size and patience when t > 0
            options = replace(options, beam_size=None, patience=None, temperature=t)
        else:
            # disable best_of when t == 0
            options = replace(options, best_of=None, temperature=0)

        decode_result = model.decode(mel, options)  # type: ignore
        assert decode_result is not None

        needs_fallback = False
        if (
            decode_result.compression_ratio > compression_ratio_threshold
            or decode_result.avg_logprob < logprob_threshold
        ):
            needs_fallback = True  # average log probability is too low
        if (
            no_speech_threshold is not None
            and decode_result.no_speech_prob > no_speech_threshold
        ):
            needs_fallback = False  # silence
        if not needs_fallback:
            break

    assert decode_result is not None
    if (
        decode_result.no_speech_prob > no_speech_threshold
        and decode_result.avg_logprob < logprob_threshold
    ):
        return ""

    return decode_result.text


@dataclass
class WhisperModuleConfig:
    model: str = "large-v2"
    lr: float = 1e-6
    warmup_steps: int = 50
    activation_checkpointing: bool = False


Prediction = tuple[str, str]


class WhisperModule(LightningModule):
    def __init__(self, config: WhisperModuleConfig):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.lr = config.lr
        self.warmup_steps = config.warmup_steps
        self.activation_checkpointing = config.activation_checkpointing
        self.validation_predictions: list[Prediction] = []
        self.tokenizer = get_tokenizer(
            multilingual=".en" not in config.model, task="transcribe"
        )
        self.special_tokens_reverse = {
            v: k for k, v in self.tokenizer.special_tokens.items()
        }
        self.model_name = config.model
        self.model: Whisper | None = None

    def configure_model(self):
        if self.model is not None:
            return None
        self.model = whisper.load_model(self.model_name, device=self.device)
        del self.model.alignment_heads

    def model_step(
        self, batch: TrainingDataType, batch_idx: int, name: str
    ) -> torch.Tensor:
        assert self.model is not None
        # mel dimensions: (batch size, channels, bins, frames)
        # token dimensions: (batch size, max sequence length)
        mel, y_in, y_out = batch
        if self.activation_checkpointing:
            audio_features = checkpoint_func(self.model.embed_audio, mel)
            logits = checkpoint_func(self.model.logits, y_in, audio_features)
        else:
            audio_features = self.model.embed_audio(mel)
            logits = self.model.logits(y_in, audio_features)
        loss = F.cross_entropy(logits.transpose(1, 2), y_out)  # type: ignore
        self.log(
            f"{name}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: TrainingDataType, batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch, batch_idx, "train")
        scheduler = self.lr_schedulers()
        if self.trainer.is_global_zero:
            self.log(
                "lr",
                scheduler.get_last_lr()[0],  # type: ignore
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        return loss

    def validation_step(self, batch: TrainingDataType, batch_idx: int) -> torch.Tensor:
        # full eval
        # mels, y_ins, y_out = batch
        # for mel, y_in in zip(mels, y_ins):
        #     language = self.special_tokens_reverse[int(y_in[1])][2:4]
        #     options = DecodingOptions(
        #         language=language,
        #         beam_size=5,
        #         best_of=5,
        #         without_timestamps=True,
        #     )
        #     assert self.model is not None
        #     with torch.no_grad():
        #         hyp = transcribe(self.model, mel, options)
        #     ref = self.tokenizer.decode(y_in[4:].tolist()).rstrip("!")
        #     self.validation_predictions.append((ref, hyp))
        # get validation loss
        return self.model_step(batch, batch_idx, "eval")

    # def on_validation_epoch_end(self) -> None:
    #     # Gather predictions from all processes
    #     if torch.distributed.is_available() and torch.distributed.is_initialized():
    #         all_predictions: list[Prediction | None] = [None] * self.trainer.world_size
    #         torch.distributed.all_gather_object(
    #             all_predictions, self.validation_predictions
    #         )

    #         gathered_predictions = []
    #         for proc_predictions in all_predictions:
    #             gathered_predictions.extend(proc_predictions)
    #     else:
    #         gathered_predictions = self.validation_predictions

    #     # Evaluate WER
    #     if self.trainer.is_global_zero:
    #         if len(gathered_predictions) > 0:
    #             refs = [p[0] for p in gathered_predictions]
    #             hyps = [p[1] for p in gathered_predictions]
    #             wer = word_error_rate(hyps, refs).item()
    #             self.log(
    #                 "eval_wer", wer, prog_bar=True, sync_dist=False, rank_zero_only=True
    #             )
    #     self.validation_predictions.clear()

    def configure_optimizers(self) -> dict:  # type: ignore
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
