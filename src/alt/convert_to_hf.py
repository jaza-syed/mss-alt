#!/usr/bin/env python
"""Converts a Whisper model in OpenAI format to Hugging Face format."""
# Copyright 2022 The HuggingFace Inc. team and the OpenAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import tempfile
import warnings
from typing import Optional, Any
from collections import OrderedDict

from whisper import ModelDimensions
from tiktoken.load import load_tiktoken_bpe
from torch import nn
from transformers import (
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    WhisperTokenizerFast,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.whisper.tokenization_whisper import LANGUAGES, bytes_to_unicode
from transformers.utils.import_utils import _is_package_available


_TOKENIZERS = {
    "multilingual": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
    "english": "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken",
}


def _get_generation_config(
    is_multilingual: bool,
    num_languages: int = 100,
    openai_version: Optional[str] = None,
) -> GenerationConfig:
    """
    Loads the appropriate generation config from HF repo
    """
    if openai_version is not None:
        repo = f"openai/whisper-{openai_version}"
    elif not is_multilingual:
        repo = "openai/whisper-medium.en"
    elif num_languages < 100:
        repo = "openai/whisper-large-v2"
    else:
        repo = "openai/whisper-large-v3"

    gen_cfg = GenerationConfig.from_pretrained(repo)
    if openai_version is None:
        gen_cfg.alignment_heads = None  # type: ignore
        warnings.warn(
            "Alignment heads have not been included in the generation config, since they are available "
            "only for the original OpenAI checkpoints."
            "If you want to use word-level timestamps with a custom version of Whisper,"
            "see https://github.com/openai/whisper/blob/main/notebooks/Multilingual_ASR.ipynb"
            "for the example of how to produce word-level timestamps manually."
        )

    return gen_cfg


def remove_ignore_keys_(state_dict):
    ignore_keys = ["layers", "blocks"]
    for k in ignore_keys:
        state_dict.pop(k, None)


WHISPER_MAPPING = {
    "blocks": "layers",
    "mlp.0": "fc1",
    "mlp.2": "fc2",
    "mlp_ln": "final_layer_norm",
    ".attn.query": ".self_attn.q_proj",
    ".attn.key": ".self_attn.k_proj",
    ".attn.value": ".self_attn.v_proj",
    ".attn_ln": ".self_attn_layer_norm",
    ".attn.out": ".self_attn.out_proj",
    ".cross_attn.query": ".encoder_attn.q_proj",
    ".cross_attn.key": ".encoder_attn.k_proj",
    ".cross_attn.value": ".encoder_attn.v_proj",
    ".cross_attn_ln": ".encoder_attn_layer_norm",
    ".cross_attn.out": ".encoder_attn.out_proj",
    "decoder.ln.": "decoder.layer_norm.",
    "encoder.ln.": "encoder.layer_norm.",
    "token_embedding": "embed_tokens",
    "encoder.positional_embedding": "encoder.embed_positions.weight",
    "decoder.positional_embedding": "decoder.embed_positions.weight",
    "ln_post": "layer_norm",
}


def rename_keys(s_dict):
    keys = list(s_dict.keys())
    for key in keys:
        new_key = key
        for k, v in WHISPER_MAPPING.items():
            if k in key:
                new_key = new_key.replace(k, v)

        print(f"{key} -> {new_key}")

        s_dict[new_key] = s_dict.pop(key)
    return s_dict


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def convert_openai_whisper_to_tfms(
    state_dict: dict[str, Any], dimensions: ModelDimensions
) -> tuple[WhisperForConditionalGeneration, bool, int]:
    openai_version = None

    proj_out_weights = state_dict["decoder.token_embedding.weight"]
    remove_ignore_keys_(state_dict)
    rename_keys(state_dict)
    tie_embeds = True
    ffn_dim = state_dict["decoder.layers.0.fc1.weight"].shape[0]

    # a hacky way to properly set up the bos/eos/pad token ids in the model
    endoftext_id = 50257 if dimensions.n_vocab > 51865 else 50256

    config = WhisperConfig(
        vocab_size=dimensions.n_vocab,
        encoder_ffn_dim=ffn_dim,
        decoder_ffn_dim=ffn_dim,
        num_mel_bins=dimensions.n_mels,
        d_model=dimensions.n_audio_state,
        max_target_positions=dimensions.n_text_ctx,
        encoder_layers=dimensions.n_audio_layer,
        encoder_attention_heads=dimensions.n_audio_head,
        decoder_layers=dimensions.n_text_layer,
        decoder_attention_heads=dimensions.n_text_head,
        max_source_positions=dimensions.n_audio_ctx,
        eos_token_id=endoftext_id,
        bos_token_id=endoftext_id,
        pad_token_id=endoftext_id,
        decoder_start_token_id=endoftext_id + 1,
    )

    model = WhisperForConditionalGeneration(config)
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 and not set(missing) <= {
        "encoder.embed_positions.weights",
        "decoder.embed_positions.weights",
    }:
        raise ValueError(
            "Only `encoder.embed_positions.weights` and `decoder.embed_positions.weights`  are allowed to be missing,"
            f" but all the following weights are missing {missing}"
        )

    if tie_embeds:
        model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)
    else:
        model.proj_out.weight.data = proj_out_weights

    # determine those parameters from a model checkpoint as Whisper repo does
    is_multilingual = model.config.vocab_size >= 51865
    num_languages = model.config.vocab_size - 51765 - int(is_multilingual)

    model.generation_config = _get_generation_config(
        is_multilingual,
        num_languages,
        openai_version,
    )

    return model, is_multilingual, num_languages


# Adapted from https://github.com/openai/tiktoken/issues/60#issuecomment-1499977960
def _bpe(mergeable_ranks, token: bytes, max_rank=None) -> list[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = (
            parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]
        )
    return parts


def convert_tiktoken_bpe_to_hf(tiktoken_url: str):
    bpe_ranks = load_tiktoken_bpe(tiktoken_url)
    byte_encoder = bytes_to_unicode()

    def token_bytes_to_string(b):
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    merges = []
    vocab = {}
    for token, rank in bpe_ranks.items():
        vocab[token_bytes_to_string(token)] = rank
        if len(token) == 1:
            continue
        merged = tuple(_bpe(bpe_ranks, token, max_rank=rank))
        if len(merged) == 2:  # account for empty token
            merges.append(" ".join(map(token_bytes_to_string, merged)))
    return vocab, merges


def convert_tiktoken_to_hf(
    multilingual: bool = True, num_languages: int = 100, time_precision=0.02
) -> WhisperTokenizer:
    # requires whisper, unless we use the path to the tiktoken file
    tiktoken_tokenizer_path = _TOKENIZERS["multilingual" if multilingual else "english"]
    start_of_transcript = ["<|endoftext|>", "<|startoftranscript|>"]
    control_tokens = [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ]
    # these are special tokens, not normalized
    language_tokens = [f"<|{k}|>" for k in list(LANGUAGES)[:num_languages]]
    # These are not special but normalized
    timestamp_tokens = [("<|%.2f|>" % (i * time_precision)) for i in range(1500 + 1)]

    vocab, merges = convert_tiktoken_bpe_to_hf(tiktoken_tokenizer_path)

    with tempfile.TemporaryDirectory() as tmpdirname:
        vocab_file = f"{tmpdirname}/vocab.json"
        merge_file = f"{tmpdirname}/merges.txt"
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            )

        with open(merge_file, "w", encoding="utf-8") as writer:
            writer.write("#version: 0.2\n")
            for bpe_tokens in merges:
                writer.write(bpe_tokens + "\n")

        hf_tokenizer = WhisperTokenizer(vocab_file, merge_file)

    hf_tokenizer.add_tokens(
        start_of_transcript + language_tokens + control_tokens, special_tokens=True
    )
    hf_tokenizer.add_tokens(timestamp_tokens, special_tokens=False)
    return hf_tokenizer


def convert_whisper(
    state_dict: dict[str, Any],
    dims: ModelDimensions,
    pytorch_dump_folder_path: os.PathLike,
):
    model, is_multilingual, num_languages = convert_openai_whisper_to_tfms(
        state_dict, dims
    )

    tokenizer = convert_tiktoken_to_hf(is_multilingual, num_languages)
    feature_extractor = WhisperFeatureExtractor(
        feature_size=model.config.num_mel_bins,
        # the rest of default parameters are the same as hardcoded in openai/whisper
    )
    processor = WhisperProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
    processor.save_pretrained(pytorch_dump_folder_path)

    # save fast tokenizer as well
    fast_tokenizer = WhisperTokenizerFast.from_pretrained(pytorch_dump_folder_path)
    fast_tokenizer.save_pretrained(pytorch_dump_folder_path, legacy_format=False)

    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument(
        "--checkpoint_path", type=str, help="Path to the downloaded checkpoints"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--convert_preprocessor",
        type=bool,
        default=False,
        help="Whether or not the preprocessor (tokenizer + feature extractor) should be converted along with the model.",
    )
    args = parser.parse_args()
    convert_whisper(
        args.checkpoint_path, args.pytorch_dump_folder_path, args.convert_preprocessor
    )
