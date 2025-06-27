import collections
from dataclasses import dataclass
import logging
from typing import Any, Union
import unittest

import iso639
import jiwer
from rapidfuzz.distance import Levenshtein

from .tokenizer import (
    LINE,
    PAREN,
    PUNCT,
    SECT,
    BACKING,
    NONLEXICAL,
    LyricsTokenizer,
    Token,
    tokens_as_words,
)

IDENTITY_TRANSFORM = jiwer.Compose([])


@dataclass
class EditOpCounts:
    """A counter for edit operations (hits, substitutions, deletions, insertions)."""

    H: int = 0
    S: int = 0
    D: int = 0
    I: int = 0  # noqa: E741


def process_alignment_chunk(
    reference: list[Token],
    hypothesis: list[Token],
    chunk_type: str,
    counts: dict[Any, EditOpCounts],
    count_substitutions: bool = True,
) -> None:
    """
    Count tag-specific edit operations in a chunk of an alignment.
    NOTE: counts is modified in place!
    If a tag is ref-only, it will be counted in delete as normal
    """
    ref_only_tags = {NONLEXICAL, BACKING}
    if chunk_type == "delete":
        assert len(hypothesis) == 0
        for token in reference:
            for tag in token.tags:
                counts[tag].D += 1
    elif chunk_type == "insert":
        assert len(reference) == 0
        for token in hypothesis:
            for tag in token.tags:
                counts[tag].I += 1
    elif chunk_type in ["substitute", "equal"]:
        assert len(reference) == len(hypothesis)
        for token_ref, token_hyp in zip(reference, hypothesis):
            # add {nonlexical, backing} to common_tags so that substitutions / hits are counted as normal
            common_tags = (
                (token_ref.tags & token_hyp.tags) if count_substitutions else set()
            )
            ref_only_tags = {
                tag for tag in [NONLEXICAL, BACKING] if tag in token_ref.tags
            }
            # NOTE: do not count tagged deletions here, as tags are not to indicate a special char
            # But to indicate a type of word
            for tag in token_ref.tags - (common_tags | ref_only_tags):
                counts[tag].D += 1
            for tag in token_hyp.tags - common_tags:
                counts[tag].I += 1
            if chunk_type == "substitute":
                for tag in common_tags | ref_only_tags:
                    counts[tag].S += 1
            elif chunk_type == "equal":
                for tag in common_tags | ref_only_tags:
                    counts[tag].H += 1
    else:
        assert False, f"Unhandled chunk type: {chunk_type}"


def process_alignments(
    references: list[list[Token]],
    hypotheses: list[list[Token]],
    alignments: list[list[jiwer.AlignmentChunk]],
    count_substitutions: bool = True,
) -> tuple[dict[Any, EditOpCounts], dict[str, int]]:
    """Count tag-specific edit operations in a list of alignments."""
    # dict tag->counter
    edit_counts = collections.defaultdict(EditOpCounts)
    error_counts = collections.defaultdict(int)

    for i in range(len(references)):
        for chunk in alignments[i]:
            chunk_hyp = hypotheses[i][chunk.hyp_start_idx : chunk.hyp_end_idx]
            chunk_ref = references[i][chunk.ref_start_idx : chunk.ref_end_idx]
            process_alignment_chunk(
                chunk_ref,
                chunk_hyp,
                chunk.type,
                edit_counts,
                count_substitutions=count_substitutions,
            )

            if chunk.type == "equal":
                for token_ref, token_hyp in zip(chunk_ref, chunk_hyp):
                    if token_ref.text != token_hyp.text:
                        assert token_ref.text.lower() == token_hyp.text.lower()
                        error_counts["case"] += 1
            if chunk.type == "substitute":
                for token_ref, token_hyp in zip(chunk_ref, chunk_hyp):
                    if near_miss(token_ref, token_hyp):
                        error_counts["near"] += 1
            if chunk.type == "delete":
                for token_ref, _token_hyp in zip(chunk_ref, chunk_hyp):
                    if BACKING in token_ref.tags:
                        error_counts["del_backing"] += 1
                    if NONLEXICAL in token_ref.tags:
                        error_counts["del_nonlexical"] += 1

    return edit_counts, error_counts


def near_miss(token_ref: Token, token_hyp: Token) -> bool:
    # From https://www.arxiv.org/abs/2408.06370 (Cifka, 2024)
    # we count a near hit if, after removing apostrophes from the two words,
    # their character-level Levenshtein distance is at most 2 and strictly
    # less than half the length of the longer of the two words
    ref = token_ref.text.lower().strip("'")
    hyp = token_hyp.text.lower().strip("'")
    edit_ops = len(Levenshtein.editops(ref, hyp))
    max_len = max(len(ref), len(hyp))
    return edit_ops in [1, 2] and max_len > edit_ops * 2


class TestNearMiss(unittest.TestCase):
    def test_an_and(self):
        assert near_miss("an", "and") is True

    def test_gon_gonna(self):
        assert near_miss("gon'", "gonna") is True

    def test_there_their_they_them(self):
        from itertools import combinations

        for ref, hyp in combinations(["their", "they", "there", "them"], 2):
            assert near_miss(ref, hyp) is True

    def test_a_an(self):
        assert near_miss("a", "an") is False

    def test_this_that(self):
        assert near_miss("this", "that") is False


def compute_word_metrics(
    references: list[list[Token]],
    hypotheses: list[list[Token]],
    count_substitutions: bool = True,
) -> tuple[dict[str, Any], jiwer.WordOutput]:
    references = [tokens_as_words(tokens) for tokens in references]
    hypotheses = [tokens_as_words(tokens) for tokens in hypotheses]

    wo = jiwer.process_words(
        [[t.text.lower() for t in tokens] for tokens in references],  # type: ignore
        [[t.text.lower() for t in tokens] for tokens in hypotheses],  # type: ignore
        reference_transform=IDENTITY_TRANSFORM,
        hypothesis_transform=IDENTITY_TRANSFORM,
    )

    _, error_counts = process_alignments(
        references,
        hypotheses,
        wo.alignments,
        count_substitutions=count_substitutions,
    )
    total_len = sum(len(tokens) for tokens in references)

    results = {
        "WER": wo.wer,
        "WER_near": wo.wer - error_counts["near"] / total_len,  # WER without near-misses
        "MER": wo.mer,  # match error rate
        "WIL": wo.wil,  # word information lost
        "hits": wo.hits,
        "near": error_counts["near"] / total_len,
        "substitutions": wo.substitutions,
        "deletions": wo.deletions,
        "insertions": wo.insertions,
        "nears": error_counts["near"],
        "sub_rate": wo.substitutions / total_len,
        "del_rate": wo.deletions / total_len,
        "ins_rate": wo.insertions / total_len,
        "ER_case": error_counts["case"] / total_len,
        "WER_case": wo.wer + error_counts["case"] / total_len,
        "total_len": total_len,
        "deletions_backing": error_counts["del_backing"],
        "deletions_nonlexical": error_counts["del_nonlexical"],
    }
    return results, wo


def compute_other_metrics(
    references: list[list[Token]],
    hypotheses: list[list[Token]],
    count_substitutions: bool = True,
) -> dict[str, Any]:
    wo = jiwer.process_words(
        [[t.text.lower() for t in tokens] for tokens in references],  # type: ignore
        [[t.text.lower() for t in tokens] for tokens in hypotheses],  # type: ignore
        reference_transform=IDENTITY_TRANSFORM,
        hypothesis_transform=IDENTITY_TRANSFORM,
    )

    counts, _ = process_alignments(
        references,
        hypotheses,
        wo.alignments,
        count_substitutions=count_substitutions,
    )

    results = {}
    for tag in [PUNCT, PAREN, LINE, SECT]:
        tg = tag[:4]
        H, S, D, I = counts[tag].H, counts[tag].S, counts[tag].D, counts[tag].I  # noqa: E741
        P = H / (H + S + I) if H + S + I else float("nan")
        R = H / (H + S + D) if H + S + D else float("nan")
        results[f"P_{tg}"] = P
        results[f"R_{tg}"] = R
        if P == float("nan") or R == float("nan"):
            results[f"F1_{tg}"] = float("nan")
        elif P + R == 0:
            results[f"F1_{tg}"] = 0.0
        else:
            results[f"F1_{tg}"] = 2 * P * R / (P + R)

    return results


def compute_metrics(
    references: list[str],
    hypotheses: list[str],
    languages: Union[list[str], str] = "en",
    include_other: bool = True,
) -> tuple[dict[str, Any], jiwer.WordOutput]:
    """Compute all metrics for the given references and hypotheses.

    Args:
        references: A list of reference transcripts.
        hypotheses: A list of hypotheses.
        languages: The language of each reference transcript or a single language to use for all
            transcripts.
        include_other: Whether to compute non-word metrics
        nonlexical_line_idxs: idxs of lines in the references that are nonlexical

    Returns:
        A dictionary of metrics.
    """
    if isinstance(languages, str):
        languages = [languages] * len(references)
    languages = [iso639.Language.match(lg).part1 for lg in languages]  # type: ignore

    tokenizer = LyricsTokenizer()
    tokens_ref: list[list[Token]] = []
    tokens_hyp: list[list[Token]] = []
    for i in range(len(references)):
        tokens_ref.append(tokenizer(references[i], language=languages[i]))
        tokens_hyp.append(tokenizer(hypotheses[i], language=languages[i]))

    results, wo = compute_word_metrics(tokens_ref, tokens_hyp)
    if include_other:
        results.update(
            compute_other_metrics(
                tokens_ref,
                tokens_hyp,
            )
        )

    return results, wo
