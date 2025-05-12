from alt.merge import merge_lines, group_lines
import pytest


def float_pair(pair):
    """Helper to compare float start/end pairs with approx"""
    return pytest.approx(pair, rel=1e-3)


def test_no_overlap_results_in_individual_segments():
    lines = [
        {"start": 0.0, "end": 1.0, "text": "A"},
        {"start": 2.0, "end": 3.0, "text": "B"},
        {"start": 4.0, "end": 5.0, "text": "C"},
    ]
    merged = merge_lines(lines, audio_duration=10.0, padding_buffer=0, song_id="song")
    assert len(merged) == 3
    starts = [seg["start"] for seg in merged]
    ends = [seg["end"] for seg in merged]
    texts = [seg["text"] for seg in merged]
    # Padding applied but no merges: first start remains 0, last end capped by audio_duration
    assert starts == pytest.approx([0.0, 1.5, 3.5])
    assert ends == pytest.approx([1.5, 3.5, 5.5])
    assert texts == ["A", "B", "C"]


def test_overlapping_lines_are_merged():
    lines = [
        {"start": 1.0, "end": 2.0, "text": "Hello"},
        {"start": 1.5, "end": 2.5, "text": "World"},
    ]
    merged = merge_lines(lines, audio_duration=5.0, song_id="song")
    assert len(merged) == 1
    seg = merged[0]
    assert seg["start"] == pytest.approx(0.5)
    assert seg["end"] == pytest.approx(3.0)
    assert seg["text"] == "HelloWorld"


def test_overlap_below_threshold_not_merged():
    lines = [
        {"start": 0.0, "end": 1.0, "text": "X"},
        {"start": 0.9, "end": 2.0, "text": "Y"},
    ]
    merged = merge_lines(lines, audio_duration=10.0, song_id="song")
    assert len(merged) == 2
    texts = [seg["text"] for seg in merged]
    assert set(texts) == {"X", "Y"}


def test_exclude_long_segment():
    lines = [{"start": 0.0, "end": 40.0, "text": "Long"}]
    merged = merge_lines(lines, audio_duration=50.0, song_id="song")
    assert merged == []


def test_custom_threshold_and_max_duration():
    lines = [
        {"start": 0.0, "end": 1.0, "text": "A"},
        {"start": 1.05, "end": 2.0, "text": "B"},
        {"start": 1.9, "end": 3.0, "text": "C"},
    ]
    merged = merge_lines(
        lines,
        audio_duration=10.0,
        song_id="song",
        overlap_threshold=0.1,
        max_duration=2.0,
    )
    assert len(merged) == 2
    group_texts = sorted([seg["text"] for seg in merged])
    assert group_texts == ["A", "BC"]
    for seg in merged:
        # Compare duration with a small tolerance (1e-3) to account for floating point inaccuracies
        assert seg["end"] - seg["start"] <= 2.0 + 1e-3


def test_padding_does_not_create_new_overlap():
    # Two non-overlapping lines close enough that naive padding could overlap
    lines = [
        {"start": 1.0, "end": 2.0, "text": "First"},
        {"start": 2.8, "end": 3.8, "text": "Second"},
    ]
    # Set padding to 0.5 (default) and no custom thresholds
    merged = merge_lines(lines, audio_duration=10.0, song_id="song", padding=0.5)
    assert len(merged) == 2
    # Sort segments
    segs = sorted(merged, key=lambda s: s["start"])
    first, second = segs
    # Ensure that the padding of the first segment does not reach into the original start of the second
    assert (
        first["end"] <= lines[1]["start"] + 1e-3
    ), f"Padding caused overlap: first.end={first['end']}, lines[1]['start']={lines[1]['start']}"
