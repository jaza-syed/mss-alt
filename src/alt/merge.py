import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from scipy.sparse.csgraph import connected_components as scipy_connected_components

logger = logging.getLogger(__name__)


def find_connected_components(matrix: np.ndarray) -> list[list[int]]:
    """
    Finds and returns the connected components of an undirected graph
    represented by an adjacency matrix.

    Args:
        matrix (np.ndarray): A 2D adjacency matrix representing the graph.

    Returns:
        list[list[int]]: A sorted list of connected components, where each
        component is represented as a list of node indices.
    """
    num_components, labels = scipy_connected_components(
        csgraph=csr_array(matrix), directed=False, return_labels=True
    )
    # Return connected groups (sorted for consistency)
    components = [np.where(labels == i)[0].tolist() for i in range(num_components)]
    return sorted(components)


def merge_lines(
    lines: list[dict],
    audio_duration: float,
    song_id: str,
    overlap_threshold: float = 0.2,
    max_duration: float = 30,
    padding: float = 0.5,
    padding_buffer: float = 0.1,
) -> list[dict]:
    """
    Merges overlapping lines of text based on lyric timing information.

    Overview of the algorithm:
    1. Find groups of overlapping lines using a pairwise overlap matrix.
    2. Compute lower (start time) and upper (end time) padding limits.
       a. Lower limit is latest end time of lines not in the group + the buffer
       b. Upper limit is earliest start time of lines not in the group - the buffer
    3. Pad up to the limits, constrained by the maximum duration of the segment.

    Note that the resulting segments may have overlaps in time due to the padding,
    however this padding does not extend into the vocal activity of the next segment.
    This function is adapted from an inital implementation by Ondrej Cifka.

    Args:
        lines (list[dict]): A list of dictionaries, each containing 'start', 'end', and 'text' keys.
        audio_duration (float): Total duration of the audio in seconds.
        song_id (str): Identifier for the song or audio segment.
        overlap_threshold (float, optional): Minimum overlap ratio to consider lines as connected. Defaults to 0.2.
        max_duration (float, optional): Maximum allowed duration for a merged segment. Defaults to 30 seconds.
        padding (float, optional): Padding to apply to segment boundaries, constrained by limits. Defaults to 0.5 seconds.

    Returns:
        list[dict]: A list of merged line dictionaries with 'start', 'end', and 'text' keys.
    """
    df = pd.DataFrame(lines)
    # Compute the full pairwise overlap matrix for all lines.
    # overlap between line i and j is: max(0, min(end_i, end_j) - max(start_i, start_j))
    starts = df["start"].values
    ends = df["end"].values
    overlap_starts = np.maximum(starts[:, None], starts)  # type: ignore
    overlap_ends = np.minimum(ends[:, None], ends)  # type: ignore
    overlaps = np.maximum(0.0, overlap_ends - overlap_starts)

    # Get groups of lines where the overlap exceeds the hard threshold.
    # Only consider overlaps above the threshold when forming groups.
    overlap_groups = find_connected_components(overlaps > overlap_threshold)

    merged_lines = []
    for group_indices in overlap_groups:
        group = df.iloc[group_indices].to_dict(orient="records")
        group_name = (
            f"{min(group_indices):03d}-{max(group_indices):03d}"
            if len(group_indices) > 1
            else f"{group_indices[0]:03d}"
        )

        # Determine the boundary of the group based on the timings.
        seg_start = min(item["start"] for item in group)
        seg_end = max(item["end"] for item in group)
        start, end = seg_start, seg_end

        # Pad the segment boundaries if possible, without creating overlaps and without exceeding max_duration.
        # Determine padding limits based on lines not in the current group.
        # latest end time + buffer of groups before current
        l_limit = max(
            [
                0.0,
                *(
                    t + padding_buffer
                    for t in df.drop(index=group_indices)["end"]
                    if t < end
                ),
            ]
        )
        # earliest start time - buffer of groups before current
        r_limit = min(
            [
                *(
                    t - padding_buffer
                    for t in df.drop(index=group_indices)["start"]
                    if t > start
                ),
                audio_duration,
            ]
        )
        # apply padding to expand up to l_limit, r_limit
        max_total_pad = max(0.0, max_duration - (end - start))
        l_pad = min(max(0.0, start - l_limit), padding)
        r_pad = min(max(0.0, r_limit - end), padding)
        if l_pad + r_pad > max_total_pad:
            l_pad = r_pad = min(l_pad, r_pad, max_total_pad / 2)
            extra = max(0.0, max_total_pad - (l_pad + r_pad)) - 1e-3
            l_pad += extra / 2
            r_pad += extra / 2
            assert l_pad + r_pad < max_total_pad
        start, end = start - l_pad, end + r_pad

        # Update the dataframe with the new start and end times
        # df.loc[group_indices, "start"] = start
        # df.loc[group_indices, "end"] = end

        text = "".join(li["text"] for li in group)

        if end - start > max_duration:
            logger.info(
                f"Excluding segment {song_id}.{group_name} of duration {end - start:.2f}"
            )
            continue

        merged_lines.append({"start": start, "end": end, "text": text})
    return merged_lines


def maximize_min_duration(segments: list[dict], max_duration: float):
    """
    Partition a list of segments into groups to maximize the minimum duration
    of any group, subject to a maximum duration constraint.

    Args:
        segments (list of dict): A list of segments, where each segment is a
            dictionary with "start" and "end" keys representing time intervals.
        max_duration (float): The maximum allowed duration for any group.

    Returns:
        list of list of dict: A partition of the input segments into groups
        that maximizes the minimum duration of any group.

    Raises:
        ValueError: If no valid partition can be found within the constraints.
    """
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dp(i) -> tuple[float, list[list[dict]] | None]:
        if i == len(segments):
            return float("inf"), []
        best = (float("-inf"), None)
        for j in range(i + 1, len(segments) + 1):
            group = segments[i:j]
            group_duration = group[-1]["end"] - group[0]["start"]
            if group_duration >= max_duration:
                break
            sub_best, sub_partition = dp(j)
            min_duration = min(group_duration, sub_best)
            if min_duration > best[0]:
                assert sub_partition is not None
                best = (min_duration, [group] + sub_partition)
        return best

    _, result = dp(0)
    if result is None:
        raise ValueError("No valid partition found")
    return result


def group_lines(
    lines: list[dict], group_threshold: float = 7, max_duration: float = 30
) -> list[dict]:
    """
    Groups ASR segments into longer segments while ensuring constraints on duration.

        lines (list[dict]): List of ASR segments with "start", "end", and "text" keys.
        group_threshold (float, optional): Threshold for splitting lines for first phase of grouping.
        max_duration (float, optional): Maximum duration (in seconds) for any
            grouped segment in the second phase. Defaults to 30.

        list[dict]: List of grouped segments with combined text and adjusted
        start and end times, ensuring durations are below the specified maximum.
    """
    df_segments = pd.DataFrame(lines)
    df_segments = df_segments.sort_values(by="start")

    # Phase 1: Find indices at which start of next is more than 7 secs after end of prev
    diffs = df_segments["start"].iloc[1:].values - df_segments["end"].iloc[:-1].values  # type: ignore
    split_indices = (diffs > group_threshold).nonzero()[0]
    if len(split_indices) == 0 or split_indices[0] != 0:
        split_indices = [0] + (split_indices + 1).tolist()
    else:
        split_indices = (split_indices + 1).tolist()
    split_indices.append(len(df_segments))

    # Phase 2: Split groups further w/ minimum subgroup dur maximised + below 30s
    under_30s: list[list[dict]] = []  # list of list of segments
    for i in range(len(split_indices) - 1):
        group_dicts = df_segments.iloc[split_indices[i] : split_indices[i + 1]].to_dict(
            "records"
        )
        grouped = maximize_min_duration(group_dicts, max_duration)
        under_30s.extend(grouped)

    # Construct segments from groups
    grouped_lines = [
        {
            "text": "".join(line["text"] for line in group),
            "start": np.min([line["start"] for line in group]),
            "end": np.max([line["end"] for line in group]),
        }
        for group in under_30s
    ]
    return grouped_lines
