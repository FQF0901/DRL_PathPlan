"""Scene-level contiguous sequence batches for BC/DAgger training.

Shuffling is applied to episodes, never to individual frames.  Each batch item
is one episode's frame sequence; frames inside an episode remain ordered.  The
trainer may further split each sequence into TBPTT chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SceneBatch:
    """Padded row indices for a batch of scenes.

    ``rows[b, t]`` is the frame row at decision index ``t`` for scene ``b``;
    rows beyond ``lengths[b]`` are padded with that scene's last valid row so
    tensors stay rectangular.  Callers must mask those positions.
    """

    rows: np.ndarray
    lengths: np.ndarray

    @property
    def batch_size(self) -> int:
        return int(self.rows.shape[0])

    @property
    def max_length(self) -> int:
        return int(self.rows.shape[1])


def _episode_sequences(
    data: Mapping[str, np.ndarray], indices: np.ndarray
) -> list[np.ndarray]:
    episode = np.asarray(data["episode_id"], dtype=np.int64)
    step = np.asarray(data["step"], dtype=np.int64)
    selected = np.asarray(indices, dtype=np.int64)
    sequences: list[np.ndarray] = []
    for ep in np.unique(episode[selected]):
        rows = selected[episode[selected] == ep]
        rows = rows[np.argsort(step[rows], kind="stable")]
        if len(rows):
            sequences.append(rows.astype(np.int64, copy=False))
    return sequences


def make_scene_batches(
    data: Mapping[str, np.ndarray],
    indices: np.ndarray,
    scenes_per_batch: int,
    *,
    rng: Optional[np.random.Generator] = None,
    clip_length: Optional[int] = None,
) -> list[SceneBatch]:
    """Group rows into episode sequences and pack ``scenes_per_batch`` of them.

    If ``rng`` is provided, episode order is shuffled; frame order never is.
    """
    if int(scenes_per_batch) < 1:
        raise ValueError("scenes_per_batch must be >= 1")
    sequences = _episode_sequences(data, indices)
    if clip_length is not None and int(clip_length) > 0:
        clipped: list[np.ndarray] = []
        for sequence in sequences:
            for start in range(0, len(sequence), int(clip_length)):
                part = sequence[start : start + int(clip_length)]
                if len(part):
                    clipped.append(part)
        sequences = clipped
    if rng is not None:
        order = rng.permutation(len(sequences))
        sequences = [sequences[int(i)] for i in order]
    batches: list[SceneBatch] = []
    for start in range(0, len(sequences), int(scenes_per_batch)):
        chunk = sequences[start : start + int(scenes_per_batch)]
        lengths = np.asarray([len(seq) for seq in chunk], dtype=np.int64)
        max_len = int(lengths.max()) if len(lengths) else 0
        rows = np.empty((len(chunk), max_len), dtype=np.int64)
        for i, seq in enumerate(chunk):
            rows[i, : len(seq)] = seq
            if len(seq) < max_len:
                rows[i, len(seq) :] = seq[-1]
        batches.append(SceneBatch(rows=rows, lengths=lengths))
    return batches


def iter_scene_chunks(
    batch: SceneBatch, chunk_length: int
) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` TBPTT chunks for one scene batch."""
    if int(chunk_length) < 1:
        raise ValueError("chunk_length must be >= 1")
    for start in range(0, batch.max_length, int(chunk_length)):
        yield start, min(batch.max_length, start + int(chunk_length))
