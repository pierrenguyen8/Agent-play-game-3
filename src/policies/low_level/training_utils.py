from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from data.low_level import BCSequenceIndex


@dataclass(frozen=True, slots=True)
class ControlBalance:
    control_names: tuple[str, ...]
    positives: tuple[int, ...]
    negatives: tuple[int, ...]
    pos_weight: tuple[float, ...]


def compute_control_balance(
    dataset_dir: str | Path,
    *,
    split: str = "train",
    max_pos_weight: float = 50.0,
) -> ControlBalance:
    """Count each exported frame once, not once per overlapping window."""

    if max_pos_weight <= 0:
        raise ValueError("max_pos_weight must be > 0")

    index = BCSequenceIndex(dataset_dir, split="all")
    if split == "train":
        sessions = set(index.manifest.train_sessions)
    elif split == "val":
        sessions = set(index.manifest.val_sessions)
    elif split == "all":
        sessions = set(index.manifest.sessions)
    else:
        raise ValueError(f"Unknown split: {split}")

    width = len(index.manifest.control_names)
    positives = [0] * width
    total = 0
    for row in index.frames:
        if row.session_id not in sessions:
            continue
        if len(row.held) != width:
            raise ValueError(
                f"Held vector width mismatch in {row.session_id}: "
                f"{len(row.held)} != {width}"
            )
        total += 1
        for action_index, value in enumerate(row.held):
            positives[action_index] += int(bool(value))

    if total <= 0:
        raise ValueError(f"Split {split!r} contains no frames")

    negatives = [total - count for count in positives]
    weights: list[float] = []
    for pos, neg in zip(positives, negatives):
        if pos <= 0:
            # No positive supervision exists for this control. Keep a neutral
            # weight and let metrics expose that the control is unseen.
            weights.append(1.0)
        else:
            weights.append(min(float(neg) / float(pos), max_pos_weight))

    return ControlBalance(
        control_names=index.manifest.control_names,
        positives=tuple(positives),
        negatives=tuple(negatives),
        pos_weight=tuple(weights),
    )


def binary_control_metrics(
    logits,
    targets,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    import torch

    if logits.shape != targets.shape:
        raise ValueError(
            f"logits/targets shape mismatch: {logits.shape} != {targets.shape}"
        )
    probabilities = torch.sigmoid(logits)
    predicted = probabilities >= threshold
    truth = targets >= 0.5

    reduce_dims = tuple(range(logits.ndim - 1))
    tp = (predicted & truth).sum(dim=reduce_dims).cpu()
    fp = (predicted & ~truth).sum(dim=reduce_dims).cpu()
    fn = (~predicted & truth).sum(dim=reduce_dims).cpu()
    tn = (~predicted & ~truth).sum(dim=reduce_dims).cpu()

    precision = tp.float() / (tp + fp).clamp_min(1).float()
    recall = tp.float() / (tp + fn).clamp_min(1).float()
    f1 = (2 * precision * recall) / (precision + recall).clamp_min(1e-12)
    accuracy = (tp + tn).float() / (tp + fp + fn + tn).clamp_min(1).float()

    return {
        "tp": tp.tolist(),
        "fp": fp.tolist(),
        "fn": fn.tolist(),
        "tn": tn.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "accuracy": accuracy.tolist(),
    }
