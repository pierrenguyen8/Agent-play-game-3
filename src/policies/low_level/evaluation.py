from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ThresholdCalibration:
    control_names: tuple[str, ...]
    thresholds: tuple[float, ...]
    metrics: dict[str, list[float] | list[int]]


def _as_flat_probabilities(probabilities, targets):
    import torch

    if probabilities.shape != targets.shape:
        raise ValueError(
            "probabilities/targets shape mismatch: "
            f"{probabilities.shape} != {targets.shape}"
        )
    if probabilities.ndim < 2:
        raise ValueError("probabilities must have a final control dimension")
    width = probabilities.shape[-1]
    return (
        probabilities.detach().float().reshape(-1, width).cpu(),
        targets.detach().float().reshape(-1, width).cpu(),
    )


def metrics_at_thresholds(
    probabilities,
    targets,
    thresholds: Sequence[float],
) -> dict[str, list[float] | list[int]]:
    """Compute per-control binary metrics using one threshold per control."""

    import torch

    probs, truth_values = _as_flat_probabilities(probabilities, targets)
    width = probs.shape[-1]
    if len(thresholds) != width:
        raise ValueError(f"Expected {width} thresholds, got {len(thresholds)}")

    threshold_tensor = torch.tensor(
        [float(value) for value in thresholds],
        dtype=probs.dtype,
    ).view(1, width)
    predicted = probs >= threshold_tensor
    truth = truth_values >= 0.5

    tp = (predicted & truth).sum(dim=0)
    fp = (predicted & ~truth).sum(dim=0)
    fn = (~predicted & truth).sum(dim=0)
    tn = (~predicted & ~truth).sum(dim=0)

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


def calibrate_thresholds(
    probabilities,
    targets,
    control_names: Sequence[str],
    *,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_step: float = 0.01,
    default_threshold: float = 0.5,
    selected_control_names: Sequence[str] | None = None,
) -> ThresholdCalibration:
    """Choose the validation F1-maximizing threshold independently per control.

    Controls with no positive targets retain ``default_threshold``. F1 ties are
    resolved toward the threshold closest to the default, keeping calibration
    conservative and deterministic.
    """

    import torch

    probs, truth_values = _as_flat_probabilities(probabilities, targets)
    names = tuple(str(name) for name in control_names)
    selected_names = (
        set(names)
        if selected_control_names is None
        else {str(name) for name in selected_control_names}
    )
    width = probs.shape[-1]
    if len(names) != width:
        raise ValueError(f"Expected {width} control names, got {len(names)}")
    if not 0.0 <= threshold_min <= threshold_max <= 1.0:
        raise ValueError("threshold range must satisfy 0 <= min <= max <= 1")
    if threshold_step <= 0:
        raise ValueError("threshold_step must be > 0")

    count = int(round((threshold_max - threshold_min) / threshold_step)) + 1
    grid = [
        min(threshold_max, threshold_min + index * threshold_step)
        for index in range(count)
    ]
    if not grid:
        raise ValueError("threshold grid is empty")

    truth = truth_values >= 0.5
    selected: list[float] = []
    epsilon = 1e-12

    for control_index in range(width):
        control_truth = truth[:, control_index]
        if names[control_index] not in selected_names:
            selected.append(float(default_threshold))
            continue
        if int(control_truth.sum()) <= 0:
            selected.append(float(default_threshold))
            continue

        control_probs = probs[:, control_index]
        best_threshold = float(default_threshold)
        best_f1 = -1.0
        best_distance = float("inf")

        for threshold in grid:
            predicted = control_probs >= float(threshold)
            tp = int((predicted & control_truth).sum())
            fp = int((predicted & ~control_truth).sum())
            fn = int((~predicted & control_truth).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 0.0 if precision + recall == 0 else (
                2.0 * precision * recall / (precision + recall)
            )
            distance = abs(float(threshold) - float(default_threshold))
            if (
                f1 > best_f1 + epsilon
                or (
                    abs(f1 - best_f1) <= epsilon
                    and distance < best_distance - epsilon
                )
            ):
                best_f1 = f1
                best_threshold = float(threshold)
                best_distance = distance

        selected.append(best_threshold)

    metrics = metrics_at_thresholds(probs, truth_values, selected)
    return ThresholdCalibration(
        control_names=names,
        thresholds=tuple(selected),
        metrics=metrics,
    )


def mean_f1_for_controls(
    control_names: Sequence[str],
    metrics: dict[str, list[float] | list[int]],
    selected_names: Sequence[str],
) -> float:
    names = tuple(control_names)
    indices = [names.index(name) for name in selected_names if name in names]
    if not indices:
        return 0.0
    f1 = metrics["f1"]
    return sum(float(f1[index]) for index in indices) / len(indices)
