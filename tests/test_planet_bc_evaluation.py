from __future__ import annotations

import torch

from policies.low_level.evaluation import (
    calibrate_thresholds,
    mean_f1_for_controls,
    metrics_at_thresholds,
)


def test_metrics_at_per_control_thresholds() -> None:
    probabilities = torch.tensor(
        [
            [[0.80, 0.30], [0.20, 0.70]],
            [[0.90, 0.10], [0.10, 0.90]],
        ]
    )
    targets = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )
    metrics = metrics_at_thresholds(probabilities, targets, (0.5, 0.5))
    assert metrics["f1"] == [1.0, 1.0]


def test_calibration_lowers_threshold_for_rare_control() -> None:
    probabilities = torch.tensor(
        [
            [0.40, 0.90],
            [0.35, 0.80],
            [0.20, 0.10],
            [0.10, 0.20],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    result = calibrate_thresholds(
        probabilities,
        targets,
        ("jump", "attack"),
        threshold_min=0.1,
        threshold_max=0.9,
        threshold_step=0.05,
    )
    assert result.thresholds[0] < 0.5
    assert result.metrics["f1"][0] == 1.0
    assert result.metrics["f1"][1] == 1.0


def test_calibration_keeps_default_for_unseen_control() -> None:
    probabilities = torch.tensor([[0.8], [0.2], [0.7]])
    targets = torch.zeros_like(probabilities)
    result = calibrate_thresholds(
        probabilities,
        targets,
        ("interact",),
    )
    assert result.thresholds == (0.5,)
    assert result.metrics["f1"] == [0.0]


def test_primary_mean_f1_ignores_auxiliary_controls() -> None:
    metrics = {"f1": [0.8, 0.6, 0.0]}
    assert mean_f1_for_controls(
        ("left", "attack", "ability_3"),
        metrics,
        ("left", "attack"),
    ) == 0.7


def test_calibration_can_leave_auxiliary_control_at_default() -> None:
    probabilities = torch.tensor(
        [
            [0.40, 0.30],
            [0.35, 0.25],
            [0.20, 0.10],
            [0.10, 0.20],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    result = calibrate_thresholds(
        probabilities,
        targets,
        ("jump", "ability_3"),
        threshold_min=0.1,
        threshold_max=0.9,
        threshold_step=0.05,
        selected_control_names=("jump",),
    )
    assert result.thresholds[0] < 0.5
    assert result.thresholds[1] == 0.5
