from __future__ import annotations

import json
from pathlib import Path

import torch

from data.low_level import BCDatasetManifest, BCFrameRecord, BCSequenceWindow, write_jsonl
from policies.low_level.pixelbc_lstm import PixelBCLSTMConfig, PixelBCLSTMPolicy
from policies.low_level.training_utils import (
    binary_control_metrics,
    compute_control_balance,
)


def test_pixelbc_lstm_forward_shape_and_backward() -> None:
    config = PixelBCLSTMConfig(
        num_actions=5,
        embedding_dim=32,
        lstm_hidden_size=32,
        lstm_num_layers=1,
        encoder_weights="none",
    )
    model = PixelBCLSTMPolicy.build(config)
    frames = torch.rand(1, 2, 3, 64, 64)
    targets = torch.randint(0, 2, (1, 2, 5), dtype=torch.float32)
    logits = model(frames)
    assert tuple(logits.shape) == (1, 2, 5)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def _write_balance_dataset(root: Path) -> None:
    root.mkdir()
    manifest = BCDatasetManifest(
        game_id="planet",
        source_root="synthetic",
        control_names=("left", "attack"),
        intent_names=(),
        skill_names=(),
        sequence_length=2,
        stride=1,
        gap_factor=3.0,
        frame_count=4,
        sequence_count=2,
        sessions=("train", "val"),
        train_sessions=("train",),
        val_sessions=("val",),
    )
    manifest.write(root / "manifest.json")
    frames = [
        BCFrameRecord("train", 0, 0, "a.png", (True, False)),
        BCFrameRecord("train", 1, 1, "b.png", (False, False)),
        BCFrameRecord("val", 0, 0, "c.png", (False, True)),
        BCFrameRecord("val", 1, 1, "d.png", (False, True)),
    ]
    write_jsonl(root / "frames.jsonl", [row.to_dict() for row in frames])
    windows = [
        BCSequenceWindow("train", 0, 2),
        BCSequenceWindow("val", 2, 2),
    ]
    write_jsonl(root / "windows.jsonl", [row.to_dict() for row in windows])


def test_control_balance_counts_unique_train_frames(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_balance_dataset(dataset)
    balance = compute_control_balance(dataset, split="train")
    assert balance.positives == (1, 0)
    assert balance.negatives == (1, 2)
    assert balance.pos_weight == (1.0, 1.0)


def test_binary_metrics_per_control() -> None:
    logits = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])
    targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    metrics = binary_control_metrics(logits, targets)
    assert metrics["f1"] == [1.0, 1.0]
    assert metrics["accuracy"] == [1.0, 1.0]
