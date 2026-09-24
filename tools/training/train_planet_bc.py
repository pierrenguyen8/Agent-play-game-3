from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.low_level import TorchBCSequenceDataset
from policies.low_level.pixelbc_lstm import PixelBCLSTMConfig, PixelBCLSTMPolicy
from policies.low_level.training_utils import (
    binary_control_metrics,
    compute_control_balance,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _pixelbc_provenance(root: Path) -> dict[str, object]:
    audit_path = root / "external" / "pixelbc.audit.json"
    commit_path = root / "external" / "pixelbc.commit.txt"
    provenance: dict[str, object] = {
        "repository": (
            "https://github.com/microsoft/"
            "imitation_learning_in_modern_video_games.git"
        ),
        "architecture_reused": (
            "visual embedding -> projection -> LSTM -> action projection"
        ),
    }
    if audit_path.exists():
        audit = _read_json(audit_path)
        provenance["audit"] = audit
        provenance["audit_path"] = str(audit_path)
        files = audit.get("files") or []
        if not any(
            isinstance(entry, dict)
            and entry.get("path") == "pixelbc/models/utils/model_utils.py"
            and "LSTM" in ((entry.get("symbols") or {}).get("classes") or [])
            for entry in files
        ):
            raise ValueError(
                "PixelBC audit does not confirm model_utils.LSTM; rerun audit"
            )
    if commit_path.exists():
        provenance["commit"] = commit_path.read_text(encoding="utf-8").strip()
    elif isinstance(provenance.get("audit"), dict):
        provenance["commit"] = provenance["audit"].get("commit")
    return provenance


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _loader(dataset, *, batch_size: int, shuffle: bool, num_workers: int):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def _accumulate_metric_counts(total: dict[str, list[int]], metrics: dict[str, object]) -> None:
    for key in ("tp", "fp", "fn", "tn"):
        values = [int(v) for v in metrics[key]]
        if key not in total:
            total[key] = values
        else:
            total[key] = [a + b for a, b in zip(total[key], values)]


def _finalize_metrics(counts: dict[str, list[int]]) -> dict[str, list[float] | list[int]]:
    result: dict[str, list[float] | list[int]] = dict(counts)
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    accuracy: list[float] = []
    for tp, fp, fn, tn in zip(
        counts.get("tp", []),
        counts.get("fp", []),
        counts.get("fn", []),
        counts.get("tn", []),
    ):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precision.append(p)
        recall.append(r)
        f1.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
        accuracy.append((tp + tn) / max(tp + fp + fn + tn, 1))
    result.update(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )
    return result


def _run_epoch(
    *,
    model,
    loader,
    criterion,
    device: torch.device,
    optimizer=None,
    max_batches: int | None = None,
    amp: bool = False,
    grad_clip: float = 1.0,
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batch_count = 0
    metric_counts: dict[str, list[int]] = {}

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        frames = batch["frames"].to(device, non_blocking=True)
        targets = batch["held"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp and device.type == "cuda"
                else nullcontext()
            )
            with autocast_context:
                logits = model(frames)
                loss = criterion(logits, targets)
            if training:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        batch_count += 1
        _accumulate_metric_counts(
            metric_counts,
            binary_control_metrics(logits.detach(), targets.detach()),
        )

    if batch_count <= 0:
        raise RuntimeError("No batches were processed")
    return total_loss / batch_count, _finalize_metrics(metric_counts)


def _metric_rows(names: tuple[str, ...], metrics: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "control": name,
                "precision": float(metrics["precision"][index]),
                "recall": float(metrics["recall"][index]),
                "f1": float(metrics["f1"][index]),
                "accuracy": float(metrics["accuracy"][index]),
                "tp": int(metrics["tp"][index]),
                "fp": int(metrics["fp"][index]),
                "fn": int(metrics["fn"][index]),
                "tn": int(metrics["tn"][index]),
            }
        )
    return rows


def _save_checkpoint(path: Path, *, model, optimizer, epoch: int, payload: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **payload,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Planet held-button BC with a PixelBC-style LSTM policy."
    )
    parser.add_argument("dataset_dir")
    parser.add_argument(
        "--output",
        default="runs/planet/low_level/pixelbc_lstm",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=160)
    parser.add_argument("--image-height", type=int, default=90)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument(
        "--encoder-weights",
        choices=("none", "imagenet"),
        default="none",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-pos-weight", type=float, default=50.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be > 0")

    root = _repo_root()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = TorchBCSequenceDataset(
        dataset_dir,
        split="train",
        image_width=args.image_width,
        image_height=args.image_height,
    )
    val_dataset = TorchBCSequenceDataset(
        dataset_dir,
        split="val",
        image_width=args.image_width,
        image_height=args.image_height,
    )
    if len(train_dataset) <= 0:
        raise SystemExit("Training split has no sequence windows")
    if len(val_dataset) <= 0:
        raise SystemExit("Validation split has no sequence windows")

    manifest = train_dataset.index.manifest
    control_names = manifest.control_names
    balance = compute_control_balance(
        dataset_dir,
        split="train",
        max_pos_weight=args.max_pos_weight,
    )

    config = PixelBCLSTMConfig(
        num_actions=len(control_names),
        embedding_dim=args.embedding_dim,
        lstm_hidden_size=args.hidden_size,
        lstm_num_layers=args.lstm_layers,
        encoder_weights=args.encoder_weights,
        freeze_encoder=args.freeze_encoder,
    )
    model = PixelBCLSTMPolicy.build(config)
    device = _device(args.device)
    model.to(device)

    pos_weight = torch.tensor(balance.pos_weight, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_loader = _loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = _loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    provenance = _pixelbc_provenance(root)
    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "control_names": list(control_names),
        "dataset_manifest": manifest.to_dict(),
        "model": config.to_dict(),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "amp": bool(args.amp),
            "seed": args.seed,
            "max_train_batches": args.max_train_batches,
            "max_val_batches": args.max_val_batches,
        },
        "class_balance": asdict(balance),
        "pixelbc_provenance": provenance,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    print("Controls:")
    for name, pos, neg, weight in zip(
        balance.control_names,
        balance.positives,
        balance.negatives,
        balance.pos_weight,
    ):
        print(
            f"  {name:>10s} pos={pos:6d} neg={neg:6d} "
            f"pos_weight={weight:7.3f}"
        )

    metrics_path = output_dir / "metrics.jsonl"
    best_val = float("inf")
    checkpoint_payload = {
        "control_names": control_names,
        "model_config": config.to_dict(),
        "dataset_manifest": manifest.to_dict(),
        "pixelbc_provenance": provenance,
    }

    try:
        with metrics_path.open("w", encoding="utf-8") as metrics_file:
            for epoch in range(1, args.epochs + 1):
                train_loss, train_metrics = _run_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=optimizer,
                    max_batches=args.max_train_batches,
                    amp=args.amp,
                    grad_clip=args.grad_clip,
                )
                with torch.no_grad():
                    val_loss, val_metrics = _run_epoch(
                        model=model,
                        loader=val_loader,
                        criterion=criterion,
                        device=device,
                        optimizer=None,
                        max_batches=args.max_val_batches,
                        amp=args.amp,
                        grad_clip=args.grad_clip,
                    )

                record = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_controls": _metric_rows(control_names, train_metrics),
                    "val_controls": _metric_rows(control_names, val_metrics),
                }
                metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                metrics_file.flush()

                mean_f1 = sum(val_metrics["f1"]) / max(len(control_names), 1)
                print(
                    f"Epoch {epoch:03d}: "
                    f"train_loss={train_loss:.5f} "
                    f"val_loss={val_loss:.5f} "
                    f"val_mean_f1={mean_f1:.4f}"
                )

                _save_checkpoint(
                    output_dir / "last.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    payload={**checkpoint_payload, "metrics": record},
                )
                if val_loss < best_val:
                    best_val = val_loss
                    _save_checkpoint(
                        output_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        payload={**checkpoint_payload, "metrics": record},
                    )
    finally:
        train_dataset.close()
        val_dataset.close()

    print(f"Best validation loss: {best_val:.6f}")
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
