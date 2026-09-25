from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.low_level import TorchBCSequenceDataset
from policies.low_level.evaluation import (
    calibrate_thresholds,
    mean_f1_for_controls,
    metrics_at_thresholds,
)
from policies.low_level.pixelbc_lstm import PixelBCLSTMConfig, PixelBCLSTMPolicy


DEFAULT_PRIMARY_CONTROLS = (
    "left",
    "right",
    "up",
    "down",
    "jump",
    "attack",
    "ability_0",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_checkpoint(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a dictionary: {path}")
    for key in ("model_state_dict", "model_config", "control_names"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing {key!r}: {path}")
    return checkpoint


def _resolve_run_config(checkpoint_path: Path) -> dict:
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        return {}
    return _read_json(config_path)


def _resolve_dataset_dir(arg_value: str | None, run_config: dict) -> Path:
    raw = arg_value or run_config.get("dataset_dir")
    if not raw:
        raise SystemExit(
            "Dataset directory is unknown. Pass --dataset-dir or keep config.json "
            "beside the checkpoint."
        )
    return Path(str(raw)).expanduser().resolve()


def _resolve_image_size(args, run_config: dict) -> tuple[int, int]:
    training = run_config.get("training")
    if not isinstance(training, dict):
        training = {}
    width = args.image_width or int(training.get("image_width", 160))
    height = args.image_height or int(training.get("image_height", 90))
    if width <= 0 or height <= 0:
        raise SystemExit("image dimensions must be > 0")
    return width, height


def _metric_rows(control_names, metrics, thresholds):
    rows = []
    for index, name in enumerate(control_names):
        rows.append(
            {
                "control": name,
                "threshold": float(thresholds[index]),
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


def _print_rows(title: str, rows: list[dict], primary_controls: tuple[str, ...]) -> None:
    print(title)
    for row in rows:
        marker = "*" if row["control"] in primary_controls else " "
        print(
            f"{marker} {row['control']:>10s} "
            f"thr={row['threshold']:.2f} "
            f"P={row['precision']:.4f} "
            f"R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} "
            f"tp={row['tp']:5d} fp={row['fp']:5d} fn={row['fn']:5d}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a Planet BC checkpoint on the validation split and "
            "calibrate one held-button threshold per control."
        )
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    root = _repo_root()
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (root / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _load_checkpoint(checkpoint_path)
    run_config = _resolve_run_config(checkpoint_path)
    dataset_dir = _resolve_dataset_dir(args.dataset_dir, run_config)
    image_width, image_height = _resolve_image_size(args, run_config)

    output_path = Path(args.output).expanduser() if args.output else (
        checkpoint_path.parent / "thresholds.json"
    )
    if not output_path.is_absolute():
        output_path = (root / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    control_names = tuple(str(name) for name in checkpoint["control_names"])
    checkpoint_primary = checkpoint.get("primary_control_names")
    if checkpoint_primary:
        primary_controls = tuple(
            str(name) for name in checkpoint_primary if str(name) in control_names
        )
    else:
        primary_controls = tuple(
            name for name in DEFAULT_PRIMARY_CONTROLS if name in control_names
        )

    model_config = PixelBCLSTMConfig(**checkpoint["model_config"])
    if model_config.num_actions != len(control_names):
        raise ValueError(
            "Checkpoint model action count does not match checkpoint control_names"
        )
    model = PixelBCLSTMPolicy.build(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    device = _device(args.device)
    model.to(device)
    model.eval()

    dataset = TorchBCSequenceDataset(
        dataset_dir,
        split="val",
        image_width=image_width,
        image_height=image_height,
    )
    if len(dataset) <= 0:
        raise SystemExit("Validation split has no sequence windows")
    if tuple(dataset.index.manifest.control_names) != control_names:
        raise ValueError(
            "Dataset control_names do not match checkpoint control_names"
        )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    probabilities_parts = []
    targets_parts = []
    batch_count = 0
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                if args.max_val_batches is not None and batch_index >= args.max_val_batches:
                    break
                frames = batch["frames"].to(device, non_blocking=True)
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if args.amp and device.type == "cuda"
                    else nullcontext()
                )
                with autocast_context:
                    logits = model(frames)
                probabilities_parts.append(torch.sigmoid(logits).cpu())
                targets_parts.append(batch["held"].cpu())
                batch_count += 1
    finally:
        dataset.close()

    if batch_count <= 0:
        raise RuntimeError("No validation batches were processed")

    probabilities = torch.cat(probabilities_parts, dim=0)
    targets = torch.cat(targets_parts, dim=0)
    default_thresholds = [0.5] * len(control_names)
    default_metrics = metrics_at_thresholds(
        probabilities,
        targets,
        default_thresholds,
    )
    calibration = calibrate_thresholds(
        probabilities,
        targets,
        control_names,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        default_threshold=0.5,
        selected_control_names=primary_controls,
    )

    default_primary_f1 = mean_f1_for_controls(
        control_names, default_metrics, primary_controls
    )
    calibrated_primary_f1 = mean_f1_for_controls(
        control_names, calibration.metrics, primary_controls
    )
    default_rows = _metric_rows(control_names, default_metrics, default_thresholds)
    calibrated_rows = _metric_rows(
        control_names, calibration.metrics, calibration.thresholds
    )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "dataset_dir": str(dataset_dir),
        "split": "val",
        "val_sequences": len(dataset),
        "evaluated_batches": batch_count,
        "evaluated_window_positions": int(targets.numel() // len(control_names)),
        "control_names": list(control_names),
        "primary_control_names": list(primary_controls),
        "image_width": image_width,
        "image_height": image_height,
        "threshold_search": {
            "min": args.threshold_min,
            "max": args.threshold_max,
            "step": args.threshold_step,
            "note": (
                "Thresholds are calibrated on the validation split and are for "
                "inference configuration, not an independent test estimate."
            ),
        },
        "default": {
            "thresholds": dict(zip(control_names, default_thresholds)),
            "primary_mean_f1": default_primary_f1,
            "controls": default_rows,
        },
        "calibrated": {
            "thresholds": dict(zip(control_names, calibration.thresholds)),
            "primary_mean_f1": calibrated_primary_f1,
            "controls": calibrated_rows,
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {payload['checkpoint_epoch']}")
    print(f"Device: {device}")
    print(f"Val sequences: {len(dataset)}")
    print(f"Evaluated window positions: {payload['evaluated_window_positions']}")
    print("* = primary control")
    _print_rows("Default thresholds (0.50):", default_rows, primary_controls)
    print(f"Default primary mean F1: {default_primary_f1:.4f}")
    print()
    _print_rows("Validation-calibrated thresholds:", calibrated_rows, primary_controls)
    print(f"Calibrated primary mean F1: {calibrated_primary_f1:.4f}")
    print(f"Threshold artifact: {output_path}")


if __name__ == "__main__":
    main()
