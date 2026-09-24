from __future__ import annotations

import argparse
from pathlib import Path

from games.planet.low_level.dataset_export import (
    PlanetBCExportConfig,
    export_planet_bc_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Planet recorder sessions to a PixelBC-style sequence index."
    )
    parser.add_argument("source_root", type=Path, help="Root containing Planet session folders")
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path("data/processed/planet/bc_pixelbc"),
    )
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--gap-factor", type=float, default=3.0)
    parser.add_argument("--num-ability-slots", type=int, default=4)
    parser.add_argument(
        "--allow-missing-frames",
        action="store_true",
        help="Index frame paths without verifying that every image currently exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_planet_bc_dataset(
        args.source_root,
        args.output_dir,
        config=PlanetBCExportConfig(
            sequence_length=args.sequence_length,
            stride=args.stride,
            val_fraction=args.val_fraction,
            split_seed=args.split_seed,
            gap_factor=args.gap_factor,
            num_ability_slots=args.num_ability_slots,
            require_frame_files=not args.allow_missing_frames,
        ),
    )
    manifest = result.manifest
    print(f"Exported to: {result.output_dir}")
    print(f"Sessions: {len(manifest.sessions)}")
    print(f"Frames: {manifest.frame_count}")
    print(f"Sequences: {manifest.sequence_count}")
    print(f"Controls ({len(manifest.control_names)}): {', '.join(manifest.control_names)}")
    print(f"Train sessions: {len(manifest.train_sessions)}")
    print(f"Val sessions: {len(manifest.val_sessions)}")
    if result.skipped_sessions:
        print("Skipped sessions: " + ", ".join(result.skipped_sessions))


if __name__ == "__main__":
    main()
