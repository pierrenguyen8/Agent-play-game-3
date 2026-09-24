from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


TARGET_RELATIVE_PATHS = (
    "pixelbc/models/bc_model.py",
    "pixelbc/models/utils/model_utils.py",
)
TOKENS = (
    "LSTM",
    "button",
    "binary_cross_entropy",
    "BCE",
    "CrossEntropy",
    "sequence_length",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def symbols(path: Path) -> dict[str, list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes: list[str] = []
    functions: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
    return {
        "classes": sorted(set(classes)),
        "functions": sorted(set(functions)),
    }


def token_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    return {
        token: len(re.findall(re.escape(token), text, flags=re.IGNORECASE))
        for token in TOKENS
    }


def git_commit(checkout: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the exact fetched PixelBC source without importing its dependencies."
    )
    parser.add_argument("--checkout", type=Path, default=Path("external/pixelbc"))
    parser.add_argument("--output", type=Path, default=Path("external/pixelbc.audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkout = args.checkout.resolve()
    if not checkout.exists():
        raise SystemExit(
            f"PixelBC checkout not found: {checkout}\n"
            "Run tools\\vendor\\fetch_pixelbc.cmd first."
        )

    files = []
    for relative in TARGET_RELATIVE_PATHS:
        path = checkout / relative
        if not path.exists():
            raise SystemExit(f"Expected PixelBC source file not found: {path}")
        files.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "symbols": symbols(path),
                "token_counts": token_counts(path),
            }
        )

    report = {
        "repository": "https://github.com/microsoft/imitation_learning_in_modern_video_games.git",
        "commit": git_commit(checkout),
        "files": files,
        "integration_target": {
            "policy": "LSTM temporal policy",
            "action_representation": "binary held button vector",
            "planet_conditioning": "intent_id + skill_id remain in our thin adapter",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Audit written to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
