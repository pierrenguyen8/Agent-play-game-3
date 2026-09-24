from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Literal

from .sequence_manifest import (
    BCDatasetManifest,
    BCFrameRecord,
    BCSequenceWindow,
    read_jsonl,
)


SplitName = Literal["all", "train", "val"]


class BCSequenceIndex:
    """Lightweight metadata index; visual frames are never loaded here."""

    def __init__(self, dataset_dir: str | Path, *, split: SplitName = "all") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest = BCDatasetManifest.read(self.dataset_dir / "manifest.json")
        self.frames = tuple(
            BCFrameRecord.from_dict(row)
            for row in read_jsonl(self.dataset_dir / "frames.jsonl")
        )
        windows = tuple(
            BCSequenceWindow.from_dict(row)
            for row in read_jsonl(self.dataset_dir / "windows.jsonl")
        )
        if len(self.frames) != self.manifest.frame_count:
            raise ValueError("frames.jsonl count does not match manifest")

        if split == "all":
            allowed = set(self.manifest.sessions)
        elif split == "train":
            allowed = set(self.manifest.train_sessions)
        elif split == "val":
            allowed = set(self.manifest.val_sessions)
        else:
            raise ValueError(f"Unknown split: {split}")
        self.windows = tuple(window for window in windows if window.session_id in allowed)

        for window in self.windows:
            stop = window.start_row + window.length
            if window.start_row < 0 or stop > len(self.frames):
                raise ValueError(f"Sequence window out of bounds: {window}")
            rows = self.frames[window.start_row:stop]
            if any(row.session_id != window.session_id for row in rows):
                raise ValueError(f"Sequence window crosses session boundary: {window}")

    def __len__(self) -> int:
        return len(self.windows)

    def sequence(self, index: int) -> tuple[BCFrameRecord, ...]:
        window = self.windows[index]
        return self.frames[window.start_row:window.start_row + window.length]


class TorchBCSequenceDataset:
    """PyTorch-facing PNG/video loader with no torchvision dependency.

    The metadata contract mirrors PixelBC's sequence-oriented dataset while
    retaining Planet recorder references. PNGs are read directly. Video-chunk
    rows are decoded lazily from the original chunk and never expanded to a PNG
    cache during export.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: SplitName = "train",
        image_width: int = 160,
        image_height: int = 90,
        max_open_videos: int = 4,
    ) -> None:
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image dimensions must be > 0")
        if max_open_videos <= 0:
            raise ValueError("max_open_videos must be > 0")
        try:
            import torch  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TorchBCSequenceDataset requires the project ML/perception extras"
            ) from exc
        self.index = BCSequenceIndex(dataset_dir, split=split)
        self.image_width = image_width
        self.image_height = image_height
        self.max_open_videos = max_open_videos
        self._video_captures: OrderedDict[str, Any] = OrderedDict()
        self._video_next_index: dict[str, int] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        # cv2.VideoCapture objects are not safe to pickle into DataLoader workers.
        state["_video_captures"] = OrderedDict()
        state["_video_next_index"] = {}
        return state

    def __len__(self) -> int:
        return len(self.index)

    def close(self) -> None:
        for capture in self._video_captures.values():
            capture.release()
        self._video_captures.clear()
        self._video_next_index.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _video_capture(self, path: str):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Video-backed BC sessions require opencv-python "
                "(installed by the perception extra)"
            ) from exc

        capture = self._video_captures.get(path)
        if capture is not None:
            self._video_captures.move_to_end(path)
            return capture

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open BC video chunk: {path}")
        self._video_captures[path] = capture
        self._video_next_index[path] = 0

        while len(self._video_captures) > self.max_open_videos:
            old_path, old_capture = self._video_captures.popitem(last=False)
            old_capture.release()
            self._video_next_index.pop(old_path, None)

        return capture

    def _read_video_rgb(self, row: BCFrameRecord):
        import cv2

        assert row.visual_frame_index is not None
        path = row.frame_path
        capture = self._video_capture(path)
        desired = int(row.visual_frame_index)
        if self._video_next_index.get(path) != desired:
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, desired):
                raise RuntimeError(
                    f"Cannot seek video chunk {path} to frame {desired}"
                )
        success, bgr = capture.read()
        if not success or bgr is None:
            raise RuntimeError(
                f"Cannot decode video chunk {path} frame {desired}"
            )
        self._video_next_index[path] = desired + 1
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _read_resized_rgb(self, row: BCFrameRecord):
        import numpy as np
        from PIL import Image

        if row.visual_storage == "image":
            with Image.open(row.frame_path) as image:
                image = image.convert("RGB").resize(
                    (self.image_width, self.image_height),
                    resample=Image.Resampling.BILINEAR,
                )
                return np.asarray(image, dtype=np.float32) / 255.0

        rgb = self._read_video_rgb(row)
        image = Image.fromarray(rgb, mode="RGB").resize(
            (self.image_width, self.image_height),
            resample=Image.Resampling.BILINEAR,
        )
        return np.asarray(image, dtype=np.float32) / 255.0

    def __getitem__(self, index: int):
        import torch

        rows = self.index.sequence(index)
        frame_tensors = []
        for row in rows:
            array = self._read_resized_rgb(row)
            frame_tensors.append(torch.from_numpy(array.copy()).permute(2, 0, 1))

        return {
            "frames": torch.stack(frame_tensors, dim=0),
            "held": torch.tensor(
                [[float(value) for value in row.held] for row in rows],
                dtype=torch.float32,
            ),
            "intent_id": torch.tensor([row.intent_id for row in rows], dtype=torch.long),
            "skill_id": torch.tensor([row.skill_id for row in rows], dtype=torch.long),
            "condition_mask": torch.tensor(
                [row.has_conditioning for row in rows],
                dtype=torch.bool,
            ),
            "session_id": rows[0].session_id,
            "frame_index": torch.tensor([row.frame_index for row in rows], dtype=torch.long),
        }
