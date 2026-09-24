from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LetterboxGeometry:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    offset_x: int
    offset_y: int


def calculate_letterbox(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> LetterboxGeometry:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image dimensions must be > 0")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    return LetterboxGeometry(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def letterbox(image, *, width: int, height: int, is_mask: bool = False):
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Segmentation preprocessing requires Pillow") from exc

    geometry = calculate_letterbox(image.width, image.height, width, height)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    resized = image.resize(
        (geometry.resized_width, geometry.resized_height),
        resample=resample,
    )
    mode = "L" if is_mask else "RGB"
    canvas = Image.new(mode, (width, height), color=0)
    canvas.paste(resized, (geometry.offset_x, geometry.offset_y))
    return canvas


def restore_letterbox_array(
    array,
    *,
    source_width: int,
    source_height: int,
):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Restoring segmentation masks requires numpy and opencv-python"
        ) from exc

    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected HxW probability array")

    target_height, target_width = values.shape
    geometry = calculate_letterbox(
        source_width, source_height, target_width, target_height
    )
    y1 = geometry.offset_y
    y2 = y1 + geometry.resized_height
    x1 = geometry.offset_x
    x2 = x1 + geometry.resized_width
    cropped = values[y1:y2, x1:x2]
    restored = cv2.resize(
        cropped,
        (source_width, source_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return np.asarray(restored, dtype=np.float32)


def image_to_tensor(image):
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Segmentation preprocessing requires numpy and torch"
        ) from exc

    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected RGB image")
    array = np.ascontiguousarray(array.transpose(2, 0, 1))
    return torch.from_numpy(array)
