from __future__ import annotations

from dataclasses import dataclass

from .types import PlatformSurface


@dataclass(frozen=True, slots=True)
class PlatformExtractorConfig:
    threshold: float = 0.5
    min_width: int = 40
    min_area: int = 120
    close_kernel_width: int = 9
    close_kernel_height: int = 3


class PlatformSurfaceExtractor:
    """Convert a platform probability map into simple horizontal surfaces.

    This preserves the proven representation from the previous Planet runtime:
    threshold -> morphology close -> connected components -> line fit.
    """

    def __init__(self, config: PlatformExtractorConfig | None = None) -> None:
        self.config = config or PlatformExtractorConfig()

    def extract(self, probability_map) -> tuple[PlatformSurface, ...]:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "Platform extraction requires numpy and opencv-python."
            ) from exc

        probability = np.asarray(probability_map, dtype=np.float32)
        if probability.ndim != 2:
            raise ValueError(
                f"Expected a 2D probability map, got shape {probability.shape}."
            )

        cfg = self.config
        binary = (probability >= cfg.threshold).astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (cfg.close_kernel_width, cfg.close_kernel_height),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        height, width = probability.shape
        surfaces: list[PlatformSurface] = []

        for component_id in range(1, count):
            component_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if component_width < cfg.min_width or area < cfg.min_area:
                continue

            ys, xs = np.where(labels == component_id)
            if xs.size < 2:
                continue

            min_x = int(xs.min())
            max_x = int(xs.max())
            if max_x <= min_x:
                continue

            # Least-squares line is stable for the mostly-horizontal Planet
            # platform masks and matches the old runtime's compact surface form.
            slope, intercept = np.polyfit(xs.astype(np.float64), ys.astype(np.float64), 1)
            y1 = int(round(slope * min_x + intercept))
            y2 = int(round(slope * max_x + intercept))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            min_x = max(0, min(width - 1, min_x))
            max_x = max(0, min(width - 1, max_x))

            mask = labels == component_id
            confidence = float(probability[mask].mean())
            surfaces.append(
                PlatformSurface(
                    points=((float(min_x), float(y1)), (float(max_x), float(y2))),
                    confidence=confidence,
                )
            )

        surfaces.sort(
            key=lambda surface: (
                (surface.points[0][1] + surface.points[-1][1]) / 2.0,
                surface.points[0][0],
            )
        )
        return tuple(surfaces)
