from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import cv2
import numpy as np


PointLike = Sequence[float] | np.ndarray


@dataclass
class HeatmapRenderConfig:
    heatmap_width: int = 320
    heatmap_height: int = 700
    sigma: float = 22.0
    alpha_power: float = 0.75
    min_alpha_threshold: float = 0.035
    max_alpha: int = 220
    contour_levels: int | Sequence[float] = 7
    contour_alpha: int = 85
    contour_thickness: int = 1
    top_color_mode: str = "blue"
    bottom_color_mode: str = "red"
    show_contours: bool = True
    heatmap_opacity: float = 0.72
    contour_level_values: Sequence[float] = field(default_factory=lambda: tuple(np.linspace(0.15, 0.85, 7)))


class HeatmapRenderer:
    """Generate transparent player-position heatmaps from standard court coordinates."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        court_width: float = 610.0,
        court_height: float = 1340.0,
        config: HeatmapRenderConfig | None = None,
    ) -> None:
        self.config = config or HeatmapRenderConfig()
        self.court_width = max(1.0, float(court_width))
        self.court_height = max(1.0, float(court_height))
        self.set_size(width, height)

    def set_size(self, width: int, height: int) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.config.heatmap_width = self.width
        self.config.heatmap_height = self.height

    def build_density(self, points: Iterable[PointLike] | None, *, normalized: bool = False) -> np.ndarray:
        density = np.zeros((self.height, self.width), dtype=np.float32)
        clean_points = self._clean_points(points, normalized=normalized)
        if clean_points.size == 0:
            return density

        xs = np.clip(np.rint(clean_points[:, 0]).astype(np.int32), 0, self.width - 1)
        ys = np.clip(np.rint(clean_points[:, 1]).astype(np.int32), 0, self.height - 1)
        np.add.at(density, (ys, xs), 1.0)
        return density

    def build_heatmap_rgba(
        self,
        points: Iterable[PointLike] | None,
        *,
        color_mode: str = "blue",
        show_contours: bool | None = None,
        normalized: bool = False,
        opacity: float | None = None,
    ) -> np.ndarray:
        density = self.build_density(points, normalized=normalized)
        blurred = self._blur_density(density)
        norm = self._normalize_density(blurred)
        layer_opacity = self.config.heatmap_opacity if opacity is None else float(opacity)
        rgba = self.density_to_rgba(norm, color_mode=color_mode, opacity=layer_opacity)
        if self.config.show_contours if show_contours is None else bool(show_contours):
            self.draw_density_contours(rgba, norm)
        return rgba

    def build_heatmap_pixmap(
        self,
        points: Iterable[PointLike] | None,
        *,
        color_mode: str = "blue",
        show_contours: bool | None = None,
        normalized: bool = False,
        opacity: float | None = None,
    ):
        rgba = self.build_heatmap_rgba(
            points,
            color_mode=color_mode,
            show_contours=show_contours,
            normalized=normalized,
            opacity=opacity,
        )
        return self.rgba_to_pixmap(rgba)

    def density_to_rgba(self, density: np.ndarray, *, color_mode: str, opacity: float) -> np.ndarray:
        norm = np.asarray(density, dtype=np.float32)
        rgba = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        if norm.size == 0 or float(norm.max(initial=0.0)) <= 0.0:
            return rgba

        color = self._custom_color_map(norm, color_mode=color_mode)
        alpha_src = np.clip(
            (norm - self.config.min_alpha_threshold) / max(1e-6, 1.0 - self.config.min_alpha_threshold),
            0.0,
            1.0,
        )
        alpha = np.power(alpha_src, max(0.05, float(self.config.alpha_power)))
        alpha *= max(0.0, min(float(opacity), 1.0))
        alpha *= max(0, min(int(self.config.max_alpha), 255))

        rgba[..., :3] = color
        rgba[..., 3] = np.rint(alpha).astype(np.uint8)
        return rgba

    def draw_density_contours(self, rgba: np.ndarray, density: np.ndarray) -> None:
        norm = np.asarray(density, dtype=np.float32)
        if norm.size == 0 or float(norm.max(initial=0.0)) <= 0.0:
            return

        overlay = np.zeros_like(rgba, dtype=np.uint8)
        contour_alpha = max(0, min(int(self.config.contour_alpha), 255))
        thickness = max(1, int(self.config.contour_thickness))
        for level in self._contour_levels():
            mask = (norm >= float(level)).astype(np.uint8) * 255
            contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(
                    overlay,
                    contours,
                    -1,
                    (238, 242, 248, contour_alpha),
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )
        rgba[:] = _alpha_composite_rgba(rgba, overlay)

    def rgba_to_pixmap(self, rgba: np.ndarray):
        from PyQt6.QtGui import QImage, QPixmap

        rgba = np.ascontiguousarray(rgba, dtype=np.uint8)
        height, width = rgba.shape[:2]
        image = QImage(
            rgba.data,
            width,
            height,
            rgba.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(image)

    def _blur_density(self, density: np.ndarray) -> np.ndarray:
        sigma = max(0.0, float(self.config.sigma))
        if sigma <= 0.0:
            return density.astype(np.float32, copy=True)
        return cv2.GaussianBlur(density, (0, 0), sigmaX=sigma, sigmaY=sigma).astype(np.float32)

    def _normalize_density(self, density: np.ndarray) -> np.ndarray:
        max_value = float(np.max(density)) if density.size else 0.0
        if max_value <= 1e-12:
            return np.zeros_like(density, dtype=np.float32)
        return np.clip(density / max_value, 0.0, 1.0).astype(np.float32)

    def _contour_levels(self) -> list[float]:
        levels = self.config.contour_levels
        if isinstance(levels, int):
            return [float(value) for value in np.linspace(0.15, 0.85, max(1, levels))]
        return [float(value) for value in levels]

    def _clean_points(self, points: Iterable[PointLike] | None, *, normalized: bool) -> np.ndarray:
        if points is None:
            return np.empty((0, 2), dtype=np.float32)

        clean: list[tuple[float, float]] = []
        for point in points:
            if point is None:
                continue
            try:
                if len(point) < 2:  # type: ignore[arg-type]
                    continue
            except TypeError:
                continue
            try:
                x = float(point[0])  # type: ignore[index]
                y = float(point[1])  # type: ignore[index]
            except (TypeError, ValueError, IndexError):
                continue
            if not np.isfinite([x, y]).all():
                continue
            if normalized:
                x = np.clip(x, 0.0, 1.0) * (self.width - 1)
                y = np.clip(y, 0.0, 1.0) * (self.height - 1)
            else:
                x = np.clip(x, 0.0, self.court_width) / self.court_width * (self.width - 1)
                y = np.clip(y, 0.0, self.court_height) / self.court_height * (self.height - 1)
            clean.append((float(x), float(y)))
        if not clean:
            return np.empty((0, 2), dtype=np.float32)
        return np.asarray(clean, dtype=np.float32)

    def _custom_color_map(self, norm: np.ndarray, *, color_mode: str) -> np.ndarray:
        value = np.clip(norm[..., None], 0.0, 1.0)
        mode = color_mode.lower().strip()
        if mode == "red":
            low = np.array([255, 150, 70], dtype=np.float32)
            high = np.array([155, 12, 24], dtype=np.float32)
        elif mode == "blue":
            low = np.array([96, 205, 255], dtype=np.float32)
            high = np.array([0, 46, 180], dtype=np.float32)
        else:
            colored = cv2.applyColorMap(np.rint(norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
            return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        gamma = np.power(value, 0.85)
        return np.rint(low * (1.0 - gamma) + high * gamma).astype(np.uint8)


def _alpha_composite_rgba(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    overlay_f = overlay.astype(np.float32) / 255.0
    base_a = base_f[..., 3:4]
    overlay_a = overlay_f[..., 3:4]
    out_a = overlay_a + base_a * (1.0 - overlay_a)
    out_rgb = np.zeros_like(base_f[..., :3])
    valid = out_a[..., 0] > 1e-6
    out_rgb[valid] = (
        overlay_f[..., :3][valid] * overlay_a[valid]
        + base_f[..., :3][valid] * base_a[valid] * (1.0 - overlay_a[valid])
    ) / out_a[valid]
    out = np.concatenate((out_rgb, out_a), axis=-1)
    return np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
