from __future__ import annotations

import unittest

import numpy as np

from apps.pyqt6.views.heatmap_renderer import HeatmapRenderer, HeatmapRenderConfig


class HeatmapRendererTest(unittest.TestCase):
    def test_empty_points_return_transparent_rgba(self) -> None:
        renderer = HeatmapRenderer(80, 120)
        rgba = renderer.build_heatmap_rgba([], color_mode="blue")

        self.assertEqual(rgba.shape, (120, 80, 4))
        self.assertEqual(int(rgba[..., 3].max()), 0)

    def test_filters_bad_points_and_builds_colored_alpha(self) -> None:
        renderer = HeatmapRenderer(
            80,
            120,
            config=HeatmapRenderConfig(sigma=4, show_contours=False, heatmap_opacity=1.0),
        )
        rgba = renderer.build_heatmap_rgba(
            [
                (305.0, 670.0),
                (np.nan, 400.0),
                None,
                (900.0, 2000.0),
            ],
            color_mode="red",
        )

        self.assertGreater(int(rgba[..., 3].max()), 0)
        self.assertGreater(int(rgba[..., 0].max()), int(rgba[..., 2].max()))

    def test_contours_add_visible_light_lines(self) -> None:
        renderer = HeatmapRenderer(
            96,
            160,
            config=HeatmapRenderConfig(sigma=7, contour_levels=5, contour_alpha=120, heatmap_opacity=0.9),
        )
        rgba = renderer.build_heatmap_rgba(
            [(300.0 + offset, 650.0) for offset in range(-45, 46, 15)],
            color_mode="blue",
            show_contours=True,
        )

        light_pixels = (
            (rgba[..., 0] > 170)
            & (rgba[..., 1] > 170)
            & (rgba[..., 2] > 170)
            & (rgba[..., 3] > 0)
        )
        self.assertTrue(bool(light_pixels.any()))


if __name__ == "__main__":
    unittest.main()
