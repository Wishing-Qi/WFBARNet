from __future__ import annotations

import unittest

import numpy as np

from src.court.opencv_court_detector import (
    CourtLineOverlayRenderer,
    CourtLinePrediction,
    OpenCVCourtLineDetector,
    draw_court_prediction,
)


class OpenCVCourtLineDetectorTest(unittest.TestCase):
    def test_blank_frame_returns_stable_payload(self) -> None:
        detector = OpenCVCourtLineDetector()

        result = detector.predict(np.zeros((120, 160, 3), dtype=np.uint8), frame_id=0, timestamp_ms=0)
        payload = result.to_dict()

        self.assertEqual(payload["frame_id"], 0)
        self.assertEqual(payload["timestamp_ms"], 0)
        self.assertEqual(payload["source_size"], [160, 120])
        self.assertTrue(payload["attempted"])
        self.assertFalse(payload["valid"])
        self.assertIn("court_to_image_h", payload)
        self.assertIn("image_to_court_h", payload)
        self.assertIn("projected_lines", payload)

    def test_cached_overlay_matches_direct_drawing(self) -> None:
        prediction = CourtLinePrediction(
            frame_id=1,
            timestamp_ms=40,
            source_size=(120, 80),
            valid=True,
            attempted=True,
            updated=True,
            update_type="reliable update",
            status="reliable update",
            confidence=0.9,
            candidate_confidence=0.9,
            reason="unit test",
            scheme="test",
            corners=[[20.0, 15.0], [100.0, 15.0], [100.0, 65.0], [20.0, 65.0]],
            keypoints=[],
            court_to_image_h=[],
            image_to_court_h=[],
            projected_lines={
                "doubles_outer": [[20.0, 15.0], [100.0, 15.0], [100.0, 65.0], [20.0, 65.0]],
                "center_line": [[60.0, 15.0], [60.0, 65.0]],
                "service_line": [[20.0, 40.0], [100.0, 40.0]],
            },
            metrics={},
            detect_ms=12.0,
            rejected_count=0,
        )
        frame = np.full((80, 120, 3), 64, dtype=np.uint8)

        direct = draw_court_prediction(frame, prediction)
        cached = CourtLineOverlayRenderer().draw(frame, prediction)

        self.assertLessEqual(np.abs(direct.astype(np.int16) - cached.astype(np.int16)).max(), 2)


if __name__ == "__main__":
    unittest.main()
