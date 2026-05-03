from __future__ import annotations

import unittest

import numpy as np

from src.utils.structures import FrameResult, PersonPoseResult, TrackResult
from src.utils.visualize import TrackTrailRenderer


def _frame_result(
    frame_id: int,
    x: float,
    y: float,
    pose: list[PersonPoseResult] | None = None,
    score: float = 0.85,
) -> FrameResult:
    track = TrackResult(ball_xy=[x, y], visible=1, score=score, heatmap_shape=[288, 512])
    return FrameResult(frame_id=frame_id, pose=list(pose or []), track=track)


def _missing_result(frame_id: int) -> FrameResult:
    track = TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.0, heatmap_shape=[288, 512])
    return FrameResult(frame_id=frame_id, pose=[], track=track)


def _person_bbox(x1: float, y1: float, x2: float, y2: float) -> PersonPoseResult:
    return PersonPoseResult(
        person_id=0,
        bbox=[x1, y1, x2, y2],
        keypoints=[],
        scores=[],
        person_score=0.9,
    )


def _person_with_right_arm(
    bbox: list[float],
    shoulder: tuple[float, float],
    elbow: tuple[float, float],
    wrist: tuple[float, float],
) -> PersonPoseResult:
    keypoints = [[0.0, 0.0] for _ in range(17)]
    scores = [0.0 for _ in range(17)]
    keypoints[6] = [shoulder[0], shoulder[1]]
    keypoints[8] = [elbow[0], elbow[1]]
    keypoints[10] = [wrist[0], wrist[1]]
    scores[6] = 0.9
    scores[8] = 0.9
    scores[10] = 0.9
    return PersonPoseResult(
        person_id=0,
        bbox=bbox,
        keypoints=keypoints,
        scores=scores,
        person_score=0.9,
    )


class TrackTrailRendererTest(unittest.TestCase):
    def test_large_trail_jump_is_not_connected_visually(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0, trail_break_threshold_px=80.0)
        frame = np.zeros((120, 240, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 20.0, 40.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 60.0, 40.0), timestamp_ms=40)
        canvas = renderer.draw_on(frame.copy(), _frame_result(2, 180.0, 40.0), timestamp_ms=80)

        self.assertTrue(np.any(canvas[40, 40] > 0))
        self.assertTrue(np.array_equal(canvas[40, 120], np.zeros(3, dtype=np.uint8)))

    def test_missing_track_segment_is_not_connected_visually(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0, trail_break_threshold_px=80.0)
        frame = np.zeros((120, 120, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 20.0, 40.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _missing_result(1), timestamp_ms=40)
        canvas = renderer.draw_on(frame.copy(), _frame_result(2, 60.0, 40.0), timestamp_ms=80)

        self.assertTrue(np.array_equal(canvas[40, 40], np.zeros(3, dtype=np.uint8)))

    def test_hit_marker_stays_solid_red_for_two_seconds_then_disappears(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 40.0, 80.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 70.0, 80.0), timestamp_ms=40)
        hit_canvas = renderer.draw_on(frame.copy(), _frame_result(2, 70.0, 45.0), timestamp_ms=80)
        hit_event = renderer.last_hit_event()
        before_expiry = renderer.draw_on(frame.copy(), _missing_result(3), timestamp_ms=2039)
        after_expiry = renderer.draw_on(frame.copy(), _missing_result(4), timestamp_ms=2040)

        red = np.array([0, 0, 255], dtype=np.uint8)
        self.assertIsNotNone(hit_event)
        self.assertEqual(hit_event["frame_id"], 1)
        self.assertEqual(hit_event["timestamp_ms"], 40)
        self.assertEqual(hit_event["ball_xy"], [70.0, 80.0])
        self.assertTrue(np.array_equal(hit_canvas[80, 70], red))
        self.assertFalse(np.array_equal(hit_canvas[45, 70], red))
        self.assertTrue(np.array_equal(before_expiry[80, 70], red))
        self.assertFalse(np.array_equal(after_expiry[80, 70], red))

    def test_does_not_mark_straight_fast_motion_as_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 40.0, 80.0))
        renderer.draw_on(frame.copy(), _frame_result(1, 70.0, 80.0))
        canvas = renderer.draw_on(frame.copy(), _frame_result(2, 100.0, 80.0))

        self.assertFalse(np.array_equal(canvas[80, 100], np.array([0, 0, 255], dtype=np.uint8)))

    def test_low_score_prediction_point_does_not_create_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 40.0, 80.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 70.0, 80.0), timestamp_ms=40)
        renderer.draw_on(frame.copy(), _frame_result(2, 70.0, 45.0, score=0.20), timestamp_ms=80)

        self.assertIsNone(renderer.last_hit_event())

    def test_abrupt_high_score_relock_creates_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=60.0)
        frame = np.zeros((760, 980, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 773.3, 673.3, score=0.20), timestamp_ms=22573)
        renderer.draw_on(frame.copy(), _frame_result(1, 852.6, 328.6, score=0.75), timestamp_ms=22589)
        renderer.draw_on(frame.copy(), _frame_result(2, 862.3, 286.6, score=0.74), timestamp_ms=22606)

        hit_event = renderer.last_hit_event()
        self.assertIsNotNone(hit_event)
        self.assertEqual(hit_event["frame_id"], 1)
        self.assertEqual(hit_event["ball_xy"], [852.6, 328.6])

    def test_large_abrupt_jump_creates_hit_without_ratio(self) -> None:
        renderer = TrackTrailRenderer(fps=60.0)
        frame = np.zeros((760, 980, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 100.0, 300.0, score=0.74), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 470.0, 300.0, score=0.75), timestamp_ms=16)
        renderer.draw_on(frame.copy(), _frame_result(2, 680.0, 300.0, score=0.74), timestamp_ms=33)

        hit_event = renderer.last_hit_event()
        self.assertIsNotNone(hit_event)
        self.assertEqual(hit_event["frame_id"], 1)
        self.assertEqual(hit_event["ball_xy"], [470.0, 300.0])

    def test_cross_segment_abrupt_relock_creates_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=60.0)
        frame = np.zeros((760, 980, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 747.9, 425.2, score=0.58), timestamp_ms=7791)
        renderer.draw_on(frame.copy(), _missing_result(1), timestamp_ms=7820)
        renderer.draw_on(frame.copy(), _frame_result(2, 799.8, 246.4, score=0.75), timestamp_ms=7841)
        renderer.draw_on(frame.copy(), _frame_result(3, 810.2, 202.5, score=0.74), timestamp_ms=7858)

        hit_event = renderer.last_hit_event()
        self.assertIsNotNone(hit_event)
        self.assertEqual(hit_event["frame_id"], 0)
        self.assertEqual(hit_event["ball_xy"], [747.9, 425.2])

    def test_cross_segment_large_jump_creates_hit_without_ratio(self) -> None:
        renderer = TrackTrailRenderer(fps=60.0)
        frame = np.zeros((760, 980, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 100.0, 300.0, score=0.58), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _missing_result(1), timestamp_ms=16)
        renderer.draw_on(frame.copy(), _frame_result(2, 380.0, 300.0, score=0.75), timestamp_ms=33)
        renderer.draw_on(frame.copy(), _frame_result(3, 607.0, 300.0, score=0.74), timestamp_ms=50)

        hit_event = renderer.last_hit_event()
        self.assertIsNotNone(hit_event)
        self.assertEqual(hit_event["frame_id"], 0)
        self.assertEqual(hit_event["ball_xy"], [100.0, 300.0])

    def test_cross_segment_abrupt_relock_does_not_duplicate_recent_corner_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=60.0)
        frame = np.zeros((760, 980, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 741.5, 415.9, score=0.67), timestamp_ms=7741)
        renderer.draw_on(frame.copy(), _frame_result(1, 740.1, 432.2, score=0.68), timestamp_ms=7758)
        renderer.draw_on(frame.copy(), _frame_result(2, 747.9, 425.2, score=0.58), timestamp_ms=7791)
        self.assertIsNotNone(renderer.last_hit_event())

        renderer.draw_on(frame.copy(), _missing_result(3), timestamp_ms=7820)
        renderer.draw_on(frame.copy(), _frame_result(4, 799.8, 246.4, score=0.75), timestamp_ms=7841)
        renderer.draw_on(frame.copy(), _frame_result(5, 810.2, 202.5, score=0.74), timestamp_ms=7858)

        self.assertIsNone(renderer.last_hit_event())

    def test_abrupt_hit_does_not_duplicate_recent_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=60.0)
        frame = np.zeros((760, 980, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 40.0, 80.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 70.0, 80.0), timestamp_ms=40)
        renderer.draw_on(frame.copy(), _frame_result(2, 70.0, 45.0), timestamp_ms=80)
        self.assertIsNotNone(renderer.last_hit_event())

        renderer.draw_on(frame.copy(), _missing_result(3), timestamp_ms=90)
        renderer.draw_on(frame.copy(), _frame_result(4, 773.3, 673.3, score=0.20), timestamp_ms=100)
        renderer.draw_on(frame.copy(), _frame_result(5, 852.6, 328.6, score=0.75), timestamp_ms=120)
        renderer.draw_on(frame.copy(), _frame_result(6, 862.3, 286.6, score=0.74), timestamp_ms=160)

        self.assertIsNone(renderer.last_hit_event())

    def test_does_not_mark_top_exit_reversal_as_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 80.0, 70.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 82.0, 12.0), timestamp_ms=40)
        renderer.draw_on(frame.copy(), _frame_result(2, 86.0, 72.0), timestamp_ms=80)

        self.assertIsNone(renderer.last_hit_event())

    def test_missing_track_breaks_hit_continuity(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 40.0, 80.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 70.0, 80.0), timestamp_ms=40)
        renderer.draw_on(frame.copy(), _missing_result(2), timestamp_ms=80)
        renderer.draw_on(frame.copy(), _frame_result(3, 70.0, 45.0), timestamp_ms=120)

        self.assertIsNone(renderer.last_hit_event())

    def test_person_occlusion_does_not_create_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)
        pose = [_person_bbox(30.0, 35.0, 95.0, 115.0)]

        renderer.draw_on(frame.copy(), _frame_result(0, 40.0, 80.0, pose), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 70.0, 80.0, pose), timestamp_ms=40)
        renderer.draw_on(frame.copy(), _frame_result(2, 70.0, 45.0, pose), timestamp_ms=80)

        self.assertIsNone(renderer.last_hit_event())

    def test_floor_bounce_does_not_create_hit_without_pose_assist(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((180, 220, 3), dtype=np.uint8)

        renderer.draw_on(frame.copy(), _frame_result(0, 90.0, 55.0), timestamp_ms=0)
        renderer.draw_on(frame.copy(), _frame_result(1, 90.0, 115.0), timestamp_ms=40)
        renderer.draw_on(frame.copy(), _frame_result(2, 94.0, 78.0), timestamp_ms=80)

        self.assertIsNone(renderer.last_hit_event())

    def test_strong_pose_assist_allows_low_real_hit_shape(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((180, 220, 3), dtype=np.uint8)

        renderer.draw_on(
            frame.copy(),
            _frame_result(0, 90.0, 55.0, [_person_with_right_arm([30.0, 35.0, 140.0, 145.0], (45.0, 80.0), (68.0, 66.0), (90.0, 55.0))]),
            timestamp_ms=0,
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(1, 90.0, 115.0, [_person_with_right_arm([30.0, 35.0, 140.0, 145.0], (45.0, 100.0), (68.0, 108.0), (90.0, 115.0))]),
            timestamp_ms=40,
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(2, 94.0, 78.0, [_person_with_right_arm([30.0, 35.0, 140.0, 145.0], (48.0, 96.0), (71.0, 86.0), (94.0, 78.0))]),
            timestamp_ms=80,
        )

        self.assertIsNotNone(renderer.last_hit_event())

    def test_nearby_wrist_motion_allows_low_hit_shape(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((220, 260, 3), dtype=np.uint8)

        renderer.draw_on(
            frame.copy(),
            _frame_result(0, 90.0, 55.0, [_person_with_right_arm([60.0, 40.0, 210.0, 190.0], (80.0, 85.0), (105.0, 95.0), (125.0, 105.0))]),
            timestamp_ms=0,
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(1, 90.0, 115.0, [_person_with_right_arm([60.0, 40.0, 210.0, 190.0], (90.0, 105.0), (125.0, 125.0), (168.0, 158.0))]),
            timestamp_ms=40,
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(2, 96.0, 78.0, [_person_with_right_arm([60.0, 40.0, 210.0, 190.0], (92.0, 98.0), (128.0, 112.0), (158.0, 126.0))]),
            timestamp_ms=80,
        )

        self.assertIsNotNone(renderer.last_hit_event())

    def test_pose_assist_speed_change_without_turn_does_not_create_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=60.0)
        frame = np.zeros((700, 1200, 3), dtype=np.uint8)

        # Nearly straight trajectory with slight speed change.
        # prev -> mid: 30px in 16ms = 1875 px/s
        # mid -> current: 15px in 16ms = 937 px/s
        # speed_change = 2.0 (passes 1.25), turn_deg ~ 5 (below 20).
        person = _person_with_right_arm(
            [880.0, 340.0, 1050.0, 600.0],
            (920.0, 400.0),
            (950.0, 410.0),
            (985.0, 420.0),
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(0, 950.0, 400.0, [person]),
            timestamp_ms=0,
        )
        person_mid = _person_with_right_arm(
            [880.0, 340.0, 1050.0, 600.0],
            (920.0, 400.0),
            (955.0, 415.0),
            (980.0, 425.0),
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(1, 980.0, 403.0, [person_mid]),
            timestamp_ms=16,
        )
        person_cur = _person_with_right_arm(
            [880.0, 340.0, 1050.0, 600.0],
            (920.0, 400.0),
            (960.0, 420.0),
            (995.0, 410.0),
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(2, 995.0, 407.0, [person_cur]),
            timestamp_ms=32,
        )

        self.assertIsNone(renderer.last_hit_event())

    def test_pose_assist_recovers_relaxed_turn_hit(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)

        renderer.draw_on(
            frame.copy(),
            _frame_result(0, 40.0, 80.0, [_person_with_right_arm([0.0, 20.0, 120.0, 130.0], (10.0, 80.0), (25.0, 80.0), (40.0, 80.0))]),
            timestamp_ms=0,
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(1, 70.0, 80.0, [_person_with_right_arm([0.0, 20.0, 120.0, 130.0], (20.0, 80.0), (45.0, 80.0), (70.0, 80.0))]),
            timestamp_ms=40,
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(2, 87.0, 50.0, [_person_with_right_arm([0.0, 20.0, 120.0, 130.0], (30.0, 80.0), (58.0, 65.0), (87.0, 50.0))]),
            timestamp_ms=80,
        )

        hit_event = renderer.last_hit_event()
        self.assertIsNotNone(hit_event)
        self.assertEqual(hit_event["frame_id"], 1)

    def test_strong_pose_assist_allows_real_hit_during_occlusion(self) -> None:
        renderer = TrackTrailRenderer(fps=25.0)
        frame = np.zeros((160, 180, 3), dtype=np.uint8)

        renderer.draw_on(
            frame.copy(),
            _frame_result(0, 40.0, 80.0, [_person_with_right_arm([20.0, 30.0, 100.0, 120.0], (10.0, 80.0), (25.0, 80.0), (40.0, 80.0))]),
            timestamp_ms=0,
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(1, 70.0, 80.0, [_person_with_right_arm([20.0, 30.0, 100.0, 120.0], (20.0, 80.0), (45.0, 80.0), (70.0, 80.0))]),
            timestamp_ms=40,
        )
        renderer.draw_on(
            frame.copy(),
            _frame_result(2, 70.0, 45.0, [_person_with_right_arm([20.0, 30.0, 100.0, 120.0], (25.0, 78.0), (48.0, 62.0), (70.0, 45.0))]),
            timestamp_ms=80,
        )

        self.assertIsNotNone(renderer.last_hit_event())


if __name__ == "__main__":
    unittest.main()
