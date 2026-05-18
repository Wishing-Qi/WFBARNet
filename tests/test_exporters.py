from __future__ import annotations

import json
import unittest
from io import StringIO

from src.utils.exporters import frame_result_log_record, write_frame_log_jsonl
from src.utils.structures import FrameResult, PersonPoseResult, TrackResult


class FrameLogExporterTest(unittest.TestCase):
    def test_frame_log_records_ball_pose_and_hit_event(self) -> None:
        pose = PersonPoseResult(
            person_id=1,
            bbox=[10.0, 20.0, 70.0, 160.0],
            keypoints=[[12.0, 24.0], [30.0, 45.0]],
            scores=[0.91, 0.82],
            person_score=0.88,
        )
        result = FrameResult(
            frame_id=12,
            pose=[pose],
            track=TrackResult(ball_xy=[123.0, 45.0], visible=1, score=0.77),
        )

        record = frame_result_log_record(
            result,
            timestamp_ms=480,
            hit_event={
                "event_type": "hit",
                "frame_id": 11,
                "timestamp_ms": 440,
                "ball_xy": [120.0, 50.0],
                "rule": "vy_reversal",
                "confidence": 0.9,
                "all_rules": ["vy_reversal"],
            },
            trajectory_event={
                "event_type": "landing",
                "frame_id": 12,
                "timestamp_ms": 480,
                "ball_xy": [123.0, 45.0],
                "rule": "speed_step",
                "confidence": 0.9,
                "all_rules": ["speed_step"],
                "features": {"v_curr": 1.0},
            },
            landing_event={
                "event_type": "landing",
                "frame_id": 12,
                "timestamp_ms": 480,
                "ball_xy": [123.0, 45.0],
                "rule": "speed_step",
                "confidence": 0.9,
            },
        )

        self.assertEqual(record["frame_id"], 12)
        self.assertEqual(record["timestamp_ms"], 480)
        self.assertEqual(record["ball"], {"xy": [123.0, 45.0], "visible": 1, "score": 0.77})
        self.assertIsNone(record["court"])
        self.assertEqual(record["pose"][0]["person_id"], 1)
        self.assertEqual(record["pose"][0]["bbox"], [10.0, 20.0, 70.0, 160.0])
        self.assertEqual(record["pose"][0]["keypoints"], [[12.0, 24.0], [30.0, 45.0]])
        self.assertEqual(record["hit_event"]["ball_xy"], [120.0, 50.0])
        self.assertEqual(record["hit_event"]["rule"], "vy_reversal")
        self.assertEqual(record["trajectory_event"]["event_type"], "landing")
        self.assertEqual(record["trajectory_event"]["rule"], "speed_step")
        self.assertEqual(record["landing_event"]["ball_xy"], [123.0, 45.0])

    def test_frame_log_can_include_court_prediction_summary(self) -> None:
        result = FrameResult(
            frame_id=4,
            pose=[],
            track=TrackResult(ball_xy=[-1.0, -1.0], visible=0, score=0.1),
        )

        record = frame_result_log_record(
            result,
            timestamp_ms=160,
            court_prediction={
                "valid": True,
                "attempted": True,
                "updated": True,
                "update_type": "reliable update",
                "status": "reliable update",
                "confidence": 0.93,
                "candidate_confidence": 0.91,
                "reason": "YOLO segmentation mask",
                "scheme": "shuttlecourt_seg",
                "corners": [[10.0, 20.0], [80.0, 20.0], [90.0, 120.0], [8.0, 120.0]],
                "metrics": {"components": {"candidate_rank": 0.88, "seg_quality": 0.9}},
                "detect_ms": 15.5,
                "rejected_count": 0,
            },
        )

        self.assertTrue(record["court"]["valid"])
        self.assertEqual(record["court"]["scheme"], "shuttlecourt_seg")
        self.assertEqual(record["court"]["corners"][0], [10.0, 20.0])
        self.assertEqual(record["court"]["metrics"]["components"]["candidate_rank"], 0.88)

    def test_write_frame_log_jsonl_writes_one_json_object_per_line(self) -> None:
        buffer = StringIO()
        write_frame_log_jsonl(buffer, {"frame_id": 1, "ball": {"xy": [1.0, 2.0]}})

        rows = buffer.getvalue().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["frame_id"], 1)


if __name__ == "__main__":
    unittest.main()
