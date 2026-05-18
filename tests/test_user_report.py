from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.utils.user_report import (
    build_user_report_data_from_rally_record,
    export_user_report_html,
    render_user_report_html,
    sample_user_report_data,
)


class UserReportExporterTest(unittest.TestCase):
    def test_sample_report_renders_required_sections(self) -> None:
        report_data = sample_user_report_data(generated_at="2026-05-14 20:30")
        html = render_user_report_html(report_data)

        self.assertIn("<!doctype html>", html)
        self.assertIn("羽毛球水平分析报告", html)
        self.assertIn("能力维度图", html)
        self.assertIn("需要改进的点", html)
        self.assertIn("目标用户", html)
        self.assertIn("<svg", html)
        self.assertIn("后场回位", html)

    def test_build_report_from_rally_record_keeps_project_metrics(self) -> None:
        record = {
            "rally_name": "clip.mp4",
            "summary": {
                "duration_s": 30.0,
                "rally_hit_count": 10,
                "avg_hit_interval_ms": 1800.0,
                "hit_confidence_avg": 0.8,
                "out_of_frame_count": 0,
                "stroke_distribution": {"高远球": 4, "杀球": 2, "搓放": 1},
                "data_reliability": {
                    "ball_visible_rate": 0.9,
                    "pose_valid_rate": 0.86,
                    "court_valid_rate": 0.82,
                    "avg_ball_confidence": 0.78,
                },
                "players": {
                    "bottom": {
                        "label": "测试用户",
                        "distance_m": 40.0,
                        "avg_speed_mps": 1.1,
                        "max_speed_mps": 3.8,
                        "stop_count": 8,
                        "start_count": 9,
                        "hit_count": 10,
                        "zone_hits": {"front": 1, "mid": 3, "back": 6},
                        "passive_hit_count": 4,
                        "high_intensity_count": 5,
                        "max_continuous_m": 5.0,
                    },
                    "top": {
                        "label": "对手",
                        "distance_m": 36.0,
                        "avg_speed_mps": 1.0,
                        "max_speed_mps": 3.5,
                        "stop_count": 7,
                        "start_count": 8,
                        "passive_hit_count": 2,
                    },
                },
            },
            "details": {"hits": [{"timestamp_ms": 1200, "player": "bottom", "zone": "back", "stroke": "高远球", "confidence": 0.8}]},
        }

        report_data = build_user_report_data_from_rally_record(record, athlete_name="测试用户", generated_at="2026-05-14 20:30")

        self.assertEqual(report_data["meta"]["video_name"], "clip.mp4")
        self.assertEqual(report_data["summary"]["metrics"][1]["value"], "10 次")
        self.assertEqual(report_data["players"]["user"]["distance_m"], 40.0)
        self.assertGreaterEqual(len(report_data["improvements"]), 3)

    def test_export_writes_utf8_html_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "report.html"
            export_user_report_html(sample_user_report_data(generated_at="2026-05-14 20:30"), output_path)

            content = output_path.read_text(encoding="utf-8")
            self.assertIn("羽毛球水平分析报告", content)
            self.assertIn("数据可信度", content)


if __name__ == "__main__":
    unittest.main()
