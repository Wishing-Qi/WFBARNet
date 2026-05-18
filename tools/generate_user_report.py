from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.user_report import export_user_report_html, sample_user_report_data


def main() -> int:
    output_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "outputs" / "user_reports" / "badminton_user_report_sample.html"
    )
    report_data = sample_user_report_data()
    export_user_report_html(report_data, output_path)
    print(f"已生成用户报告: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
