from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

QApplication.setHighDpiScaleFactorRoundingPolicy(
    Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
)

from apps.pyqt6.utils.style import load_stylesheet


def main() -> int:
    from apps.pyqt6.controllers.analysis_controller_refined import MainController
    from apps.pyqt6.views.main_window_refined import MainWindow

    app = QApplication(sys.argv)

    app.setEffectEnabled(Qt.UIEffect.UI_AnimateCombo, False)
    app.setEffectEnabled(Qt.UIEffect.UI_AnimateTooltip, False)

    load_stylesheet(app, "office_light")

    window = MainWindow()
    controller = MainController(window)

    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
