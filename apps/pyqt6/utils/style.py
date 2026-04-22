from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication


STYLE_DIR = Path(__file__).resolve().parents[1] / "resources" / "styles"

# 片段加载顺序（越靠后优先级越高，可覆盖前面的规则）
_FRAGMENT_ORDER = [
    "tokens.qss",
    "base.qss",
    "layout.qss",
    "buttons.qss",
    "status.qss",
    "tabs.qss",
    "table.qss",
    "scrollbar.qss",
    "video_player.qss",
    "form.qss",
]


def discover_themes() -> list[Path]:
    """返回 styles/ 下所有包含 .qss 文件的子文件夹，按名称排序。"""
    if not STYLE_DIR.exists():
        return []
    return sorted(
        d for d in STYLE_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.qss"))
    )


def _merge_theme(theme_dir: Path) -> str:
    """按 _FRAGMENT_ORDER 顺序合并主题文件夹内的 .qss 片段。"""
    parts: list[str] = []
    # 先按预定顺序加载已知片段
    for name in _FRAGMENT_ORDER:
        frag = theme_dir / name
        if frag.exists():
            parts.append(f"/* --- {name} --- */\n" + frag.read_text(encoding="utf-8"))
    # 再追加文件夹内其他未列出的 .qss（扩展性）
    known = set(_FRAGMENT_ORDER)
    for frag in sorted(theme_dir.glob("*.qss")):
        if frag.name not in known:
            parts.append(f"/* --- {frag.name} --- */\n" + frag.read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def apply_theme(app: QApplication, theme_dir: Path) -> None:
    """合并并应用指定主题文件夹的所有 .qss 片段。"""
    if not theme_dir.exists():
        return
    app.setStyleSheet(_merge_theme(theme_dir))


def load_stylesheet(app: QApplication, theme_name: str = "office_light") -> Path | None:
    """初始化字体、Fusion 风格，并加载默认主题。返回已加载的主题目录。"""
    app.setStyle("Fusion")

    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    themes = discover_themes()
    selected = next((t for t in themes if t.name == theme_name), None)
    if selected is None and themes:
        selected = themes[0]
    if selected is not None:
        apply_theme(app, selected)
    return selected


# ---------------------------------------------------------------------------
# 向后兼容别名（旧代码通过 apply_stylesheet / discover_stylesheets 调用）
# ---------------------------------------------------------------------------

def discover_stylesheets() -> list[Path]:
    """兼容旧接口：返回主题目录列表（原来返回 .qss 文件列表）。"""
    return discover_themes()


def apply_stylesheet(app: QApplication, theme_path: Path) -> None:
    """兼容旧接口：theme_path 可以是目录（新）或 .qss 文件（旧）。"""
    if theme_path.is_dir():
        apply_theme(app, theme_path)
    else:
        app.setStyleSheet(theme_path.read_text(encoding="utf-8"))
