# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
LOGO_PATH = ROOT / "tools" / "hylogo.png"
OUTPUT_BASE = ROOT / "outputs" / "ppt_style_mockups"

W, H = 1600, 900


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    return (*hex_to_rgb(value), alpha)


def blend(a: str, b: str, t: float) -> tuple[int, int, int]:
    ar, ag, ab = hex_to_rgb(a)
    br, bg, bb = hex_to_rgb(b)
    return (
        int(ar + (br - ar) * t),
        int(ag + (bg - ag) * t),
        int(ab + (bb - ab) * t),
    )


def unique_output_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for idx in range(2, 100):
        candidate = base.with_name(f"{base.name}_v{idx}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Too many existing output directories.")


def font_path(*names: str) -> str:
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return str(path)
    return str(Path("C:/Windows/Fonts/msyh.ttc"))


FONT_SANS = font_path("NotoSansSC-VF.ttf", "msyh.ttc", "simhei.ttf")
FONT_SANS_BOLD = font_path("msyhbd.ttc", "NotoSansSC-VF.ttf", "simhei.ttf")
FONT_SERIF = font_path("NotoSerifSC-VF.ttf", "simsun.ttc", "msyh.ttc")
FONT_DENG = font_path("Deng.ttf", "msyh.ttc", "NotoSansSC-VF.ttf")


def fnt(size: int, kind: str = "sans") -> ImageFont.FreeTypeFont:
    if kind == "serif":
        return ImageFont.truetype(FONT_SERIF, size)
    if kind == "bold":
        return ImageFont.truetype(FONT_SANS_BOLD, size)
    if kind == "deng":
        return ImageFont.truetype(FONT_DENG, size)
    return ImageFont.truetype(FONT_SANS, size)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if not text:
        return 0, 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str | tuple[int, int, int, int],
    anchor: str | None = None,
    align: str = "left",
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor, align=align)


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int | None = None,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in text:
        if char == "\n":
            lines.append(current)
            current = ""
            continue
        trial = current + char
        if text_size(draw, trial, font)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = char
        if max_lines is not None and len(lines) >= max_lines:
            return lines[:max_lines]
    if current:
        lines.append(current)
    if max_lines is not None:
        return lines[:max_lines]
    return lines


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill, outline=None, width: int = 1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def alpha_layer(base: Image.Image, fn) -> None:
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fn(draw)
    base.alpha_composite(layer)


def paste_shadowed(base: Image.Image, item: Image.Image, xy: tuple[int, int], shadow=(0, 0, 0, 70), blur=18, offset=(0, 12)):
    shadow_img = Image.new("RGBA", item.size, (0, 0, 0, 0))
    alpha = item.split()[-1]
    shadow_img.putalpha(alpha)
    shadow_colored = Image.new("RGBA", item.size, shadow)
    shadow_colored.putalpha(alpha)
    shadow_colored = shadow_colored.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(shadow_colored, (xy[0] + offset[0], xy[1] + offset[1]))
    base.alpha_composite(item, xy)


def make_logo_transparent(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    pix = img.load()
    w, h = img.size
    seen = [[False] * h for _ in range(w)]
    q: deque[tuple[int, int]] = deque()
    for x in range(w):
        q.append((x, 0))
        q.append((x, h - 1))
    for y in range(h):
        q.append((0, y))
        q.append((w - 1, y))

    def is_bg(x: int, y: int) -> bool:
        r, g, b, a = pix[x, y]
        if a < 8:
            return True
        return r > 238 and g > 238 and b > 238

    while q:
        x, y = q.popleft()
        if x < 0 or y < 0 or x >= w or y >= h or seen[x][y] or not is_bg(x, y):
            continue
        seen[x][y] = True
        q.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    for x in range(w):
        for y in range(h):
            if seen[x][y]:
                r, g, b, _ = pix[x, y]
                pix[x, y] = (r, g, b, 0)
    return img


LOGO = make_logo_transparent(LOGO_PATH)


@dataclass(frozen=True)
class Style:
    key: str
    name: str
    brief: str
    bg: str
    bg2: str
    primary: str
    secondary: str
    accent: str
    accent2: str
    text: str
    muted: str
    panel: str
    line: str
    title_font: str
    dark: bool = False
    radius: int = 12


STYLES = [
    Style(
        key="01_stable_academic",
        name="稳重学术型",
        brief="深蓝主轴、青紫点缀、严谨分栏，适合正式答辩。",
        bg="#F6F8FC",
        bg2="#EAF2FA",
        primary="#132B5E",
        secondary="#2E7BC2",
        accent="#46C8E9",
        accent2="#8A55E6",
        text="#152238",
        muted="#64748B",
        panel="#FFFFFF",
        line="#C8D6EA",
        title_font="serif",
        radius=8,
    ),
    Style(
        key="02_modern_minimal",
        name="现代简约型",
        brief="大留白、细线框、轻量色块，表达清爽和易读。",
        bg="#FFFFFF",
        bg2="#F2F7FB",
        primary="#2348D8",
        secondary="#20BFD9",
        accent="#DA4FB6",
        accent2="#7B5CEB",
        text="#172033",
        muted="#667085",
        panel="#F8FBFE",
        line="#D8E2F0",
        title_font="sans",
        radius=10,
    ),
    Style(
        key="03_tech_rational",
        name="科技理性型",
        brief="深色画布、网格线和冷色高亮，强调算法系统感。",
        bg="#08111F",
        bg2="#111B31",
        primary="#74E3FF",
        secondary="#7A66F0",
        accent="#FF65C8",
        accent2="#FFD5B5",
        text="#EEF6FF",
        muted="#9CB3CC",
        panel="#101B2D",
        line="#29405D",
        title_font="sans",
        dark=True,
        radius=8,
    ),
    Style(
        key="04_fresh_premium",
        name="高级清爽型",
        brief="浅青底色、蓝紫线条、柔和卡片，正式但不沉重。",
        bg="#F4FBFC",
        bg2="#E9F8FB",
        primary="#1F6FB6",
        secondary="#46C6E6",
        accent="#9B5DE5",
        accent2="#F2A7D8",
        text="#223145",
        muted="#617184",
        panel="#FFFFFF",
        line="#CFE9F0",
        title_font="sans",
        radius=14,
    ),
]


PAGES = [
    ("cover", "封面页"),
    ("agenda", "目录页"),
    ("chapter", "章节页"),
    ("framework", "研究框架页"),
    ("results", "实验结果页"),
    ("summary", "总结页"),
]


def gradient_bg(style: Style) -> Image.Image:
    img = Image.new("RGBA", (W, H), rgba(style.bg))
    px = img.load()
    for y in range(H):
        t = y / max(H - 1, 1)
        col = blend(style.bg, style.bg2, t)
        for x in range(W):
            px[x, y] = (*col, 255)
    return img


def decorate_background(img: Image.Image, style: Style, page: str) -> None:
    draw = ImageDraw.Draw(img)
    if style.key == "01_stable_academic":
        draw.rectangle((0, 0, 104, H), fill=rgba(style.primary))
        draw.rectangle((104, 0, 112, H), fill=rgba(style.accent))
        draw.line((170, 88, 1480, 88), fill=rgba(style.line), width=2)
        draw.line((170, 815, 1480, 815), fill=rgba(style.line), width=1)
        alpha_layer(
            img,
            lambda d: (
                d.polygon([(1320, 0), (1600, 0), (1600, 230)], fill=rgba(style.secondary, 24)),
                d.polygon([(1390, 0), (1600, 0), (1600, 150)], fill=rgba(style.accent2, 26)),
            ),
        )
    elif style.key == "02_modern_minimal":
        draw.rectangle((0, 0, W, 18), fill=rgba(style.primary))
        draw.rectangle((0, 18, W, 24), fill=rgba(style.secondary))
        for x in (118, 1482):
            draw.line((x, 92, x, 806), fill=rgba(style.line), width=1)
        alpha_layer(
            img,
            lambda d: (
                d.ellipse((1250, -160, 1720, 310), fill=rgba(style.secondary, 25)),
                d.ellipse((-180, 670, 260, 1110), fill=rgba(style.accent, 17)),
            ),
        )
    elif style.key == "03_tech_rational":
        for x in range(120, W, 80):
            draw.line((x, 0, x, H), fill=rgba(style.line, 70), width=1)
        for y in range(80, H, 80):
            draw.line((0, y, W, y), fill=rgba(style.line, 55), width=1)
        draw.rectangle((0, 0, W, H), outline=rgba(style.primary, 90), width=3)
        alpha_layer(
            img,
            lambda d: (
                d.polygon([(0, 0), (410, 0), (0, 310)], fill=rgba(style.secondary, 34)),
                d.polygon([(1290, 900), (1600, 560), (1600, 900)], fill=rgba(style.accent, 26)),
            ),
        )
    else:
        draw.rectangle((0, 0, W, 64), fill=rgba("#FFFFFF", 150))
        draw.line((126, 104, 1474, 104), fill=rgba(style.line), width=2)
        draw.line((126, 812, 1474, 812), fill=rgba(style.line), width=1)
        alpha_layer(
            img,
            lambda d: (
                d.rounded_rectangle((1176, 82, 1510, 202), radius=60, fill=rgba(style.secondary, 34)),
                d.rounded_rectangle((70, 660, 420, 802), radius=70, fill=rgba(style.accent2, 40)),
            ),
        )


def new_slide(style: Style, page: str) -> Image.Image:
    img = gradient_bg(style)
    decorate_background(img, style, page)
    return img


def draw_logo(img: Image.Image, xy: tuple[int, int], size: int, style: Style, shadow: bool = True) -> None:
    logo = ImageOps.contain(LOGO, (size, size), method=Image.Resampling.LANCZOS)
    if shadow:
        paste_shadowed(img, logo, xy, shadow=(0, 0, 0, 75 if not style.dark else 120), blur=16, offset=(0, 8))
    else:
        img.alpha_composite(logo, xy)


def draw_page_label(draw: ImageDraw.ImageDraw, style: Style, text: str, page_no: int) -> None:
    color = rgba(style.muted if not style.dark else style.muted, 255)
    draw_text(draw, (138, 838), text, fnt(20), color)
    draw_text(draw, (1462, 838), f"{page_no:02d}", fnt(20), color, anchor="ra")


def title_bar(draw: ImageDraw.ImageDraw, style: Style, title: str, subtitle: str | None = None) -> None:
    draw_text(draw, (138, 54), title, fnt(34, "bold" if style.title_font != "serif" else "serif"), rgba(style.text))
    if subtitle:
        draw_text(draw, (138, 104), subtitle, fnt(18), rgba(style.muted))
    draw.rounded_rectangle((138, 132, 306, 138), radius=3, fill=rgba(style.accent))
    draw.rounded_rectangle((314, 132, 400, 138), radius=3, fill=rgba(style.accent2))


def draw_cover(style: Style) -> Image.Image:
    img = new_slide(style, "cover")
    draw = ImageDraw.Draw(img)

    if style.key == "03_tech_rational":
        draw_logo(img, (1160, 155), 250, style)
        draw_text(draw, (150, 144), "本科毕业论文答辩", fnt(28), rgba(style.primary))
        draw_text(draw, (150, 220), "WFBARNet", fnt(78, "bold"), rgba(style.text))
        lines = ["面向羽毛球视频的", "本地智能分析系统设计与实现"]
        y = 316
        for line in lines:
            draw_text(draw, (150, y), line, fnt(50, "bold"), rgba(style.text))
            y += 70
        draw.line((150, 508, 930, 508), fill=rgba(style.primary), width=3)
        info = [("答辩人", "沃寅博"), ("专业", "软件工程"), ("学院", "江苏师范大学科文学院"), ("答辩时间", "5月11日")]
        x0, y0 = 152, 570
        for i, (k, v) in enumerate(info):
            x = x0 + (i % 2) * 390
            y = y0 + (i // 2) * 72
            draw_text(draw, (x, y), k, fnt(20), rgba(style.muted))
            draw_text(draw, (x + 104, y - 4), v, fnt(25, "bold"), rgba(style.text))
    else:
        draw_logo(img, (1192, 146), 254, style)
        draw_text(draw, (158, 150), "本科毕业论文答辩", fnt(28), rgba(style.secondary if not style.dark else style.primary))
        draw_text(draw, (158, 228), "WFBARNet", fnt(80, "bold" if style.title_font != "serif" else "serif"), rgba(style.primary if not style.dark else style.text))
        draw_text(draw, (158, 330), "面向羽毛球视频的本地智能分析系统", fnt(50, "bold" if style.title_font != "serif" else "serif"), rgba(style.text))
        draw_text(draw, (158, 402), "设计与实现", fnt(50, "bold" if style.title_font != "serif" else "serif"), rgba(style.text))
        draw.rounded_rectangle((158, 492, 744, 500), radius=4, fill=rgba(style.accent))
        draw.rounded_rectangle((762, 492, 1030, 500), radius=4, fill=rgba(style.accent2))
        info = [("答辩人", "沃寅博"), ("专业", "软件工程"), ("学院", "江苏师范大学科文学院"), ("答辩时间", "5月11日")]
        y = 580
        for k, v in info:
            draw_text(draw, (164, y), f"{k}：", fnt(22), rgba(style.muted))
            draw_text(draw, (290, y - 2), v, fnt(26, "bold"), rgba(style.text))
            y += 52
    draw_page_label(draw, style, style.name, 1)
    return img


def draw_agenda(style: Style) -> Image.Image:
    img = new_slide(style, "agenda")
    draw = ImageDraw.Draw(img)
    title_bar(draw, style, "目录", "论文主要章节结构")
    draw_logo(img, (1338, 42), 90, style, shadow=False)

    items = [
        ("01", "绪论", "研究背景、意义与国内外现状"),
        ("02", "系统需求与总体设计", "功能需求、架构设计与模块划分"),
        ("03", "关键算法与实现", "轨迹跟踪、姿态估计、球场映射、事件识别"),
        ("04", "实验设计与结果分析", "测试数据、指标对比、可视化结果"),
        ("05", "总结与展望", "研究结论、创新点与后续优化"),
    ]
    x0, y0 = 210, 192
    for i, (num, head, desc) in enumerate(items):
        y = y0 + i * 108
        if style.key == "03_tech_rational":
            rounded(draw, (x0, y, 1390, y + 78), style.radius, rgba(style.panel, 205), rgba(style.line), 1)
        else:
            rounded(draw, (x0, y, 1390, y + 78), style.radius, rgba(style.panel, 235), rgba(style.line), 1)
        draw.rounded_rectangle((x0 + 28, y + 18, x0 + 92, y + 60), radius=8, fill=rgba(style.primary if not style.dark else style.secondary))
        draw_text(draw, (x0 + 60, y + 38), num, fnt(22, "bold"), rgba("#FFFFFF"), anchor="mm")
        draw_text(draw, (x0 + 128, y + 14), head, fnt(28, "bold"), rgba(style.text))
        draw_text(draw, (x0 + 128, y + 50), desc, fnt(20), rgba(style.muted))
        draw.line((x0 + 1000, y + 39, x0 + 1120, y + 39), fill=rgba(style.accent), width=3)
    draw_page_label(draw, style, style.name, 2)
    return img


def draw_chapter(style: Style) -> Image.Image:
    img = new_slide(style, "chapter")
    draw = ImageDraw.Draw(img)
    draw_logo(img, (1258, 112), 200, style)
    if style.key == "01_stable_academic":
        draw_text(draw, (196, 230), "03", fnt(140, "serif"), rgba(style.primary))
        draw_text(draw, (202, 394), "关键算法与系统实现", fnt(58, "serif"), rgba(style.text))
    elif style.key == "03_tech_rational":
        draw_text(draw, (180, 220), "SECTION 03", fnt(34, "bold"), rgba(style.primary))
        draw_text(draw, (180, 300), "关键算法与系统实现", fnt(64, "bold"), rgba(style.text))
    else:
        draw_text(draw, (192, 218), "03", fnt(132, "bold"), rgba(style.primary))
        draw_text(draw, (196, 380), "关键算法与系统实现", fnt(60, "bold"), rgba(style.text))
    draw.rounded_rectangle((202, 474, 900, 482), radius=4, fill=rgba(style.accent))
    draw_text(draw, (204, 526), "本章重点展示轨迹跟踪、姿态估计、球场映射与事件识别的实现思路。", fnt(27), rgba(style.muted))
    chips = ["TrackNetV3", "YOLO Pose", "Court Homography", "Trajectory Events"]
    x = 204
    for chip in chips:
        tw, _ = text_size(draw, chip, fnt(20, "bold"))
        rounded(draw, (x, 624, x + tw + 42, 672), 24, rgba(style.panel, 235 if not style.dark else 210), rgba(style.line), 1)
        draw_text(draw, (x + 21, 635), chip, fnt(20, "bold"), rgba(style.secondary if not style.dark else style.primary))
        x += tw + 64
    draw_page_label(draw, style, style.name, 3)
    return img


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color, width=3):
    draw.line((start, end), fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    if x2 >= x1:
        points = [(x2, y2), (x2 - 14, y2 - 8), (x2 - 14, y2 + 8)]
    else:
        points = [(x2, y2), (x2 + 14, y2 - 8), (x2 + 14, y2 + 8)]
    draw.polygon(points, fill=color)


def flow_box(draw: ImageDraw.ImageDraw, style: Style, box: tuple[int, int, int, int], title: str, desc: str, fill: str | None = None):
    fill_col = fill or style.panel
    rounded(draw, box, style.radius, rgba(fill_col, 240 if not style.dark else 218), rgba(style.line), 2)
    x1, y1, x2, y2 = box
    draw.rounded_rectangle((x1 + 18, y1 + 18, x1 + 72, y1 + 72), radius=12, fill=rgba(style.primary if not style.dark else style.secondary))
    draw_text(draw, (x1 + 45, y1 + 45), "●", fnt(24), rgba("#FFFFFF"), anchor="mm")
    draw_text(draw, (x1 + 90, y1 + 20), title, fnt(25, "bold"), rgba(style.text))
    for i, line in enumerate(wrap_text(draw, desc, fnt(18), x2 - x1 - 112, 2)):
        draw_text(draw, (x1 + 90, y1 + 56 + i * 25), line, fnt(18), rgba(style.muted))


def draw_framework(style: Style) -> Image.Image:
    img = new_slide(style, "framework")
    draw = ImageDraw.Draw(img)
    title_bar(draw, style, "研究框架与技术路线", "从视频帧到回合统计的端到端分析链路")
    draw_logo(img, (1350, 44), 78, style, shadow=False)

    input_box = (150, 236, 410, 366)
    branch_boxes = [(540, 176, 875, 286), (540, 338, 875, 448), (540, 500, 875, 610)]
    merge_box = (1010, 266, 1350, 406)
    output_box = (1010, 508, 1350, 650)
    flow_box(draw, style, input_box, "视频输入", "本地视频 / 摄像头 / 批量帧")
    flow_box(draw, style, branch_boxes[0], "轨迹分支", "TrackNetV3 热力图与候选球点")
    flow_box(draw, style, branch_boxes[1], "姿态分支", "YOLO Pose 人体框与关键点")
    flow_box(draw, style, branch_boxes[2], "球场分支", "场线检测与单应性映射")
    flow_box(draw, style, merge_box, "事件识别", "hit / landing / out_of_frame")
    flow_box(draw, style, output_box, "统计与展示", "回合指标、热力图、日志导出")

    for bx in branch_boxes:
        arrow(draw, (410, 301), (540, (bx[1] + bx[3]) // 2), rgba(style.accent), width=4)
        arrow(draw, (875, (bx[1] + bx[3]) // 2), (1010, 336), rgba(style.accent2), width=4)
    arrow(draw, (1180, 406), (1180, 508), rgba(style.primary if not style.dark else style.primary), width=4)

    draw_text(draw, (150, 700), "关键设计：多路视觉感知 + 规则式可解释后处理 + PyQt6 可视化工作台", fnt(26, "bold"), rgba(style.text))
    draw_page_label(draw, style, style.name, 4)
    return img


def draw_chart(draw: ImageDraw.ImageDraw, style: Style, box: tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    rounded(draw, box, style.radius, rgba(style.panel, 238 if not style.dark else 216), rgba(style.line), 1)
    draw_text(draw, (x1 + 28, y1 + 22), "轨迹检测与事件识别结果", fnt(24, "bold"), rgba(style.text))
    axis_x, axis_y = x1 + 58, y2 - 62
    draw.line((axis_x, y1 + 80, axis_x, axis_y), fill=rgba(style.line), width=2)
    draw.line((axis_x, axis_y, x2 - 36, axis_y), fill=rgba(style.line), width=2)
    bars = [0.82, 0.76, 0.89, 0.71]
    labels = ["轨迹", "姿态", "场地", "事件"]
    colors = [style.primary, style.secondary, style.accent, style.accent2]
    bw = 60
    gap = 80
    for i, value in enumerate(bars):
        bx = axis_x + 56 + i * (bw + gap)
        bh = int(230 * value)
        draw.rounded_rectangle((bx, axis_y - bh, bx + bw, axis_y), radius=8, fill=rgba(colors[i]))
        draw_text(draw, (bx + bw // 2, axis_y + 18), labels[i], fnt(17), rgba(style.muted), anchor="ma")
        draw_text(draw, (bx + bw // 2, axis_y - bh - 28), f"{value:.0%}", fnt(18, "bold"), rgba(style.text), anchor="ma")
    points = [(axis_x + 70, axis_y - 120), (axis_x + 190, axis_y - 160), (axis_x + 315, axis_y - 138), (axis_x + 440, axis_y - 206), (axis_x + 555, axis_y - 190)]
    draw.line(points, fill=rgba(style.accent), width=4)
    for p in points:
        draw.ellipse((p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6), fill=rgba(style.accent2))


def draw_table(draw: ImageDraw.ImageDraw, style: Style, box: tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    rounded(draw, box, style.radius, rgba(style.panel, 238 if not style.dark else 216), rgba(style.line), 1)
    draw_text(draw, (x1 + 28, y1 + 22), "对比指标示意", fnt(24, "bold"), rgba(style.text))
    cols = ["指标", "方案A", "方案B", "本文"]
    rows = [
        ["可见率", "78.4", "82.1", "88.9"],
        ["误检率", "12.6", "9.8", "6.1"],
        ["FPS", "18", "24", "31"],
        ["回合统计", "部分", "部分", "完整"],
    ]
    tx, ty = x1 + 26, y1 + 78
    col_w = [130, 90, 90, 90]
    row_h = 48
    draw.rounded_rectangle((tx, ty, x2 - 26, ty + row_h), radius=8, fill=rgba(style.primary if not style.dark else style.secondary, 230))
    cx = tx
    for i, col in enumerate(cols):
        draw_text(draw, (cx + 14, ty + 13), col, fnt(17, "bold"), rgba("#FFFFFF"))
        cx += col_w[i]
    for r, row in enumerate(rows):
        y = ty + row_h * (r + 1)
        fill = rgba("#FFFFFF" if not style.dark else "#142239", 165 if style.dark else 210)
        draw.rectangle((tx, y, x2 - 26, y + row_h), fill=fill)
        cx = tx
        for i, val in enumerate(row):
            color = style.accent if i == 3 else style.text
            draw_text(draw, (cx + 14, y + 13), val, fnt(17, "bold" if i == 3 else "sans"), rgba(color))
            cx += col_w[i]
        draw.line((tx, y + row_h, x2 - 26, y + row_h), fill=rgba(style.line), width=1)


def draw_results(style: Style) -> Image.Image:
    img = new_slide(style, "results")
    draw = ImageDraw.Draw(img)
    title_bar(draw, style, "实验结果与分析", "结果展示页模板：图表、曲线、表格与关键指标")
    draw_logo(img, (1350, 44), 78, style, shadow=False)
    draw_chart(draw, style, (132, 176, 880, 618))
    draw_table(draw, style, (926, 176, 1468, 618))

    cards = [
        ("31 FPS", "实时推理速度"),
        ("88.9%", "球点可见率"),
        ("6.1%", "轨迹误检率"),
        ("完整", "回合统计输出"),
    ]
    x = 132
    for value, label in cards:
        rounded(draw, (x, 668, x + 315, 760), style.radius, rgba(style.panel, 238 if not style.dark else 216), rgba(style.line), 1)
        draw_text(draw, (x + 28, 690), value, fnt(32, "bold"), rgba(style.primary if not style.dark else style.primary))
        draw_text(draw, (x + 28, 733), label, fnt(19), rgba(style.muted))
        x += 338
    draw_page_label(draw, style, style.name, 5)
    return img


def draw_summary(style: Style) -> Image.Image:
    img = new_slide(style, "summary")
    draw = ImageDraw.Draw(img)
    title_bar(draw, style, "总结与展望", "研究结论、创新点与后续优化方向")
    draw_logo(img, (1346, 45), 82, style, shadow=False)

    left = (150, 188, 760, 642)
    right = (832, 188, 1450, 642)
    rounded(draw, left, style.radius, rgba(style.panel, 238 if not style.dark else 216), rgba(style.line), 1)
    rounded(draw, right, style.radius, rgba(style.panel, 238 if not style.dark else 216), rgba(style.line), 1)
    draw_text(draw, (190, 226), "研究结论", fnt(30, "bold"), rgba(style.text))
    conclusions = [
        "构建了围绕羽毛球、球员、球场三对象的本地分析链路。",
        "实现了轨迹事件识别与回合级数据统计，支持训练复盘。",
        "提供 PyQt6 可视化界面与调试日志，便于算法验证和迭代。",
    ]
    y = 292
    for idx, item in enumerate(conclusions, 1):
        draw.rounded_rectangle((190, y - 4, 228, y + 34), radius=19, fill=rgba(style.primary if not style.dark else style.secondary))
        draw_text(draw, (209, y + 14), str(idx), fnt(17, "bold"), rgba("#FFFFFF"), anchor="mm")
        for line in wrap_text(draw, item, fnt(22), 455, 2):
            draw_text(draw, (248, y), line, fnt(22), rgba(style.text))
            y += 30
        y += 24

    draw_text(draw, (872, 226), "创新点与展望", fnt(30, "bold"), rgba(style.text))
    future = [
        ("创新点", "可解释事件规则、多源数据融合、可追溯调试体系"),
        ("展望", "多回合自动切分、界内外判断、跨场景模型泛化"),
        ("应用", "训练复盘、技战术观察、批量数据整理"),
    ]
    y = 292
    for head, body in future:
        draw.rounded_rectangle((872, y, 970, y + 38), radius=19, fill=rgba(style.accent, 220))
        draw_text(draw, (921, y + 18), head, fnt(17, "bold"), rgba("#FFFFFF"), anchor="mm")
        for line in wrap_text(draw, body, fnt(22), 390, 2):
            draw_text(draw, (994, y + 2), line, fnt(22), rgba(style.text))
            y += 30
        y += 44

    if style.key == "03_tech_rational":
        draw_text(draw, (800, 750), "谢谢聆听", fnt(46, "bold"), rgba(style.text), anchor="mm")
        draw_text(draw, (800, 806), "欢迎各位老师批评指正", fnt(24), rgba(style.primary), anchor="mm")
    else:
        draw_text(draw, (800, 744), "谢谢聆听", fnt(48, "bold" if style.title_font != "serif" else "serif"), rgba(style.primary), anchor="mm")
        draw_text(draw, (800, 802), "欢迎各位老师批评指正", fnt(24), rgba(style.muted), anchor="mm")
    draw_page_label(draw, style, style.name, 6)
    return img


DRAWERS = {
    "cover": draw_cover,
    "agenda": draw_agenda,
    "chapter": draw_chapter,
    "framework": draw_framework,
    "results": draw_results,
    "summary": draw_summary,
}


def create_style_card(style: Style, style_dir: Path) -> Image.Image:
    card = Image.new("RGBA", (1600, 1000), rgba("#FFFFFF" if not style.dark else "#07101D"))
    draw = ImageDraw.Draw(card)
    draw_text(draw, (70, 42), f"{style.name}｜{style.brief}", fnt(34, "bold"), rgba("#1F2937" if not style.dark else "#EEF6FF"))
    palette = [style.primary, style.secondary, style.accent, style.accent2, style.bg, style.panel]
    x = 70
    for col in palette:
        draw.rounded_rectangle((x, 104, x + 104, 164), radius=12, fill=rgba(col), outline=rgba("#D0D7E2"))
        draw_text(draw, (x + 52, 184), col, fnt(15), rgba("#475467" if not style.dark else "#B9C7D9"), anchor="ma")
        x += 142

    thumbs = []
    for filename, label in PAGES:
        img = Image.open(style_dir / f"{filename}.png").convert("RGBA")
        thumb = ImageOps.contain(img, (460, 259), method=Image.Resampling.LANCZOS)
        framed = Image.new("RGBA", (500, 314), rgba("#F7F9FC" if not style.dark else "#0D182A"))
        d = ImageDraw.Draw(framed)
        framed.alpha_composite(thumb, ((500 - thumb.width) // 2, 20))
        d.rounded_rectangle((18, 18, 482, 282), radius=10, outline=rgba("#CDD5E1" if not style.dark else "#2D4565"), width=1)
        d.text((250, 292), label, font=fnt(17, "bold"), fill=rgba("#344054" if not style.dark else "#DCE8F6"), anchor="ma")
        thumbs.append(framed)

    coords = [(58, 250), (550, 250), (1042, 250), (58, 598), (550, 598), (1042, 598)]
    for thumb, xy in zip(thumbs, coords):
        card.alpha_composite(thumb, xy)
    return card


def write_gallery_md(output_dir: Path, saved: dict[str, list[tuple[str, Path]]]) -> None:
    lines = [
        "# WFBARNet 答辩 PPT 视觉风格方案",
        "",
        "本目录包含 4 套视觉风格方向，每套均提供 6 类页面效果图：封面页、目录页、章节页、研究框架页、实验结果页、总结页。",
        "",
        "页面尺寸为 16:9，输出 PNG 可直接作为 PPT 设计参考。",
        "",
    ]
    for style in STYLES:
        lines.extend(
            [
                f"## {style.name}",
                "",
                f"- 风格说明：{style.brief}",
                f"- 主色：`{style.primary}`，辅助色：`{style.secondary}`，点缀色：`{style.accent}` / `{style.accent2}`",
                f"- 字体建议：{'宋体/思源宋体标题 + 微软雅黑正文' if style.title_font == 'serif' else '微软雅黑/思源黑体标题 + 微软雅黑正文'}",
                f"- 版式建议：{'深色网格与线框信息卡' if style.dark else '浅色背景、细线分隔、低饱和卡片'}",
                "",
                f"![{style.name} contact sheet]({style.key}/contact_sheet.png)",
                "",
            ]
        )
        for label, path in saved[style.key]:
            rel = path.relative_to(output_dir).as_posix()
            lines.append(f"- [{label}]({rel})")
        lines.append("")
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = unique_output_dir(OUTPUT_BASE)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, list[tuple[str, Path]]] = {}
    for style in STYLES:
        style_dir = output_dir / style.key
        style_dir.mkdir(parents=True, exist_ok=True)
        saved[style.key] = []
        for page_key, label in PAGES:
            slide = DRAWERS[page_key](style).convert("RGB")
            path = style_dir / f"{page_key}.png"
            slide.save(path, quality=95)
            saved[style.key].append((label, path))
        contact = create_style_card(style, style_dir).convert("RGB")
        contact.save(style_dir / "contact_sheet.png", quality=95)

    write_gallery_md(output_dir, saved)
    print(output_dir)


if __name__ == "__main__":
    main()
