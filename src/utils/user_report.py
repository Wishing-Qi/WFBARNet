from __future__ import annotations

from datetime import datetime
from html import escape
from math import cos, pi, sin
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAYER_KEYS = ("top", "bottom")
PLAYER_LABELS = {
    "top": "上方球员",
    "bottom": "下方球员",
}
ZONE_LABELS = {
    "front": "前场",
    "mid": "中场",
    "back": "后场",
}
RELIABILITY_LABELS = {
    "ball_visible_rate": "球点可见率",
    "pose_valid_rate": "姿态有效率",
    "court_valid_rate": "球场有效率",
    "avg_ball_confidence": "平均球置信度",
}


def export_user_report_html(report_data: Mapping[str, Any] | None, path: Path) -> None:
    """Write a standalone UTF-8 HTML badminton user report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_user_report_html(report_data or sample_user_report_data()), encoding="utf-8")


def sample_user_report_data(*, generated_at: str | None = None) -> dict[str, Any]:
    """Build a realistic demo report from data shaped like RallyStatsAccumulator output."""
    sample_record = {
        "rally_id": "demo-training-20260514",
        "rally_name": "user_uploaded_training_clip.mp4",
        "summary": {
            "rally_name": "user_uploaded_training_clip.mp4",
            "duration_s": 138.0,
            "rally_duration_s": 92.4,
            "rally_state": "回合结束",
            "frame_count": 4140,
            "rally_hit_count": 42,
            "landing_count": 8,
            "out_of_frame_count": 2,
            "avg_hit_interval_ms": 2180.0,
            "hit_confidence_avg": 0.84,
            "motion_intensity_score": 78.6,
            "high_intensity_count": 29,
            "stroke_distribution": {
                "高远球": 12,
                "杀球": 9,
                "吊球": 7,
                "挑球": 6,
                "搓放": 5,
                "平抽挡": 3,
            },
            "data_reliability": {
                "ball_visible_rate": 0.91,
                "pose_valid_rate": 0.87,
                "court_valid_rate": 0.82,
                "avg_ball_confidence": 0.78,
            },
            "players": {
                "top": {
                    "label": "陪练球员",
                    "distance_m": 82.6,
                    "avg_speed_mps": 1.08,
                    "max_speed_mps": 4.62,
                    "stop_count": 19,
                    "start_count": 24,
                    "hit_count": 39,
                    "zone_hits": {"front": 11, "mid": 15, "back": 13},
                    "passive_hit_count": 7,
                    "high_intensity_count": 12,
                    "max_continuous_m": 8.1,
                    "front_back_movement_ratio": 0.53,
                    "left_right_movement_ratio": 0.47,
                    "avg_stance_depth_cm": 382.0,
                },
                "bottom": {
                    "label": "目标用户",
                    "distance_m": 96.4,
                    "avg_speed_mps": 1.24,
                    "max_speed_mps": 4.15,
                    "stop_count": 31,
                    "start_count": 34,
                    "hit_count": 42,
                    "zone_hits": {"front": 8, "mid": 16, "back": 18},
                    "passive_hit_count": 13,
                    "high_intensity_count": 17,
                    "max_continuous_m": 6.4,
                    "front_back_movement_ratio": 0.41,
                    "left_right_movement_ratio": 0.56,
                    "avg_stance_depth_cm": 428.0,
                },
            },
        },
        "details": {
            "hits": [
                {
                    "timestamp_ms": 4160,
                    "player": "bottom",
                    "player_label": "目标用户",
                    "zone": "back",
                    "stroke": "高远球",
                    "confidence": 0.86,
                    "court_xy": [472.0, 1198.0],
                    "passive": False,
                },
                {
                    "timestamp_ms": 6210,
                    "player": "top",
                    "player_label": "陪练球员",
                    "zone": "mid",
                    "stroke": "吊球",
                    "confidence": 0.81,
                    "court_xy": [282.0, 438.0],
                    "passive": False,
                },
                {
                    "timestamp_ms": 8260,
                    "player": "bottom",
                    "player_label": "目标用户",
                    "zone": "front",
                    "stroke": "搓放",
                    "confidence": 0.79,
                    "court_xy": [318.0, 724.0],
                    "passive": True,
                },
                {
                    "timestamp_ms": 10880,
                    "player": "bottom",
                    "player_label": "目标用户",
                    "zone": "back",
                    "stroke": "杀球",
                    "confidence": 0.88,
                    "court_xy": [503.0, 1246.0],
                    "passive": False,
                },
                {
                    "timestamp_ms": 13320,
                    "player": "top",
                    "player_label": "陪练球员",
                    "zone": "front",
                    "stroke": "挑球",
                    "confidence": 0.83,
                    "court_xy": [238.0, 612.0],
                    "passive": False,
                },
                {
                    "timestamp_ms": 15980,
                    "player": "bottom",
                    "player_label": "目标用户",
                    "zone": "mid",
                    "stroke": "平抽挡",
                    "confidence": 0.76,
                    "court_xy": [366.0, 944.0],
                    "passive": True,
                },
                {
                    "timestamp_ms": 18420,
                    "player": "bottom",
                    "player_label": "目标用户",
                    "zone": "back",
                    "stroke": "高远球",
                    "confidence": 0.85,
                    "court_xy": [132.0, 1222.0],
                    "passive": False,
                },
                {
                    "timestamp_ms": 21140,
                    "player": "bottom",
                    "player_label": "目标用户",
                    "zone": "back",
                    "stroke": "杀球",
                    "confidence": 0.90,
                    "court_xy": [492.0, 1264.0],
                    "passive": False,
                },
            ]
        },
    }
    return build_user_report_data_from_rally_record(
        sample_record,
        user_player="bottom",
        athlete_name="体验用户 A",
        video_name="user_uploaded_training_clip.mp4",
        generated_at=generated_at,
    )


def build_user_report_data_from_rally_record(
    record: Mapping[str, Any],
    *,
    user_player: str = "bottom",
    athlete_name: str = "训练用户",
    video_name: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Convert a WFBARNet rally record into presentation-ready report data."""
    summary = _as_mapping(record.get("summary"))
    details = _as_mapping(record.get("details"))
    players = _as_mapping(summary.get("players"))
    user_key = user_player if user_player in PLAYER_KEYS else "bottom"
    opponent_key = "top" if user_key == "bottom" else "bottom"
    user = _as_mapping(players.get(user_key))
    opponent = _as_mapping(players.get(opponent_key))
    reliability = _as_mapping(summary.get("data_reliability"))
    strokes = _counter_items(summary.get("stroke_distribution"))
    zones = _zone_counts(user.get("zone_hits"))
    dimensions = _dimension_scores(summary, user, reliability, strokes, zones)
    overall_score = round(sum(item["score"] * item["weight"] for item in dimensions) / sum(item["weight"] for item in dimensions))
    improvements = _build_improvements(dimensions, summary, user, reliability, zones, strokes)

    return {
        "meta": {
            "title": "羽毛球水平分析报告",
            "subtitle": "上传视频后由 WFBARNet 自动生成的训练复盘",
            "athlete_name": athlete_name,
            "video_name": video_name or str(record.get("video_name") or record.get("rally_name") or summary.get("rally_name") or "uploaded_video.mp4"),
            "generated_at": generated_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pipeline": "TrackNetV3 轨迹 + YOLO Pose 姿态 + 球场单应性 + BST 击球识别",
            "user_player": PLAYER_LABELS[user_key],
        },
        "summary": {
            "overall_score": overall_score,
            "level_label": _level_label(overall_score),
            "level_note": _level_note(overall_score),
            "takeaways": _takeaways(overall_score, dimensions, user, reliability),
            "metrics": [
                {"label": "分析时长", "value": _fmt_seconds(summary.get("duration_s")), "hint": "上传视频有效片段"},
                {"label": "有效击球", "value": f"{int(_num(summary.get('rally_hit_count')))} 次", "hint": "轨迹/BST 合并后"},
                {"label": "目标用户跑动", "value": f"{_num(user.get('distance_m')):.1f} m", "hint": "标准球场坐标累计"},
                {"label": "平均击球间隔", "value": _fmt_seconds(_num(summary.get("avg_hit_interval_ms")) / 1000.0), "hint": "节奏越低越紧凑"},
            ],
        },
        "dimensions": dimensions,
        "players": {
            "user": _player_report(user, "目标用户"),
            "opponent": _player_report(opponent, "对手/陪练"),
        },
        "stroke_distribution": strokes,
        "zone_distribution": [{"key": key, "label": ZONE_LABELS[key], "value": zones[key]} for key in ("front", "mid", "back")],
        "court_heat": _court_heat_points(zones, user_key),
        "reliability": _reliability_items(reliability),
        "improvements": improvements,
        "training_plan": _training_plan(improvements),
        "events": _hit_events(details.get("hits"), user_key),
        "disclaimer": "本报告基于视频中的轨迹、姿态和球场投影自动估计，适合训练复盘，不作为比赛裁判或等级认证依据。",
    }


def render_user_report_html(report_data: Mapping[str, Any]) -> str:
    meta = _as_mapping(report_data.get("meta"))
    summary = _as_mapping(report_data.get("summary"))
    title = str(meta.get("title", "羽毛球水平分析报告"))
    subtitle = str(meta.get("subtitle", "上传视频后自动生成的训练复盘"))
    score = int(_num(summary.get("overall_score")))

    metric_cards = "\n".join(_render_metric_card(item) for item in _as_sequence(summary.get("metrics")))
    takeaways = "\n".join(f"<li>{escape(str(item))}</li>" for item in _as_sequence(summary.get("takeaways")))
    dimensions = [_as_mapping(item) for item in _as_sequence(report_data.get("dimensions"))]
    players = _as_mapping(report_data.get("players"))
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
{_report_css()}
  </style>
</head>
<body>
  <main class="report-shell">
    <section class="hero panel-rise">
      <div class="hero-copy">
        <p class="eyebrow">WFBARNet 智能视频分析</p>
        <h1>{escape(title)}</h1>
        <p class="hero-subtitle">{escape(subtitle)}</p>
        <div class="meta-row" aria-label="报告元信息">
          <span>{escape(str(meta.get("athlete_name", "训练用户")))}</span>
          <span>{escape(str(meta.get("video_name", "uploaded_video.mp4")))}</span>
          <span>{escape(str(meta.get("generated_at", "")))}</span>
          <span>{escape(str(meta.get("user_player", "目标用户")))}</span>
        </div>
      </div>
      <div class="score-card" aria-label="综合水平评分">
        {_render_score_ring(score)}
        <strong>{escape(str(summary.get("level_label", "")))}</strong>
        <span>{escape(str(summary.get("level_note", "")))}</span>
      </div>
    </section>

    <section class="metric-grid" aria-label="核心数据">
      {metric_cards}
    </section>

    <section class="grid-two">
      <article class="panel">
        <div class="section-heading">
          <p class="eyebrow">Skill Radar</p>
          <h2>能力维度图</h2>
        </div>
        {_render_radar_svg(dimensions)}
      </article>
      <article class="panel insight-panel">
        <div class="section-heading">
          <p class="eyebrow">Diagnosis</p>
          <h2>本次结论</h2>
        </div>
        <ul class="takeaway-list">
          {takeaways}
        </ul>
        {_render_dimension_rows(dimensions)}
      </article>
    </section>

    <section class="grid-two">
      <article class="panel">
        <div class="section-heading">
          <p class="eyebrow">Movement</p>
          <h2>双方运动表现对比</h2>
        </div>
        {_render_player_comparison(players)}
      </article>
      <article class="panel">
        <div class="section-heading">
          <p class="eyebrow">Court Map</p>
          <h2>站位热区与击球区域</h2>
        </div>
        <div class="court-layout">
          {_render_court_svg(_as_sequence(report_data.get("court_heat")))}
          {_render_zone_bars(_as_sequence(report_data.get("zone_distribution")))}
        </div>
      </article>
    </section>

    <section class="grid-two grid-two-wide">
      <article class="panel">
        <div class="section-heading">
          <p class="eyebrow">Shots</p>
          <h2>击球类型分布</h2>
        </div>
        {_render_stroke_distribution(_as_sequence(report_data.get("stroke_distribution")))}
      </article>
      <article class="panel">
        <div class="section-heading">
          <p class="eyebrow">Reliability</p>
          <h2>数据可信度</h2>
        </div>
        {_render_reliability(_as_sequence(report_data.get("reliability")))}
      </article>
    </section>

    <section class="grid-two">
      <article class="panel improvement-panel">
        <div class="section-heading">
          <p class="eyebrow">Coaching Notes</p>
          <h2>需要改进的点</h2>
        </div>
        {_render_improvements(_as_sequence(report_data.get("improvements")))}
      </article>
      <article class="panel">
        <div class="section-heading">
          <p class="eyebrow">Next 2 Weeks</p>
          <h2>训练建议</h2>
        </div>
        {_render_training_plan(_as_sequence(report_data.get("training_plan")))}
      </article>
    </section>

    <section class="panel">
      <div class="section-heading">
        <p class="eyebrow">Hit Timeline</p>
        <h2>关键击球明细</h2>
      </div>
      {_render_events_table(_as_sequence(report_data.get("events")))}
    </section>

    <footer class="report-footer">
      <span>{escape(str(meta.get("pipeline", "")))}</span>
      <span>{escape(str(report_data.get("disclaimer", "")))}</span>
    </footer>
  </main>
</body>
</html>
"""
    return html


def _report_css() -> str:
    return """
:root {
  --ink: #10231d;
  --muted: #5f7068;
  --paper: #fffaf0;
  --paper-strong: #fffdf7;
  --court: #0e704a;
  --court-deep: #073d31;
  --line: rgba(16, 35, 29, 0.12);
  --lime: #b6e95e;
  --gold: #f0be49;
  --coral: #ee705d;
  --sky: #6bc5d2;
  --shadow: 0 24px 80px rgba(7, 61, 49, 0.18);
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  color: var(--ink);
  font-family: "MiSans", "HarmonyOS Sans SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
  background:
    radial-gradient(circle at top left, rgba(182, 233, 94, 0.35), transparent 30rem),
    radial-gradient(circle at 85% 12%, rgba(107, 197, 210, 0.24), transparent 28rem),
    linear-gradient(135deg, #f7efd9 0%, #edf5df 46%, #fdf8ec 100%);
}

.report-shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 32px 0 42px;
}

.panel-rise,
.panel,
.metric-card {
  border: 1px solid rgba(16, 35, 29, 0.1);
  box-shadow: var(--shadow);
}

.hero {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 250px;
  gap: 28px;
  overflow: hidden;
  padding: 42px;
  color: #f8fff2;
  border-radius: 34px;
  background:
    linear-gradient(135deg, rgba(7, 61, 49, 0.96), rgba(14, 112, 74, 0.92) 55%, rgba(240, 190, 73, 0.88)),
    var(--court-deep);
}

.hero::before,
.hero::after {
  content: "";
  position: absolute;
  pointer-events: none;
}

.hero::before {
  inset: 18px;
  border: 2px solid rgba(248, 255, 242, 0.22);
  border-radius: 26px;
}

.hero::after {
  right: -80px;
  bottom: -120px;
  width: 360px;
  height: 360px;
  border-radius: 999px;
  background: repeating-linear-gradient(35deg, rgba(255, 255, 255, 0.16) 0 2px, transparent 2px 16px);
  opacity: 0.8;
}

.hero-copy,
.score-card {
  position: relative;
  z-index: 1;
}

.eyebrow {
  margin: 0 0 10px;
  color: var(--coral);
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.hero .eyebrow {
  color: var(--lime);
}

h1,
h2,
h3 {
  margin: 0;
  font-family: "Source Han Serif SC", "Noto Serif CJK SC", "STZhongsong", serif;
  letter-spacing: -0.03em;
}

h1 {
  max-width: 760px;
  font-size: clamp(2.55rem, 6vw, 5.6rem);
  line-height: 0.92;
}

h2 {
  font-size: clamp(1.35rem, 2vw, 2rem);
}

h3 {
  font-size: 1.05rem;
}

.hero-subtitle {
  max-width: 660px;
  margin: 18px 0 0;
  color: rgba(248, 255, 242, 0.86);
  font-size: 1.08rem;
  line-height: 1.75;
}

.meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 26px;
}

.meta-row span {
  min-height: 36px;
  padding: 8px 13px;
  border: 1px solid rgba(248, 255, 242, 0.22);
  border-radius: 999px;
  color: rgba(248, 255, 242, 0.92);
  background: rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(10px);
}

.score-card {
  align-self: center;
  display: grid;
  place-items: center;
  gap: 8px;
  min-height: 280px;
  padding: 22px;
  border: 1px solid rgba(248, 255, 242, 0.18);
  border-radius: 28px;
  background: rgba(255, 255, 255, 0.12);
  text-align: center;
}

.score-card strong {
  font-size: 1.3rem;
}

.score-card span {
  max-width: 190px;
  color: rgba(248, 255, 242, 0.78);
  line-height: 1.55;
}

.score-ring text {
  font-family: "DIN Condensed", "Bahnschrift", "MiSans", sans-serif;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin-top: 18px;
}

.metric-card,
.panel {
  border-radius: 26px;
  background: rgba(255, 253, 247, 0.86);
  backdrop-filter: blur(16px);
}

.metric-card {
  padding: 20px;
}

.metric-card span,
.bar-label,
.table-note {
  color: var(--muted);
  font-size: 0.86rem;
}

.metric-card strong {
  display: block;
  margin-top: 10px;
  font-size: clamp(1.65rem, 3vw, 2.55rem);
  line-height: 1;
}

.metric-card small {
  display: block;
  margin-top: 10px;
  color: #75847d;
  line-height: 1.55;
}

.grid-two {
  display: grid;
  grid-template-columns: minmax(0, 0.94fr) minmax(0, 1.06fr);
  gap: 18px;
  margin-top: 18px;
}

.grid-two-wide {
  grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr);
}

.panel {
  padding: 26px;
}

.section-heading {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 18px;
  margin-bottom: 18px;
}

.takeaway-list {
  display: grid;
  gap: 10px;
  margin: 0 0 20px;
  padding: 0;
  list-style: none;
}

.takeaway-list li {
  position: relative;
  padding: 13px 14px 13px 38px;
  border-radius: 18px;
  background: rgba(14, 112, 74, 0.08);
  line-height: 1.6;
}

.takeaway-list li::before {
  content: "";
  position: absolute;
  left: 16px;
  top: 21px;
  width: 9px;
  height: 9px;
  border-radius: 99px;
  background: var(--court);
}

.radar-wrap {
  display: grid;
  place-items: center;
  min-height: 390px;
}

.radar-grid {
  fill: none;
  stroke: rgba(16, 35, 29, 0.14);
}

.radar-axis {
  stroke: rgba(16, 35, 29, 0.18);
}

.radar-area {
  fill: rgba(14, 112, 74, 0.28);
  stroke: var(--court);
  stroke-width: 3;
}

.radar-benchmark {
  fill: rgba(240, 190, 73, 0.16);
  stroke: rgba(240, 190, 73, 0.95);
  stroke-width: 2;
  stroke-dasharray: 6 7;
}

.radar-label {
  fill: var(--ink);
  font-size: 13px;
  font-weight: 800;
}

.dimension-list,
.comparison-list,
.stroke-list,
.reliability-list,
.training-list {
  display: grid;
  gap: 13px;
}

.dimension-row,
.comparison-row,
.stroke-row,
.reliability-row {
  display: grid;
  gap: 8px;
}

.row-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-weight: 800;
}

.row-note {
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.5;
}

.track {
  height: 12px;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(16, 35, 29, 0.09);
}

.fill {
  height: 100%;
  width: var(--value);
  border-radius: inherit;
  background: linear-gradient(90deg, var(--court), var(--lime));
}

.fill-opponent {
  background: linear-gradient(90deg, var(--coral), var(--gold));
}

.comparison-row {
  padding: 14px;
  border-radius: 18px;
  background: rgba(16, 35, 29, 0.045);
}

.comparison-bars {
  display: grid;
  gap: 7px;
}

.bar-line {
  display: grid;
  grid-template-columns: 74px minmax(0, 1fr) 72px;
  align-items: center;
  gap: 9px;
}

.court-layout {
  display: grid;
  grid-template-columns: minmax(180px, 0.84fr) minmax(180px, 1fr);
  gap: 22px;
  align-items: center;
}

.court-svg {
  width: 100%;
  max-height: 460px;
}

.court-line {
  fill: #168958;
  stroke: rgba(255, 255, 255, 0.9);
  stroke-width: 4;
}

.court-inner-line {
  stroke: rgba(255, 255, 255, 0.9);
  stroke-width: 3;
}

.heat {
  mix-blend-mode: screen;
}

.zone-stack {
  display: grid;
  gap: 14px;
}

.zone-card {
  padding: 14px;
  border-radius: 18px;
  background: rgba(14, 112, 74, 0.07);
}

.stroke-row {
  grid-template-columns: 110px minmax(0, 1fr) 58px;
  align-items: center;
}

.stroke-name {
  font-weight: 800;
}

.reliability-row {
  padding-bottom: 13px;
  border-bottom: 1px solid var(--line);
}

.reliability-row:last-child {
  padding-bottom: 0;
  border-bottom: 0;
}

.improvement-list {
  display: grid;
  gap: 14px;
}

.improvement-card,
.training-card {
  position: relative;
  padding: 18px;
  border-radius: 20px;
  background: linear-gradient(135deg, rgba(238, 112, 93, 0.12), rgba(240, 190, 73, 0.12));
  border: 1px solid rgba(238, 112, 93, 0.18);
}

.improvement-card h3,
.training-card h3 {
  margin-bottom: 8px;
}

.improvement-card p,
.training-card p {
  margin: 8px 0 0;
  color: var(--muted);
  line-height: 1.65;
}

.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.tag-row span,
.priority {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 4px 9px;
  border-radius: 999px;
  background: rgba(16, 35, 29, 0.08);
  color: var(--ink);
  font-size: 0.8rem;
  font-weight: 800;
}

.priority {
  color: #7b2c21;
  background: rgba(238, 112, 93, 0.2);
}

.training-list {
  counter-reset: training;
}

.training-card {
  counter-increment: training;
  background: rgba(14, 112, 74, 0.08);
  border-color: rgba(14, 112, 74, 0.14);
}

.training-card::before {
  content: counter(training, decimal-leading-zero);
  display: inline-grid;
  place-items: center;
  width: 38px;
  height: 38px;
  margin-bottom: 12px;
  border-radius: 14px;
  color: var(--paper);
  background: var(--court);
  font-weight: 900;
}

.events-table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 18px;
}

table {
  width: 100%;
  border-collapse: collapse;
  min-width: 720px;
}

th,
td {
  padding: 14px 15px;
  text-align: left;
  border-bottom: 1px solid var(--line);
}

th {
  color: var(--muted);
  font-size: 0.82rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  background: rgba(16, 35, 29, 0.045);
}

td {
  line-height: 1.5;
}

tr:last-child td {
  border-bottom: 0;
}

.status-passive {
  color: #a33b2d;
  font-weight: 800;
}

.status-active {
  color: #0e704a;
  font-weight: 800;
}

.report-footer {
  display: grid;
  gap: 8px;
  margin-top: 18px;
  padding: 18px 4px 0;
  color: var(--muted);
  font-size: 0.9rem;
  line-height: 1.6;
}

@media (max-width: 900px) {
  .hero,
  .grid-two,
  .grid-two-wide,
  .court-layout {
    grid-template-columns: 1fr;
  }

  .hero {
    padding: 30px;
  }

  .score-card {
    min-height: 220px;
  }

  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 560px) {
  .report-shell {
    width: min(100% - 20px, 1180px);
    padding-top: 10px;
  }

  .hero,
  .panel {
    border-radius: 22px;
    padding: 22px;
  }

  .metric-grid {
    grid-template-columns: 1fr;
  }

  .bar-line {
    grid-template-columns: 58px minmax(0, 1fr) 58px;
  }
}

@media (prefers-reduced-motion: no-preference) {
  .hero,
  .metric-card,
  .panel {
    animation: rise-in 520ms ease both;
  }

  .metric-card:nth-child(2),
  .grid-two:nth-of-type(2) .panel:first-child {
    animation-delay: 60ms;
  }

  .metric-card:nth-child(3),
  .grid-two:nth-of-type(2) .panel:last-child {
    animation-delay: 110ms;
  }

  .metric-card:nth-child(4) {
    animation-delay: 160ms;
  }
}

@keyframes rise-in {
  from {
    transform: translateY(16px);
    opacity: 0;
  }

  to {
    transform: translateY(0);
    opacity: 1;
  }
}

@media print {
  body {
    background: #ffffff;
  }

  .report-shell {
    width: 100%;
    padding: 0;
  }

  .hero,
  .panel,
  .metric-card {
    box-shadow: none;
    break-inside: avoid;
  }
}
"""


def _render_metric_card(item: object) -> str:
    payload = _as_mapping(item)
    return (
        '<article class="metric-card">'
        f'<span>{escape(str(payload.get("label", "")))}</span>'
        f'<strong>{escape(str(payload.get("value", "")))}</strong>'
        f'<small>{escape(str(payload.get("hint", "")))}</small>'
        "</article>"
    )


def _render_score_ring(score: int) -> str:
    value = _clamp(score, 0, 100)
    circumference = 2 * pi * 54
    dash = circumference * value / 100.0
    gap = circumference - dash
    return f"""
<svg class="score-ring" width="164" height="164" viewBox="0 0 164 164" role="img" aria-labelledby="scoreTitle scoreDesc">
  <title id="scoreTitle">综合评分 {value} 分</title>
  <desc id="scoreDesc">圆环表示当前羽毛球水平综合评分。</desc>
  <circle cx="82" cy="82" r="54" fill="none" stroke="rgba(248,255,242,0.18)" stroke-width="18"/>
  <circle cx="82" cy="82" r="54" fill="none" stroke="#b6e95e" stroke-width="18" stroke-linecap="round"
    stroke-dasharray="{dash:.2f} {gap:.2f}" transform="rotate(-90 82 82)"/>
  <text x="82" y="76" text-anchor="middle" font-size="46" font-weight="900" fill="#f8fff2">{value}</text>
  <text x="82" y="103" text-anchor="middle" font-size="14" font-weight="800" fill="rgba(248,255,242,0.75)">/100</text>
</svg>
"""


def _render_radar_svg(dimensions: Sequence[Mapping[str, Any]]) -> str:
    if not dimensions:
        return '<p class="table-note">暂无维度数据。</p>'
    size = 360
    center = size / 2
    radius = 118
    labels_radius = radius + 31
    axis_points = []
    value_points = []
    benchmark_points = []
    grid_polygons = []
    count = len(dimensions)

    for level in (0.2, 0.4, 0.6, 0.8, 1.0):
        points = [_radar_point(index, count, center, radius * level) for index in range(count)]
        grid_polygons.append(f'<polygon class="radar-grid" points="{_points_attr(points)}"/>')

    for index, item in enumerate(dimensions):
        axis_points.append(_radar_point(index, count, center, radius))
        value_points.append(_radar_point(index, count, center, radius * _clamp(_num(item.get("score")) / 100.0, 0, 1)))
        benchmark_points.append(_radar_point(index, count, center, radius * _clamp(_num(item.get("benchmark", 82)) / 100.0, 0, 1)))

    axis_lines = "\n".join(
        f'<line class="radar-axis" x1="{center:.1f}" y1="{center:.1f}" x2="{point[0]:.1f}" y2="{point[1]:.1f}"/>'
        for point in axis_points
    )
    labels = []
    for index, item in enumerate(dimensions):
        x, y = _radar_point(index, count, center, labels_radius)
        anchor = "middle"
        if x < center - 18:
            anchor = "end"
        elif x > center + 18:
            anchor = "start"
        labels.append(
            f'<text class="radar-label" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle">'
            f'{escape(str(item.get("name", "")))}</text>'
        )

    return f"""
<div class="radar-wrap">
  <svg width="100%" viewBox="0 0 {size} {size}" role="img" aria-labelledby="radarTitle radarDesc">
    <title id="radarTitle">能力维度图</title>
    <desc id="radarDesc">展示目标用户在步法、进攻、网前、转换、稳定性等维度的评分。</desc>
    {"".join(grid_polygons)}
    {axis_lines}
    <polygon class="radar-benchmark" points="{_points_attr(benchmark_points)}"/>
    <polygon class="radar-area" points="{_points_attr(value_points)}"/>
    {"".join(labels)}
  </svg>
</div>
"""


def _render_dimension_rows(dimensions: Sequence[Mapping[str, Any]]) -> str:
    rows = []
    for item in dimensions:
        score = _clamp(_num(item.get("score")), 0, 100)
        rows.append(
            '<div class="dimension-row">'
            f'<div class="row-head"><span>{escape(str(item.get("name", "")))}</span><span>{score:.0f}</span></div>'
            f'<div class="track"><div class="fill" style="--value:{score:.0f}%"></div></div>'
            f'<div class="row-note">{escape(str(item.get("note", "")))}</div>'
            "</div>"
        )
    return f'<div class="dimension-list">{"".join(rows)}</div>'


def _render_player_comparison(players: Mapping[str, Any]) -> str:
    user = _as_mapping(players.get("user"))
    opponent = _as_mapping(players.get("opponent"))
    metrics = [
        ("累计跑动", "distance_m", "m", 130.0),
        ("平均速度", "avg_speed_mps", "m/s", 2.2),
        ("最大速度", "max_speed_mps", "m/s", 5.5),
        ("启动次数", "start_count", "次", 42.0),
        ("急停次数", "stop_count", "次", 42.0),
        ("被动击球", "passive_hit_count", "次", 20.0),
    ]
    rows = []
    for label, key, unit, max_value in metrics:
        user_value = _num(user.get(key))
        opponent_value = _num(opponent.get(key))
        rows.append(
            '<div class="comparison-row">'
            f'<div class="row-head"><span>{label}</span><span>{user_value:.1f}{unit} / {opponent_value:.1f}{unit}</span></div>'
            '<div class="comparison-bars">'
            f'<div class="bar-line"><span class="bar-label">目标用户</span><div class="track"><div class="fill" style="--value:{_percent_of(user_value, max_value):.0f}%"></div></div><span>{user_value:.1f}{unit}</span></div>'
            f'<div class="bar-line"><span class="bar-label">对手</span><div class="track"><div class="fill fill-opponent" style="--value:{_percent_of(opponent_value, max_value):.0f}%"></div></div><span>{opponent_value:.1f}{unit}</span></div>'
            "</div>"
            "</div>"
        )
    return f'<div class="comparison-list">{"".join(rows)}</div>'


def _render_court_svg(points: Sequence[Any]) -> str:
    circles = []
    for item in points:
        payload = _as_mapping(item)
        x = _clamp(_num(payload.get("x")), 40, 570)
        y = _clamp(_num(payload.get("y")), 40, 1300)
        radius = _clamp(_num(payload.get("radius", 48)), 16, 84)
        opacity = _clamp(_num(payload.get("intensity", 0.75)), 0.2, 0.95)
        label = escape(str(payload.get("label", "站位热区")))
        circles.append(
            f'<circle class="heat" cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="rgba(238,112,93,{opacity:.2f})">'
            f"<title>{label}</title></circle>"
        )
    return f"""
<svg class="court-svg" viewBox="0 0 610 1340" role="img" aria-labelledby="courtTitle courtDesc">
  <title id="courtTitle">用户站位热区</title>
  <desc id="courtDesc">标准羽毛球场上的站位热点和击球集中区域。</desc>
  <rect x="18" y="18" width="574" height="1304" rx="28" fill="#0e704a"/>
  <rect class="court-line" x="44" y="44" width="522" height="1252" rx="2"/>
  <line class="court-inner-line" x1="44" y1="670" x2="566" y2="670"/>
  <line class="court-inner-line" x1="44" y1="242" x2="566" y2="242"/>
  <line class="court-inner-line" x1="44" y1="1098" x2="566" y2="1098"/>
  <line class="court-inner-line" x1="44" y1="470" x2="566" y2="470"/>
  <line class="court-inner-line" x1="44" y1="870" x2="566" y2="870"/>
  <line class="court-inner-line" x1="305" y1="44" x2="305" y2="470"/>
  <line class="court-inner-line" x1="305" y1="870" x2="305" y2="1296"/>
  <line class="court-inner-line" x1="92" y1="44" x2="92" y2="1296"/>
  <line class="court-inner-line" x1="518" y1="44" x2="518" y2="1296"/>
  {"".join(circles)}
</svg>
"""


def _render_zone_bars(zones: Sequence[Any]) -> str:
    total = sum(_num(_as_mapping(item).get("value")) for item in zones) or 1.0
    cards = []
    for item in zones:
        payload = _as_mapping(item)
        value = _num(payload.get("value"))
        percent = value / total * 100.0
        cards.append(
            '<div class="zone-card">'
            f'<div class="row-head"><span>{escape(str(payload.get("label", "")))}</span><span>{int(value)} 次</span></div>'
            f'<div class="track"><div class="fill" style="--value:{percent:.0f}%"></div></div>'
            "</div>"
        )
    return f'<div class="zone-stack">{"".join(cards)}</div>'


def _render_stroke_distribution(strokes: Sequence[Any]) -> str:
    cleaned = [_as_mapping(item) for item in strokes]
    max_value = max((_num(item.get("value")) for item in cleaned), default=1.0)
    rows = []
    for item in cleaned:
        value = _num(item.get("value"))
        rows.append(
            '<div class="stroke-row">'
            f'<span class="stroke-name">{escape(str(item.get("name", "")))}</span>'
            f'<div class="track"><div class="fill" style="--value:{_percent_of(value, max_value):.0f}%"></div></div>'
            f'<strong>{int(value)} 次</strong>'
            "</div>"
        )
    return f'<div class="stroke-list">{"".join(rows)}</div>' if rows else '<p class="table-note">暂无击球类型数据。</p>'


def _render_reliability(items: Sequence[Any]) -> str:
    rows = []
    for item in items:
        payload = _as_mapping(item)
        value = _clamp(_num(payload.get("value")) * 100.0, 0, 100)
        rows.append(
            '<div class="reliability-row">'
            f'<div class="row-head"><span>{escape(str(payload.get("label", "")))}</span><span>{value:.1f}%</span></div>'
            f'<div class="track"><div class="fill" style="--value:{value:.0f}%"></div></div>'
            f'<div class="row-note">{escape(str(payload.get("note", "")))}</div>'
            "</div>"
        )
    return f'<div class="reliability-list">{"".join(rows)}</div>'


def _render_improvements(items: Sequence[Any]) -> str:
    cards = []
    for item in items:
        payload = _as_mapping(item)
        tags = "".join(f"<span>{escape(str(tag))}</span>" for tag in _as_sequence(payload.get("tags")))
        cards.append(
            '<article class="improvement-card">'
            f'<span class="priority">{escape(str(payload.get("priority", "重点")))}</span>'
            f'<h3>{escape(str(payload.get("title", "")))}</h3>'
            f'<p>{escape(str(payload.get("evidence", "")))}</p>'
            f'<p>{escape(str(payload.get("suggestion", "")))}</p>'
            f'<div class="tag-row">{tags}</div>'
            "</article>"
        )
    return f'<div class="improvement-list">{"".join(cards)}</div>'


def _render_training_plan(items: Sequence[Any]) -> str:
    cards = []
    for item in items:
        payload = _as_mapping(item)
        cards.append(
            '<article class="training-card">'
            f'<h3>{escape(str(payload.get("title", "")))}</h3>'
            f'<p>{escape(str(payload.get("content", "")))}</p>'
            f'<div class="tag-row"><span>{escape(str(payload.get("frequency", "")))}</span><span>{escape(str(payload.get("target", "")))}</span></div>'
            "</article>"
        )
    return f'<div class="training-list">{"".join(cards)}</div>'


def _render_events_table(events: Sequence[Any]) -> str:
    rows = []
    for item in events:
        payload = _as_mapping(item)
        passive = bool(payload.get("passive"))
        rows.append(
            "<tr>"
            f"<td>{escape(str(payload.get('time', '')))}</td>"
            f"<td>{escape(str(payload.get('player_label', '')))}</td>"
            f"<td>{escape(str(payload.get('zone_label', '')))}</td>"
            f"<td>{escape(str(payload.get('stroke', '')))}</td>"
            f"<td>{_num(payload.get('confidence')) * 100:.1f}%</td>"
            f"<td class=\"{'status-passive' if passive else 'status-active'}\">{'被动' if passive else '主动'}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="6">暂无关键击球明细。</td></tr>')
    return (
        '<div class="events-table-wrap">'
        "<table>"
        "<thead><tr><th>时间</th><th>球员</th><th>区域</th><th>动作</th><th>置信度</th><th>状态</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        "</div>"
    )


def _dimension_scores(
    summary: Mapping[str, Any],
    user: Mapping[str, Any],
    reliability: Mapping[str, Any],
    strokes: Sequence[Mapping[str, Any]],
    zones: Mapping[str, int],
) -> list[dict[str, Any]]:
    hit_total = max(1, int(_num(user.get("hit_count")) or _num(summary.get("rally_hit_count")) or sum(zones.values())))
    passive_ratio = _num(user.get("passive_hit_count")) / hit_total
    front_ratio = zones["front"] / max(1, sum(zones.values()))
    back_ratio = zones["back"] / max(1, sum(zones.values()))
    smash_ratio = _stroke_ratio(strokes, "杀球")
    stroke_count = len([item for item in strokes if _num(item.get("value")) > 0])
    reliability_avg = sum(_num(reliability.get(key)) for key in RELIABILITY_LABELS) / 4.0

    footwork = 56 + min(13, _num(user.get("avg_speed_mps")) * 8.0) + min(13, _num(user.get("max_speed_mps")) * 2.2) + min(8, _num(user.get("high_intensity_count")) * 0.35) - min(9, passive_ratio * 18)
    back_attack = 58 + back_ratio * 18 + smash_ratio * 28 - passive_ratio * 10
    front_control = 54 + front_ratio * 46 + min(8, zones["front"] * 0.8)
    transition = 64 + min(8, _num(user.get("max_continuous_m")) * 0.9) - min(12, _num(user.get("stop_count")) * 0.23) - passive_ratio * 18
    variety = 50 + min(28, stroke_count * 4.8) + min(12, _zone_coverage_count(zones) * 4)
    stability = 42 + reliability_avg * 42 + _num(summary.get("hit_confidence_avg")) * 16 - min(8, _num(summary.get("out_of_frame_count")) * 2)

    return [
        {
            "name": "步法覆盖",
            "score": round(_clamp(footwork, 0, 100)),
            "benchmark": 84,
            "weight": 1.1,
            "note": "由平均速度、最大速度、高强度移动和被动击球比例综合估计。",
        },
        {
            "name": "后场进攻",
            "score": round(_clamp(back_attack, 0, 100)),
            "benchmark": 82,
            "weight": 1.0,
            "note": "后场击球占比不低，但连续压迫后的回位质量仍有提升空间。",
        },
        {
            "name": "网前控制",
            "score": round(_clamp(front_control, 0, 100)),
            "benchmark": 80,
            "weight": 0.95,
            "note": "根据前场击球次数与前场参与率估计，反映搓放、挑球和扑球机会。",
        },
        {
            "name": "攻防转换",
            "score": round(_clamp(transition, 0, 100)),
            "benchmark": 83,
            "weight": 1.0,
            "note": "急停、连续移动距离和被动击球比例会直接影响下一拍衔接。",
        },
        {
            "name": "击球多样性",
            "score": round(_clamp(variety, 0, 100)),
            "benchmark": 81,
            "weight": 0.85,
            "note": "由动作类别数量和前中后场覆盖度估计，类别越均衡越利于战术变化。",
        },
        {
            "name": "稳定性",
            "score": round(_clamp(stability, 0, 100)),
            "benchmark": 86,
            "weight": 1.05,
            "note": "融合球点可见率、姿态有效率、球场有效率和击球置信度。",
        },
    ]


def _build_improvements(
    dimensions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    user: Mapping[str, Any],
    reliability: Mapping[str, Any],
    zones: Mapping[str, int],
    strokes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scores = {str(item.get("name")): _num(item.get("score")) for item in dimensions}
    hit_total = max(1, int(_num(user.get("hit_count")) or _num(summary.get("rally_hit_count")) or sum(zones.values())))
    passive_hits = int(_num(user.get("passive_hit_count")))
    front_ratio = zones["front"] / max(1, sum(zones.values()))
    back_ratio = zones["back"] / max(1, sum(zones.values()))
    smash_count = int(next((_num(item.get("value")) for item in strokes if "杀" in str(item.get("name"))), 0))
    improvements = []

    if scores.get("攻防转换", 100) < 76 or passive_hits / hit_total > 0.24:
        improvements.append(
            {
                "priority": "优先级 1",
                "title": "后场回位后的第一步偏慢",
                "evidence": f"目标用户被动击球 {passive_hits} 次，占有效击球约 {passive_hits / hit_total * 100:.1f}%；急停 {int(_num(user.get('stop_count')))} 次，说明连续压迫后衔接会变紧。",
                "suggestion": "练习后场击球后向中线小跳回位，下一拍先找重心再启动，避免杀球或高远球后直接站死。",
                "tags": ["回位", "攻防转换", "第一步"],
            }
        )

    if front_ratio < 0.24 or scores.get("网前控制", 100) < 72:
        improvements.append(
            {
                "priority": "优先级 2",
                "title": "网前参与率偏低",
                "evidence": f"前场击球 {zones['front']} 次，占比 {front_ratio * 100:.1f}%；多数击球集中在中后场，主动上网终结机会不足。",
                "suggestion": "加入搓放后跟进、挑后封网和半场扑球练习，把对手回球从后场拉到网前处理。",
                "tags": ["网前", "搓放", "封网"],
            }
        )

    if back_ratio > 0.38 and smash_count >= 6:
        improvements.append(
            {
                "priority": "优先级 3",
                "title": "后场进攻质量需要配合落点变化",
                "evidence": f"后场击球 {zones['back']} 次，杀球 {smash_count} 次，说明进攻意图明确，但落点变化不足时容易被对手借力防起。",
                "suggestion": "后场不要只追求重杀，可用杀吊结合、点杀追身和高远压底线交替，减少同一区域连续两拍。",
                "tags": ["后场", "落点", "杀吊结合"],
            }
        )

    if sum(_num(reliability.get(key)) for key in RELIABILITY_LABELS) / 4.0 < 0.82:
        improvements.append(
            {
                "priority": "数据提示",
                "title": "拍摄角度仍可优化",
                "evidence": "球场有效率或姿态有效率未达到高可信区间，部分高速球与边线附近动作可能低估。",
                "suggestion": "建议使用固定机位，尽量拍完整双打边线和底线，保证球场白线清晰，帧率保持 50fps 或以上。",
                "tags": ["数据可信度", "机位", "帧率"],
            }
        )

    fallback = [
        {
            "priority": "巩固项",
            "title": "提升击球前准备节奏",
            "evidence": "平均击球间隔和启动次数显示，本次回合节奏已经较紧，下一阶段应减少多余小碎步。",
            "suggestion": "每拍结束后用一次分腿垫步统一启动时机，训练中以“击球后回位、对手击球时分腿”为口令。",
            "tags": ["节奏", "分腿垫步", "准备动作"],
        },
        {
            "priority": "巩固项",
            "title": "保持动作类型多样性",
            "evidence": "本次识别到多种击球类型，具备基本变化能力。",
            "suggestion": "继续保持高远、吊、杀、搓的组合，不要在连续得分或丢分后只依赖单一线路。",
            "tags": ["战术", "多拍", "变化"],
        },
    ]
    for item in fallback:
        if len(improvements) >= 3:
            break
        improvements.append(item)
    return improvements[:4]


def _training_plan(improvements: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    titles = [str(item.get("title", "")) for item in improvements]
    plan = []
    if any("后场" in title or "回位" in title for title in titles):
        plan.append(
            {
                "title": "后场两点回位",
                "content": "后场正手、头顶各 12 球为一组，击球后必须回到中线后再启动下一拍，重点看第一步是否干净。",
                "frequency": "每次 4 组",
                "target": "被动击球占比下降 5%",
            }
        )
    if any("网前" in title for title in titles):
        plan.append(
            {
                "title": "搓放跟进与封网",
                "content": "半场网前搓放 10 球后接 6 球扑推，训练从低重心过渡到前压封网的节奏。",
                "frequency": "每周 2 次",
                "target": "前场击球占比提升到 25%",
            }
        )
    plan.append(
        {
            "title": "杀吊结合线路",
            "content": "同一后场点位按“高远、吊、点杀、重杀”循环，要求落点覆盖直线、斜线和追身。",
            "frequency": "每次 15 分钟",
            "target": "后场进攻得分质量提升",
        }
    )
    plan.append(
        {
            "title": "固定机位复测",
            "content": "两周后用同一机位重新上传 3 段 60 秒训练视频，对比综合评分、被动击球和网前参与率。",
            "frequency": "两周后复测",
            "target": "报告指标可横向比较",
        }
    )
    return plan[:4]


def _takeaways(
    overall_score: int,
    dimensions: Sequence[Mapping[str, Any]],
    user: Mapping[str, Any],
    reliability: Mapping[str, Any],
) -> list[str]:
    weakest = min(dimensions, key=lambda item: _num(item.get("score")), default={"name": "攻防转换", "score": 0})
    strongest = max(dimensions, key=lambda item: _num(item.get("score")), default={"name": "稳定性", "score": 0})
    reliability_avg = sum(_num(reliability.get(key)) for key in RELIABILITY_LABELS) / 4.0
    return [
        f"综合评分 {overall_score}/100，当前更接近{_level_label(overall_score)}：已有稳定多拍能力，但还未完全进入比赛型高压节奏。",
        f"优势维度是{strongest.get('name')}（{_num(strongest.get('score')):.0f}分），说明本次视频中的基础输出较稳定。",
        f"短板维度是{weakest.get('name')}（{_num(weakest.get('score')):.0f}分），建议优先安排专项训练。",
        f"目标用户累计跑动 {_num(user.get('distance_m')):.1f}m，最大速度 {_num(user.get('max_speed_mps')):.2f}m/s，具备中等偏上的移动能力。",
        f"本次数据平均可信度约 {reliability_avg * 100:.1f}%，适合做训练复盘和趋势比较。",
    ]


def _player_report(player: Mapping[str, Any], fallback_label: str) -> dict[str, Any]:
    return {
        "label": str(player.get("label") or fallback_label),
        "distance_m": _num(player.get("distance_m")),
        "avg_speed_mps": _num(player.get("avg_speed_mps")),
        "max_speed_mps": _num(player.get("max_speed_mps")),
        "start_count": _num(player.get("start_count")),
        "stop_count": _num(player.get("stop_count")),
        "passive_hit_count": _num(player.get("passive_hit_count")),
        "hit_count": _num(player.get("hit_count")),
    }


def _court_heat_points(zones: Mapping[str, int], user_key: str) -> list[dict[str, Any]]:
    base_y = {"front": 760.0, "mid": 965.0, "back": 1215.0} if user_key == "bottom" else {"front": 580.0, "mid": 375.0, "back": 145.0}
    total = max(1, sum(zones.values()))
    points = []
    offsets = {"front": -78.0, "mid": 54.0, "back": 118.0}
    for key in ("front", "mid", "back"):
        value = zones[key]
        if value <= 0:
            continue
        intensity = 0.28 + value / total * 0.75
        points.append(
            {
                "x": 305.0 + offsets[key],
                "y": base_y[key],
                "radius": 34.0 + value / total * 70.0,
                "intensity": intensity,
                "label": f"{ZONE_LABELS[key]}热区 {value} 次",
            }
        )
    if not points:
        points.append({"x": 305.0, "y": 1000.0, "radius": 42.0, "intensity": 0.5, "label": "默认站位热区"})
    return points


def _reliability_items(reliability: Mapping[str, Any]) -> list[dict[str, Any]]:
    notes = {
        "ball_visible_rate": "高速小目标跟踪质量，低于 80% 时需谨慎解读落点。",
        "pose_valid_rate": "人体关键点可用比例，影响移动距离和步法判断。",
        "court_valid_rate": "球场线/单应性可用比例，影响标准场地坐标。",
        "avg_ball_confidence": "轨迹模型平均置信度，影响击球事件稳定性。",
    }
    return [
        {
            "label": label,
            "value": _clamp(_num(reliability.get(key)), 0, 1),
            "note": notes[key],
        }
        for key, label in RELIABILITY_LABELS.items()
    ]


def _hit_events(raw_hits: object, user_key: str) -> list[dict[str, Any]]:
    if not isinstance(raw_hits, list):
        return []
    events = []
    for hit in raw_hits[:10]:
        payload = _as_mapping(hit)
        player = str(payload.get("player", ""))
        player_label = str(payload.get("player_label") or ("目标用户" if player == user_key else PLAYER_LABELS.get(player, player)))
        zone = str(payload.get("zone", ""))
        events.append(
            {
                "time": _fmt_time_ms(payload.get("timestamp_ms")),
                "player_label": player_label,
                "zone_label": ZONE_LABELS.get(zone, zone or "未知"),
                "stroke": str(payload.get("stroke") or "轨迹击球"),
                "confidence": _num(payload.get("confidence") or payload.get("event_confidence")),
                "passive": bool(payload.get("passive")),
            }
        )
    return events


def _level_label(score: int | float) -> str:
    score = _num(score)
    if score >= 88:
        return "校队竞技级"
    if score >= 78:
        return "业余高阶"
    if score >= 68:
        return "业余进阶"
    if score >= 55:
        return "稳定入门"
    return "基础建立期"


def _level_note(score: int | float) -> str:
    score = _num(score)
    if score >= 88:
        return "具备较强比赛对抗能力，重点转向战术细节。"
    if score >= 78:
        return "技术框架较完整，短板主要在高压衔接。"
    if score >= 68:
        return "多拍能力已经成型，建议强化步法与落点变化。"
    if score >= 55:
        return "能完成基础回合，优先建立稳定动作模式。"
    return "先从握拍、发接发和基础移动开始。"


def _counter_items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    items = [
        {"name": str(key), "value": int(_num(item_value))}
        for key, item_value in value.items()
        if int(_num(item_value)) > 0
    ]
    return sorted(items, key=lambda item: (-int(item["value"]), str(item["name"])))


def _zone_counts(value: object) -> dict[str, int]:
    payload = _as_mapping(value)
    return {key: max(0, int(_num(payload.get(key)))) for key in ("front", "mid", "back")}


def _zone_coverage_count(zones: Mapping[str, int]) -> int:
    return sum(1 for value in zones.values() if value > 0)


def _stroke_ratio(strokes: Sequence[Mapping[str, Any]], keyword: str) -> float:
    total = sum(_num(item.get("value")) for item in strokes)
    if total <= 0:
        return 0.0
    matched = sum(_num(item.get("value")) for item in strokes if keyword in str(item.get("name", "")))
    return matched / total


def _radar_point(index: int, count: int, center: float, radius: float) -> tuple[float, float]:
    angle = -pi / 2 + (2 * pi * index / max(1, count))
    return center + cos(angle) * radius, center + sin(angle) * radius


def _points_attr(points: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _fmt_seconds(value: object) -> str:
    seconds = _num(value)
    if seconds >= 60:
        minutes = int(seconds // 60)
        rest = int(round(seconds - minutes * 60))
        return f"{minutes}分{rest:02d}秒"
    return f"{seconds:.1f}秒"


def _fmt_time_ms(value: object) -> str:
    seconds = max(0.0, _num(value) / 1000.0)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    if minutes:
        return f"{minutes}:{rest:05.2f}"
    return f"{rest:.2f}s"


def _percent_of(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return _clamp(value / max_value * 100.0, 0, 100)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(float(value), upper))


def _num(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    return number


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: object) -> Sequence[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return value
    return []
