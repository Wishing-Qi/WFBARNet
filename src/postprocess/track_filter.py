from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from src.utils.structures import TrackResult


Point = tuple[float, float]


@dataclass(slots=True)
class BallTrackFilterConfig:
    fps: float = 25.0
    min_confidence: float = 0.35
    relock_confidence: float = 0.50
    strong_relock_confidence: float = 0.85
    base_gate_px: float = 80.0
    max_gate_px: float = 360.0
    missed_gate_growth_px: float = 55.0
    max_speed_px_per_sec: float = 12000.0
    velocity_blend: float = 0.66
    inertia_min_speed_px_per_sec: float = 250.0
    max_accel_px_per_sec2: float = 120000.0
    max_lateral_error_px: float = 82.0
    max_reverse_px: float = 36.0
    max_coast_frames: int = 3
    min_coast_speed_px_per_sec: float = 450.0
    coast_velocity_decay: float = 0.82
    coast_score_decay: float = 0.55
    coast_on_outlier: bool = False
    relock_distance_px: float = 220.0
    relock_max_speed_px_per_sec: float = 9000.0
    relock_confirm_frames: int = 3
    relock_after_missed_frames: int = 2
    max_missed_frames: int = 8
    render_smoothing: float = 0.0


@dataclass(slots=True)
class _RelockCandidate:
    point: Point
    score: float
    count: int = 1


class BallTrackFilter:
    """Low-latency robust gate for shuttle detections.

    The predicted position is used for gating and short coasting. Missing
    detections can be filled briefly, but explicit outlier detections are hidden
    until they form a stable new trajectory.
    """

    def __init__(self, config: BallTrackFilterConfig | None = None, *, fps: float | None = None) -> None:
        self.config = config or BallTrackFilterConfig()
        if fps is not None and fps > 0:
            self.config.fps = float(fps)
        self._last_point: Point | None = None
        self._render_point: Point | None = None
        self._velocity: Point = (0.0, 0.0)
        self._missed_frames = 0
        self._coast_frames = 0
        self._locked = False
        self._candidate: _RelockCandidate | None = None

    def reset(self) -> None:
        self._last_point = None
        self._render_point = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._coast_frames = 0
        self._locked = False
        self._candidate = None

    def update(self, track: TrackResult, *, dt: float | None = None) -> TrackResult:
        step_dt = self._resolve_dt(dt)
        measurement = self._measurement(track)

        if measurement is None:
            return self._reject(track, step_dt, allow_coast=True)

        if not self._locked or self._last_point is None:
            return self._bootstrap(track, measurement, step_dt)

        if self._passes_gate(measurement, float(track.score), step_dt):
            return self._accept(track, measurement, step_dt)

        relock = self._update_candidate(measurement, float(track.score), step_dt)
        if relock and self._should_relock():
            self._drop_lock()
            return self._accept(track, measurement, step_dt)

        return self._reject(track, step_dt, allow_coast=self.config.coast_on_outlier)

    def _resolve_dt(self, dt: float | None) -> float:
        if dt is not None and dt > 0:
            return float(dt)
        fps = self.config.fps if self.config.fps > 0 else 25.0
        return 1.0 / fps

    def _measurement(self, track: TrackResult) -> Point | None:
        if not track.visible or float(track.score) < self.config.min_confidence or len(track.ball_xy) < 2:
            return None

        x, y = float(track.ball_xy[0]), float(track.ball_xy[1])
        if x < 0 or y < 0 or not isfinite(x) or not isfinite(y):
            return None
        return x, y

    def _bootstrap(self, track: TrackResult, measurement: Point, dt: float) -> TrackResult:
        if float(track.score) >= self.config.strong_relock_confidence:
            return self._accept(track, measurement, dt)

        if self._update_candidate(measurement, float(track.score), dt):
            return self._accept(track, measurement, dt)

        return self._invisible(track)

    def _passes_gate(self, measurement: Point, score: float, dt: float) -> bool:
        assert self._last_point is not None

        predicted = self._predict(dt)
        distance_to_prediction = _distance(measurement, predicted)
        distance_to_last = _distance(measurement, self._last_point)
        observed_speed = distance_to_last / max(dt * max(self._missed_frames + 1, 1), 1e-6)

        if observed_speed > self.config.max_speed_px_per_sec:
            return False

        velocity_px_per_frame = _length(self._velocity) * dt
        score_bonus = max(0.0, score - self.config.min_confidence) * 160.0
        allowed_distance = (
            self.config.base_gate_px
            + min(velocity_px_per_frame * 1.8, self.config.max_gate_px * 0.55)
            + self._missed_frames * self.config.missed_gate_growth_px
            + score_bonus
        )
        allowed_distance = min(max(allowed_distance, self.config.base_gate_px), self.config.max_gate_px)
        if distance_to_prediction > allowed_distance:
            return False

        return self._passes_inertia(measurement, score, dt)

    def _passes_inertia(self, measurement: Point, score: float, dt: float) -> bool:
        assert self._last_point is not None

        speed = _length(self._velocity)
        if speed < self.config.inertia_min_speed_px_per_sec:
            return True

        elapsed = max(dt * max(self._missed_frames + 1, 1), 1e-6)
        displacement = (
            measurement[0] - self._last_point[0],
            measurement[1] - self._last_point[1],
        )
        candidate_velocity = (displacement[0] / elapsed, displacement[1] / elapsed)
        acceleration = _distance(candidate_velocity, self._velocity) / elapsed
        if acceleration > self.config.max_accel_px_per_sec2:
            return False

        forward_px = _dot(displacement, self._velocity) / max(speed, 1e-6)
        if forward_px < -self.config.max_reverse_px:
            return False

        lateral_px = abs(displacement[0] * self._velocity[1] - displacement[1] * self._velocity[0]) / max(speed, 1e-6)
        expected_step_px = speed * elapsed
        score_bonus = max(0.0, score - self.config.min_confidence) * 35.0
        allowed_lateral = min(
            self.config.max_lateral_error_px,
            34.0 + expected_step_px * 0.45 + score_bonus,
        )
        return lateral_px <= allowed_lateral

    def _predict(self, dt: float) -> Point:
        assert self._last_point is not None
        frames = 1 if self._coast_frames > 0 else max(self._missed_frames + 1, 1)
        return (
            self._last_point[0] + self._velocity[0] * dt * frames,
            self._last_point[1] + self._velocity[1] * dt * frames,
        )

    def _accept(self, track: TrackResult, measurement: Point, dt: float) -> TrackResult:
        if self._last_point is not None:
            raw_velocity = (
                (measurement[0] - self._last_point[0]) / max(dt * max(self._missed_frames + 1, 1), 1e-6),
                (measurement[1] - self._last_point[1]) / max(dt * max(self._missed_frames + 1, 1), 1e-6),
            )
            blend = min(max(self.config.velocity_blend, 0.0), 1.0)
            self._velocity = (
                blend * raw_velocity[0] + (1.0 - blend) * self._velocity[0],
                blend * raw_velocity[1] + (1.0 - blend) * self._velocity[1],
            )
        else:
            self._velocity = (0.0, 0.0)

        self._last_point = measurement
        self._render_point = self._smooth_render_point(measurement)
        self._missed_frames = 0
        self._coast_frames = 0
        self._locked = True
        self._candidate = None
        return self._visible(track, self._render_point)

    def _reject(self, track: TrackResult, dt: float, *, allow_coast: bool) -> TrackResult:
        if allow_coast and self._can_coast():
            return self._coast(track, dt)

        self._missed_frames += 1
        if self._missed_frames > self.config.max_missed_frames:
            self._drop_lock()
        return self._invisible(track)

    def _can_coast(self) -> bool:
        return (
            self._locked
            and self._last_point is not None
            and self._coast_frames < self.config.max_coast_frames
            and _length(self._velocity) >= self.config.min_coast_speed_px_per_sec
        )

    def _coast(self, track: TrackResult, dt: float) -> TrackResult:
        assert self._last_point is not None

        predicted = (
            self._last_point[0] + self._velocity[0] * dt,
            self._last_point[1] + self._velocity[1] * dt,
        )
        self._last_point = predicted
        self._render_point = self._smooth_render_point(predicted)
        self._velocity = (
            self._velocity[0] * self.config.coast_velocity_decay,
            self._velocity[1] * self.config.coast_velocity_decay,
        )
        self._missed_frames += 1
        self._coast_frames += 1
        score = float(track.score) * (self.config.coast_score_decay ** self._coast_frames)
        return TrackResult(
            ball_xy=[float(self._render_point[0]), float(self._render_point[1])],
            visible=1,
            score=max(0.0, score),
            heatmap_shape=list(track.heatmap_shape),
        )

    def _update_candidate(self, measurement: Point, score: float, dt: float) -> bool:
        relock_distance = max(
            self.config.relock_distance_px,
            self.config.relock_max_speed_px_per_sec * max(dt, 1e-6),
        )
        if self._candidate is None or _distance(measurement, self._candidate.point) > relock_distance:
            self._candidate = _RelockCandidate(point=measurement, score=score)
        else:
            self._candidate.point = (
                0.35 * self._candidate.point[0] + 0.65 * measurement[0],
                0.35 * self._candidate.point[1] + 0.65 * measurement[1],
            )
            self._candidate.score = max(self._candidate.score, score)
            self._candidate.count += 1

        return (
            self._candidate.count >= self.config.relock_confirm_frames
            and self._candidate.score >= self.config.relock_confidence
        )

    def _should_relock(self) -> bool:
        if self._candidate is None:
            return False
        if self._candidate.score >= self.config.strong_relock_confidence:
            return True
        return self._missed_frames >= self.config.relock_after_missed_frames

    def _drop_lock(self) -> None:
        self._locked = False
        self._last_point = None
        self._render_point = None
        self._velocity = (0.0, 0.0)
        self._missed_frames = 0
        self._coast_frames = 0
        self._candidate = None

    def _smooth_render_point(self, measurement: Point) -> Point:
        smoothing = min(max(self.config.render_smoothing, 0.0), 0.85)
        if self._render_point is None or smoothing <= 0.0:
            return measurement
        return (
            smoothing * self._render_point[0] + (1.0 - smoothing) * measurement[0],
            smoothing * self._render_point[1] + (1.0 - smoothing) * measurement[1],
        )

    def _visible(self, original: TrackResult, point: Point) -> TrackResult:
        return TrackResult(
            ball_xy=[float(point[0]), float(point[1])],
            visible=1,
            score=float(original.score),
            heatmap_shape=list(original.heatmap_shape),
        )

    def _invisible(self, original: TrackResult) -> TrackResult:
        return TrackResult(
            ball_xy=[-1.0, -1.0],
            visible=0,
            score=float(original.score),
            heatmap_shape=list(original.heatmap_shape),
        )


def _distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def _length(v: Point) -> float:
    return hypot(v[0], v[1])


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def filter_track_results(tracks: list[TrackResult], *, fps: float = 25.0) -> list[TrackResult]:
    tracker = BallTrackFilter(fps=fps)
    return [tracker.update(track) for track in tracks]
