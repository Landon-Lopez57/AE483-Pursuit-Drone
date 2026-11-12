"""Horizontal avoidance shared by flight code and simulations."""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class AvoidanceSettings:
    min_separation: float = 0.1
    margin: float = 0.08
    trigger_distance: float = 0.45
    release_distance: float = 0.55
    horizon: float = 0.8
    response_time: float = 0.5
    samples: int = 32

    def __post_init__(self):
        if not (0 < self.min_separation < self.min_separation + self.margin
                < self.trigger_distance < self.release_distance):
            raise ValueError('Expected separation < buffered separation < trigger < release')
        if self.horizon <= 0 or self.response_time <= 0 or self.samples < 8:
            raise ValueError('Use positive times and at least eight direction samples')


@dataclass
class AvoidanceState:
    active: bool = False
    right: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0]))


def closest_separation(relative_position, relative_velocity, horizon):
    """Closest predicted distance while both velocities stay constant."""
    speed_squared = relative_velocity @ relative_velocity
    t = 0.0 if speed_squared < 1e-12 else np.clip(
        -(relative_position @ relative_velocity) / speed_squared, 0.0, horizon)
    return np.linalg.norm(relative_position + t * relative_velocity)


def avoidance_step(position, velocity, previous_command, desired_command, neighbors,
                   box_min, box_max, max_step, state, settings=None):
    """Return (command, status), updating this drone's avoidance state.

    Neighbors are (horizontal position, horizontal velocity) pairs.
    Each drone needs its own state. Prediction assumes constant neighbor velocity
    and a first-order response to commands; it is not a flight safety guarantee.
    """
    cfg = settings or AvoidanceSettings()
    position, velocity, previous, desired, lower, upper = [
        np.asarray(value, dtype=float)[:2] for value in
        (position, velocity, previous_command, desired_command, box_min, box_max)]
    neighbors = [(np.asarray(p, dtype=float)[:2], np.asarray(v, dtype=float)[:2])
                 for p, v in neighbors]
    vectors = [position, velocity, previous, desired, lower, upper]
    vectors += [value for pair in neighbors for value in pair]
    if any(v.shape != (2,) or not np.all(np.isfinite(v)) for v in vectors):
        return previous.copy(), 'stop'
    if max_step <= 0 or np.any(lower > upper):
        raise ValueError('Invalid command limits')

    def limit(point):
        delta = np.clip(point, lower, upper) - previous
        length = np.linalg.norm(delta)
        return previous + delta * min(1.0, max_step / length) if length else previous.copy()

    desired = limit(desired)
    if not neighbors:
        state.active = False
        return desired, 'tracking'
    distances = [np.linalg.norm(p - position) for p, _ in neighbors]
    if min(distances) <= cfg.min_separation:
        return previous.copy(), 'stop'

    def clearance(command):
        # Approximate motion toward each candidate command.
        response_velocity = (command - position) / cfg.response_time
        return min(closest_separation(p - position, v - own_velocity, cfg.horizon)
                   for p, v in neighbors for own_velocity in (response_velocity,))

    buffer = cfg.min_separation + cfg.margin
    current_clearance = min(closest_separation(p - position, v - velocity, cfg.horizon)
                            for p, v in neighbors)
    danger = (clearance(desired) < buffer or current_clearance < buffer
              or min(distances) < cfg.trigger_distance)
    if not state.active and not danger:
        return desired, 'tracking'
    if state.active and min(distances) > cfg.release_distance and clearance(desired) >= buffer and current_clearance >= buffer:
        state.active = False
        return desired, 'tracking'
    if not state.active:
        forward = velocity if np.linalg.norm(velocity) > 0.01 else desired - position
        if np.linalg.norm(forward) < 1e-9:
            forward = neighbors[0][0] - position
        forward = forward / np.linalg.norm(forward)
        state.right = np.array([forward[1], -forward[0]])
        state.active = True

    # Choose a bounded step. Prefer passing right, with an outward push nearby.
    outward = sum(((position - p) / max(np.linalg.norm(position - p), 1e-9)
                   * max(0.0, cfg.trigger_distance - np.linalg.norm(position - p))
                   / cfg.trigger_distance for p, _ in neighbors), np.zeros(2))
    preferred = state.right + 2.0 * outward
    preferred /= max(np.linalg.norm(preferred), 1e-9)
    candidates = [desired, limit(position), previous.copy()]
    for angle in np.linspace(0, 2*np.pi, cfg.samples, endpoint=False):
        direction = np.array([np.cos(angle), np.sin(angle)])
        candidates.append(limit(previous + max_step * direction))
    safe = [candidate for candidate in candidates
            if np.all(candidate >= lower) and np.all(candidate <= upper)
            and clearance(candidate) >= buffer]
    if not safe:
        return previous.copy(), 'stop'

    def score(candidate):
        movement = (candidate - previous) / max_step
        return movement @ preferred - 0.25 * np.linalg.norm(candidate - desired) / max_step

    return max(safe, key=score).copy(), 'avoiding'
