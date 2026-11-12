"""Shared pursuit calculations"""

import numpy as np


def segment_distance(start, end, point):
    """Shortest distance from a point to a line segment."""
    delta = end - start
    length_squared = float(delta @ delta)
    fraction = 0.0 if length_squared == 0 else np.clip(
        (point - start) @ delta / length_squared, 0.0, 1.0)
    return np.linalg.norm(start + fraction * delta - point)

def travel_direction(velocity, previous_direction, min_speed=0.01):
    """Keep the last heading when horizontal motion is too small to trust."""
    if velocity is not None:
        velocity = np.asarray(velocity[:2], dtype=float)
        speed = np.linalg.norm(velocity)
        if np.all(np.isfinite(velocity)) and speed >= min_speed:
            return velocity / speed
    return np.asarray(previous_direction, dtype=float).copy()


def trailing_point(p_runner, direction, follow_distance):
    """Point behind the runner along its travel direction."""
    return np.asarray(p_runner[:2]) - follow_distance * direction


def tracking_step(command, p_pursuer, p_runner, direction, follow_distance,
                  box_min, box_max, max_step, min_separation):
    """Return the next horizontal command, clipped target, and proximity flag."""
    target = np.clip(trailing_point(p_runner, direction, follow_distance), box_min[:2], box_max[:2])
    step = target - command[:2]
    step_length = np.linalg.norm(step)
    if step_length > max_step:
        step *= max_step / step_length
    p_next = command[:2] + step

    too_close = (
        segment_distance(p_pursuer, p_next, p_runner) < min_separation
        or segment_distance(command[:2], p_next, p_runner) < min_separation
    )
    return p_next, target, too_close
