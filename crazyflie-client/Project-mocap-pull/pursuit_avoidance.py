"""Fixed-altitude pursuit using the existing ellipsoidal velocity-obstacle test.

Search only feasible horizontal commands, rather than clipping a 3-D maneuver
later. Both measured and planned positions are checked. This is a sampled,
constant-velocity prediction, not a guarantee for arbitrary opponent motion.
"""
from pathlib import Path
import sys
import numpy as np
from match_config import CONFIG
from tracking import trailing_point

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'collision-avoidance'))
from collision_avoidance2 import VOParams, in_velocity_obstacle, normalized_separation


class PursuitAvoidance:
    def __init__(self, position, box_min=None, box_max=None, height=None,
                 period=None, max_step=None, follow_distance=None, config=CONFIG):
        self.config = config
        self.height = config.height if height is None else height
        self.period = config.period if period is None else period
        speed = config.horizontal_speed
        if max_step is not None:
            speed = min(speed, max_step / self.period)
        self.params = VOParams(R_i=config.radius, R_j=config.radius,
            margin=config.margin, tau=config.horizon, v_max=speed,
            a_max=config.horizontal_acceleration, dt=self.period,
            z_stretch=config.vertical_radius / config.radius)
        self.follow_distance = config.follow_distance if follow_distance is None else follow_distance
        self.lower = config.lower if box_min is None else np.asarray(box_min, float)
        self.upper = config.upper if box_max is None else np.asarray(box_max, float)
        if not self.lower[2] <= self.height <= self.upper[2]:
            raise ValueError('Tracking height is outside the flight box')
        self.sp = self.point(position)
        self.sp[2] = self.height
        if np.any(self.sp < self.lower) or np.any(self.sp > self.upper):
            raise ValueError('Initial command is outside the flight box')
        self.v_cmd = np.zeros(3)
        angles = np.linspace(0, 2*np.pi, config.candidate_directions, endpoint=False)
        self.directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
        self.last_reason = ''

    def point(self, value, velocity=False):
        value = np.asarray(value, float)
        return np.array([value[0], value[1], value[2] if len(value) >= 3
                         else (0.0 if velocity else self.height)])

    def step(self, position, velocity, runner, runner_velocity, direction):
        target = np.clip(trailing_point(runner, np.asarray(direction), self.follow_distance),
                         self.lower[:2], self.upper[:2])
        self.last_reason = ''
        if any(v is None or not np.all(np.isfinite(v))
               for v in (position, velocity, runner, runner_velocity, direction)):
            self.last_reason = 'Missing or invalid state'
            return self.sp.copy(), target, 'stop'
        pi, pj = self.point(position), self.point(runner)
        vi, vj = self.point(velocity, True), self.point(runner_velocity, True)
        if normalized_separation(pi, pj, self.params) <= 1.0:
            self.last_reason = 'Measured envelopes overlap or touch'
            return self.sp.copy(), target, 'stop'
        desired = vj + self.config.tracking_gain * (np.r_[target, self.height] - pi)
        desired[2] = 0
        # Establish a consistent passing side before the lagging plant reaches
        # the encounter. Both drones choose their own right in a head-on approach.
        relative = pj[:2] - pi[:2]
        distance = np.linalg.norm(relative)
        closing = np.dot(relative, vi[:2]-vj[:2]) > 0.0
        if (distance < self.config.passing_distance and closing
                and in_velocity_obstacle(desired[None, :], pi, pj, vj, self.params)[0]):
            right = np.array([relative[1], -relative[0], 0.0]) / max(distance, 1e-12)
            desired += self.config.passing_speed * right
        desired *= min(1., self.params.v_max / max(np.linalg.norm(desired), 1e-12))
        delta = desired - self.v_cmd
        radius = self.params.a_max * self.period
        preferred = self.v_cmd + delta * min(1., radius / max(np.linalg.norm(delta), 1e-12))
        radii = radius * np.arange(1, self.config.candidate_shells+1) / self.config.candidate_shells
        candidates = np.vstack((preferred, self.v_cmd, np.zeros(3),
            (radii[:, None, None] * self.directions).reshape(-1, 3) + self.v_cmd))
        next_sp = self.sp + candidates * self.period
        feasible = ((np.linalg.norm(candidates, axis=1) <= self.params.v_max + 1e-10)
            & (np.linalg.norm(candidates-self.v_cmd, axis=1) <= radius+1e-10)
            & np.all(next_sp >= self.lower-1e-10, axis=1)
            & np.all(next_sp <= self.upper+1e-10, axis=1)
            & (np.linalg.norm(next_sp-pi, axis=1) <= self.config.max_lead+1e-10))
        # Use the existing VO geometry from both the drone and the planned point.
        safe = feasible & ~in_velocity_obstacle(candidates, pi, pj, vj, self.params)
        safe &= ~in_velocity_obstacle(candidates, self.sp, pj, vj, self.params)
        # One-step ellipsoid guard, including the opponent's predicted displacement.
        safe &= normalized_separation(next_sp, pj+vj*self.period,
                                      self.params, include_margin=True) >= 1.0
        if not safe.any():
            self.last_reason = 'No feasible horizontal velocity outside the velocity obstacle'
            return self.sp.copy(), target, 'stop'
        # Small consistent right-side preference breaks head-on symmetry.
        toward = pj[:2] - pi[:2]
        right = np.array([toward[1], -toward[0], 0.0])
        right /= max(np.linalg.norm(right), 1e-12)
        cost = np.sum((candidates-desired)**2, axis=1) - self.config.side_preference*(candidates @ right)
        k = int(np.argmin(np.where(safe, cost, np.inf)))
        self.v_cmd = candidates[k].copy()
        self.sp = next_sp[k].copy()
        return self.sp.copy(), target, 'free' if k == 0 else 'avoiding'
