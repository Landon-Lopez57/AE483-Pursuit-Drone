"""Milestone 2 operating limits"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class MatchConfig:
    arena_half_width: float = 2.0
    boundary_inset: float = 0.15
    altitude_min: float = 0.5
    altitude_max: float = 1.0
    height: float = 0.5
    horizontal_speed: float = 0.6
    vertical_speed: float = 0.3
    horizontal_acceleration: float = 1.5
    vertical_acceleration: float = 0.5
    radius: float = 0.0762
    vertical_radius: float = 0.0381
    follow_distance: float = 0.5
    duration: float = 10.0
    scoring_min: float = 0.3
    scoring_max: float = 0.6
    scoring_angle_degrees: float = 15.0
    # Tuning assumptions, not measured hardware limits or report requirements.
    period: float = 0.05
    horizon: float = 2.0
    margin: float = 0.02
    tracking_gain: float = 1.5
    passing_distance: float = 1.5
    passing_speed: float = 0.4
    side_preference: float = 0.015
    max_lead: float = 0.35
    velocity_filter_time: float = 0.28
    heading_min_speed: float = 0.01
    candidate_directions: int = 180
    candidate_shells: int = 6

    @property
    def lower(self):
        b = self.arena_half_width - self.boundary_inset
        return np.array([-b, -b, self.altitude_min])

    @property
    def upper(self):
        b = self.arena_half_width - self.boundary_inset
        return np.array([b, b, self.altitude_max])


CONFIG = MatchConfig()
