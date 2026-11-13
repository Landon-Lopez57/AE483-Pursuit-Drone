"""
collision_avoidance.py

Velocity obstacle (VO) collision avoidance for two drones, plus a simulation
that needs no hardware. Put this file in the same folder as clients.py (the
ae483 package folder) so flight.py can import it:

    from ae483.collision_avoidance import VOParams, select_velocity

THEORY (matches the report)
---------------------------
Drone i is fully controlled. Drone j is not: it is assumed to keep its current
velocity. With

    p_rel = p_j - p_i        v_rel = v_i - v_j        p_rel(t) = p_rel - v_rel * t

a candidate velocity v_i is in the velocity obstacle VO_{i|j}^tau if

    exists t in [0, tau] such that ||p_rel - v_rel * t|| <= R_i + R_j.

The chosen velocity is the one closest to the desired velocity v_des that is
(a) reachable in one control step:   ||v - v_current|| <= a_max * dt
(b) not too fast:                    ||v|| <= v_max
(c) outside the VO:                  v not in VO_{i|j}^tau

HOW IT IS IMPLEMENTED
---------------------
* VO membership is tested with the first time ||p_rel(t)|| = R, found by
  solving a quadratic. This is exactly the "exists t in [0, tau]" condition,
  so no time-stepping is needed. A velocity is in the VO when that time is
  at most tau.
* The velocity is chosen by sampling candidates inside the reachable set
  (shells of a ball around v_current), discarding those in the VO, and taking
  the one closest to v_des. Sampling is simple, fast (a few hundred candidates,
  vectorized), and handles 3D without special cases.
* Every candidate is clipped to ||v|| <= v_max. Radial clipping onto a ball
  never moves a point farther from v_current (when v_current is inside the
  ball), so the acceleration limit (b) is still satisfied afterwards.
* If NO candidate is safe (the drone is cornered), the candidate with the
  longest time-to-collision is used. If the drones are already closer than R,
  the candidate that increases the separation most is used.
* z_stretch is the ratio of the vertical to the horizontal semi-axis of the
  envelope (0.5 in the report: flatter than wide). The VO test is done after
  dividing z by z_stretch, which makes the ellipsoid a sphere again.

USING IT IN FLIGHT (sketch, inside your control loop)
-----------------------------------------------------
    params = VOParams()
    pose_i, v_i, _ = mocap_client.get_latest(pursuer)
    pose_j, v_j, _ = mocap_client.get_latest(runner)
    if pose_i is None or pose_j is None:
        ...hold position...
    v_i = np.zeros(3) if v_i is None else v_i
    v_j = np.zeros(3) if v_j is None else v_j
    v_cmd, info = select_velocity(pose_i[:3], v_i, pose_j[:3], v_j, v_des, params)
    # then turn v_cmd into a position setpoint, e.g. pose_i[:3] + v_cmd * lookahead

All units are meters and seconds, the same as the mocap data.

RUNNING THE SIMULATION
----------------------
    python collision_avoidance.py                 # all scenarios, saves PNGs
    python collision_avoidance.py head_on --show  # one scenario, show windows
    python collision_avoidance.py chase --vel-lag 0.15 --z-stretch 1.5
"""

import argparse
import os
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np


###################################
# PARAMETERS

@dataclass
class VOParams:
    """
    Envelope: each drone has an axis-aligned ellipsoid with semi-axes a = b = R and
    c = z_stretch * R. The defaults are the values from the report: a = b = 0.0762 m
    (3 in) and c = 0.0381 m (1.5 in), so z_stretch = 0.5 (flatter than wide). For two
    identical aligned ellipsoids the Minkowski sum has semi-axes A = B = R_i + R_j and
    C = z_stretch * (R_i + R_j), i.e. 0.1524, 0.1524, 0.0762 m.
    """
    R_i: float = 0.0762     # horizontal semi-axis a = b of drone i's envelope (m)
    R_j: float = 0.0762     # horizontal semi-axis a = b of drone j's envelope (m)
    margin: float = 0.02    # extra buffer added to R_i + R_j, scaled like the envelope (m). PLACEHOLDER
    tau: float = 2.0        # prediction time horizon (s)
    v_max: float = 1.0      # max allowable speed of drone i (m/s)
    a_max: float = 2.0      # max allowable acceleration of drone i (m/s^2)
    dt: float = 0.05        # control period (s); 0.05 s = 20 Hz
    z_stretch: float = 0.5  # c / a: vertical semi-axis over horizontal semi-axis
    vz_weight: float = 1.0  # >1 makes the VO prefer sideways over up/down evasion
    n_dirs: int = 200       # candidate directions per shell
    n_radii: int = 4        # number of shells in the reachable set

    @property
    def pair_axes(self):
        """Semi-axes (A, B, C) of the pairwise envelope (q = 1), without the margin."""
        A = self.R_i + self.R_j
        return (A, A, self.z_stretch * A)

    @property
    def R(self):
        """Radius the VO test uses: R_i + R_j + margin."""
        return self.R_i + self.R_j + self.margin


###################################
# CORE ALGORITHM

def _clip_norm(v, max_norm):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n > max_norm and n > 0.0:
        return v * (max_norm / n)
    return v.copy()


def _clip_rows(V, max_norm):
    n = np.linalg.norm(V, axis=1, keepdims=True)
    scale = np.where(n > max_norm, max_norm / np.maximum(n, 1e-12), 1.0)
    return V * scale


def _scale_z(a, z_stretch):
    out = np.array(a, dtype=float, copy=True)
    out[..., 2] /= z_stretch
    return out


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)],
        axis=1,
    )


def time_to_collision(p_rel, v_rel, R):
    """
    First time t >= 0 at which ||p_rel - v_rel * t|| = R, for each row of v_rel.
    Returns 0 if already within R, and inf if there is no collision at any t >= 0.
    A velocity is in the VO with horizon tau exactly when this value is <= tau.
    """
    p = np.asarray(p_rel, dtype=float)
    v = np.atleast_2d(np.asarray(v_rel, dtype=float))
    c = float(p @ p) - R * R
    if c <= 0.0:
        return np.zeros(len(v))
    a = np.sum(v * v, axis=1)
    b = -2.0 * (v @ p)
    disc = b * b - 4.0 * a * c
    ok = (a > 1e-12) & (disc >= 0.0)
    t1 = np.full(len(v), np.inf)
    t1[ok] = (-b[ok] - np.sqrt(disc[ok])) / (2.0 * a[ok])
    return np.where(t1 >= 0.0, t1, np.inf)


def in_velocity_obstacle(v_i, p_i, p_j, v_j, P):
    """True/False for each row of v_i: is that velocity inside VO_{i|j}^tau?"""
    p_rel = _scale_z(np.asarray(p_j, float) - np.asarray(p_i, float), P.z_stretch)
    v_rel = _scale_z(np.atleast_2d(v_i) - np.asarray(v_j, float), P.z_stretch)
    return time_to_collision(p_rel, v_rel, P.R) <= P.tau


def _candidates(v_current, v_des, P):
    """Row 0 is the best velocity with no obstacle. All rows obey a_max and v_max."""
    r = P.a_max * P.dt
    dv = v_des - v_current
    n = np.linalg.norm(dv)
    v_target = v_current + (dv if n <= r else dv * (r / n))
    dirs = _fibonacci_sphere(P.n_dirs)
    radii = r * np.arange(1, P.n_radii + 1) / P.n_radii
    shells = (radii[:, None, None] * dirs[None, :, :]).reshape(-1, 3) + v_current
    C = np.vstack([v_target, v_current, shells])
    return _clip_rows(C, P.v_max)


def select_velocity(p_i, v_i, p_j, v_j, v_des, P):
    """
    Choose the velocity command for drone i.

    p_i, v_i : position and velocity of drone i (3-vectors)
    p_j, v_j : position and velocity of drone j (3-vectors), assumed constant
    v_des    : the velocity drone i would like to fly (3-vector)
    P        : VOParams

    Returns (v_cmd, info). info is a dict with
        'status'  : 'free'      desired velocity was already safe
                    'avoiding'  a different safe velocity was chosen
                    'cornered'  no safe velocity existed; best effort chosen
                    'inside'    already closer than R; moving apart
        'ttc'     : time to collision if v_cmd is held (inf if none)
        'ttc_des' : the same for the unobstructed choice (row 0)
    """
    p_i, v_i, p_j, v_j, v_des = (np.asarray(a, dtype=float) for a in (p_i, v_i, p_j, v_j, v_des))
    v_des = _clip_norm(v_des, P.v_max)

    C = _candidates(v_i, v_des, P)
    p_rel = _scale_z(p_j - p_i, P.z_stretch)
    v_rel = _scale_z(C - v_j, P.z_stretch)
    ttc = time_to_collision(p_rel, v_rel, P.R)
    safe = ttc > P.tau
    cost = np.sum(np.array([1.0, 1.0, P.vz_weight]) * (C - v_des) ** 2, axis=1)

    if safe[0]:
        k, status = 0, 'free'
    elif safe.any():
        k = int(np.argmin(np.where(safe, cost, np.inf)))
        status = 'avoiding'
    elif float(p_rel @ p_rel) <= P.R ** 2:
        # Already inside the exclusion zone: every velocity "collides" at t = 0,
        # so instead pick the one that leaves the drones farthest apart next step.
        after = np.linalg.norm(p_rel[None, :] - v_rel * P.dt, axis=1)
        k = int(np.argmax(after - 1e-6 * cost))
        status = 'inside'
    else:
        # Cornered: every option collides within tau. Delay the collision as long
        # as possible, breaking ties by closeness to v_des.
        k = int(np.argmax(ttc - 1e-6 * cost))
        status = 'cornered'

    return C[k].copy(), {'status': status, 'ttc': float(ttc[k]), 'ttc_des': float(ttc[0])}


###################################
# ELLIPSOIDAL ENVELOPE, TRAJECTORY, AND CONTROLLER FOR THE PASS-THROUGH TEST

def normalized_separation(p_i, p_j, P, include_margin=False):
    """
    Normalized separation q between two drones' ellipsoidal envelopes.

    With A = B = R_i + R_j and C = z_stretch * (R_i + R_j) (the Minkowski sum of the two
    envelopes) and r = p_i - p_j,

        q = sqrt((r_x / A)^2 + (r_y / B)^2 + (r_z / C)^2)

    q < 1 means the envelopes overlap, q = 1 means they touch. With include_margin=True
    the safety margin is added to R_i + R_j (that is the zone the controller defends).
    Works on single 3-vectors or on arrays of shape (N, 3).
    """
    R = P.R_i + P.R_j + (P.margin if include_margin else 0.0)
    d = _scale_z(np.asarray(p_j, float) - np.asarray(p_i, float), P.z_stretch)
    return np.linalg.norm(d, axis=-1) / R


def envelope_filter(p_plan, v_cmd, p_j, P):
    """
    Hard keep-out: if moving at v_cmd for one step would put the planned position
    inside the ellipsoid around drone j, project that position onto the ellipsoid
    surface and return the velocity that gets there. Returns (v_cmd, was_active).

    This is a purely geometric guard with no look-ahead. A drone that heads
    straight at the center of the ellipsoid is stopped at its surface and, if
    nothing pushes it sideways, stays there.
    """
    R = P.R
    r_next = _scale_z(p_plan + v_cmd * P.dt - p_j, P.z_stretch)
    n = np.linalg.norm(r_next)
    if n >= R:
        return v_cmd, False
    if n > 1e-9:
        u = r_next / n
    else:
        r_cur = _scale_z(p_plan - p_j, P.z_stretch)
        nc = np.linalg.norm(r_cur)
        u = r_cur / nc if nc > 1e-9 else np.array([1.0, 0.0, 0.0])
    d = u * R
    d[2] *= P.z_stretch
    v_new = (p_j + d - p_plan) / P.dt
    return _clip_norm(v_new, P.v_max), True


def extend_through(start, through, bounds=None):
    """
    End point of a straight path that starts at `start`, passes through `through`,
    and continues the same distance beyond it. If bounds = (xmin, xmax, ymin, ymax,
    zmin, zmax) is given, the part beyond `through` is shortened to stay inside.
    """
    start, through = np.asarray(start, float), np.asarray(through, float)
    d = through - start
    s = 1.0
    if bounds is not None:
        lo = np.array([bounds[0], bounds[2], bounds[4]])
        hi = np.array([bounds[1], bounds[3], bounds[5]])
        for k in range(3):
            if d[k] > 1e-9:
                s = min(s, (hi[k] - through[k]) / d[k])
            elif d[k] < -1e-9:
                s = min(s, (lo[k] - through[k]) / d[k])
        s = max(s, 0.0)
    return through + s * d


class LineTrajectory:
    """Straight line from p_start to p_end at constant speed, then hold at p_end."""

    def __init__(self, p_start, p_end, speed):
        self.p_start = np.asarray(p_start, float)
        self.p_end = np.asarray(p_end, float)
        self.speed = float(speed)
        d = self.p_end - self.p_start
        self.length = float(np.linalg.norm(d))
        self.u = d / self.length if self.length > 1e-9 else np.zeros(3)
        self.duration = self.length / self.speed if self.speed > 0 else 0.0

    def __call__(self, t):
        if self.length < 1e-9 or t >= self.duration:
            return self.p_end.copy(), np.zeros(3)
        s = max(t, 0.0) * self.speed
        return self.p_start + s * self.u, self.speed * self.u


@dataclass
class TestConfig:
    """Settings of the pass-through test (separate from the VO parameters)."""
    mode: str = 'both'              # 'none', 'vo', 'envelope', or 'both'
    v_nom: float = 0.5              # speed along the nominal path (m/s)
    track_gain: float = 1.5         # 1/s: pulls the plan back toward the nominal path
    max_lead: float = 0.35          # max distance the setpoint may get ahead of the drone (m)
    vel_deadband: float = 0.05      # speeds of drone j below this are treated as 0 (mocap noise)
    inflate_for_tracking_error: bool = True   # add |setpoint - drone| to the margin
    bounds: tuple = None            # (xmin, xmax, ymin, ymax, zmin, zmax) or None


class AvoidanceController:
    """
    Turns "follow this trajectory" into position setpoints that avoid drone j.
    The same object is used by the simulation and by flight_collision_avoidance.py.

    It keeps a plan (self.sp, a setpoint that moves at self.v_cmd). Each step:
      1. v_des steers the plan along the nominal trajectory.
      2. VO (if enabled) picks a safe velocity; otherwise v_des is only rate-limited.
      3. The ellipsoid filter (if enabled) projects the step out of the keep-out zone.
      4. The setpoint advances by v_cmd * dt, unless it would get too far ahead of
         the drone (then it waits), and is clipped to the bounds.
    Call step() once per control period and send the returned setpoint to the drone.
    """

    def __init__(self, P, cfg, traj, p0):
        self.P, self.cfg, self.traj = P, cfg, traj
        self.sp = np.asarray(p0, float).copy()
        self.v_cmd = np.zeros(3)

    def step(self, t, p_i, v_i, p_j, v_j):
        P, cfg = self.P, self.cfg
        p_i, v_i, p_j, v_j = (np.asarray(a, float) for a in (p_i, v_i, p_j, v_j))
        if np.linalg.norm(v_j) < cfg.vel_deadband:
            v_j = np.zeros(3)

        # Safety margin grows with the gap between setpoint and drone (tracking error)
        lead = float(np.linalg.norm(self.sp - p_i))
        P_eff = replace(P, margin=P.margin + (min(lead, cfg.max_lead) if cfg.inflate_for_tracking_error else 0.0))

        p_ref, v_ref = self.traj(t)
        v_des = _clip_norm(v_ref + cfg.track_gain * (p_ref - self.sp), P.v_max)

        status = 'free'
        if cfg.mode in ('vo', 'both'):
            v_cmd, info = select_velocity(self.sp, self.v_cmd, p_j, v_j, v_des, P_eff)
            status = info['status']
        else:
            dv = _clip_norm(v_des - self.v_cmd, P.a_max * P.dt)
            v_cmd = _clip_norm(self.v_cmd + dv, P.v_max)
        if cfg.mode in ('envelope', 'both'):
            v_cmd, active = envelope_filter(self.sp, v_cmd, p_j, P_eff)
            if active:
                status = 'envelope'

        sp_next = self.sp + v_cmd * P.dt
        lead_next = float(np.linalg.norm(sp_next - p_i))
        if lead_next > cfg.max_lead and lead_next > lead:
            sp_next, v_cmd = self.sp.copy(), np.zeros(3)       # drone is not keeping up: wait
        if cfg.bounds is not None:
            lo = np.array([cfg.bounds[0], cfg.bounds[2], cfg.bounds[4]])
            hi = np.array([cfg.bounds[1], cfg.bounds[3], cfg.bounds[5]])
            sp_next = np.clip(sp_next, lo, hi)
            v_cmd = (sp_next - self.sp) / P.dt
        self.sp, self.v_cmd = sp_next, v_cmd
        return self.sp.copy(), {'status': status, 'v_cmd': v_cmd.copy(), 'v_des': v_des,
                                'p_ref': p_ref, 'lead': lead}


def min_q_from_mocap(data_i, data_j, P):
    """
    Minimum normalized separation from logged mocap data (each a dict with 'x', 'y', 'z'
    lists, e.g. mocap_client.data[name]). Samples where either body was not visible
    (nan) are ignored. Returns (q_min, q_array).
    """
    n = min(len(data_i['x']), len(data_j['x']))
    pi = np.stack([np.asarray(data_i[k][:n], float) for k in 'xyz'], axis=1)
    pj = np.stack([np.asarray(data_j[k][:n], float) for k in 'xyz'], axis=1)
    q = normalized_separation(pi, pj, P)
    return (float(np.nanmin(q)) if np.any(np.isfinite(q)) else float('nan')), q


@dataclass
class PlantParams:
    """
    Stand-in for the drone's position controller: a second-order response
    a = wn^2 (setpoint - p) - 2 zeta wn v. wn = 5 rad/s is a GUESS; fit it to your
    flight data (step response of commanded vs. measured position).
    """
    wn: float = 5.0
    zeta: float = 1.0
    pos_noise: float = 0.0   # std of mocap position noise (m)
    vel_noise: float = 0.0   # std of mocap velocity noise (m/s)
    seed: int = 0


def simulate_pass_through(P, cfg, plant=None, start=(-1.5, 0.0, 0.5), runner=(0.0, 0.0, 0.5),
                          offset=(0.0, 0.0, 0.0), settle=3.0):
    """
    The pass-through test: the runner hovers at `runner`; the pursuer is commanded
    along a straight line from `start` through runner + offset and out the other side.
    Returns a log dict (arrays) including the normalized separation q.
    """
    plant = plant or PlantParams()
    rng = np.random.default_rng(plant.seed)
    start, runner = np.asarray(start, float), np.asarray(runner, float)
    through = runner + np.asarray(offset, float)
    end = start + 2.0 * (through - start)
    traj = LineTrajectory(start, end, cfg.v_nom)
    n = int(round((traj.duration + settle) / P.dt))
    n_sub = 5
    h = P.dt / n_sub

    ctrl = AvoidanceController(P, cfg, traj, p0=start)
    p, v = start.copy(), np.zeros(3)
    L = {k: [] for k in ['t', 'p_i', 'v_i', 'sp', 'v_cmd', 'v_des', 'p_ref', 'status']}
    for k in range(n):
        t = k * P.dt
        p_meas = p + rng.normal(0, plant.pos_noise, 3)
        v_meas = v + rng.normal(0, plant.vel_noise, 3)
        pj_meas = runner + rng.normal(0, plant.pos_noise, 3)
        vj_meas = rng.normal(0, plant.vel_noise, 3)
        sp, info = ctrl.step(t, p_meas, v_meas, pj_meas, vj_meas)
        for key, val in [('t', t), ('p_i', p), ('v_i', v), ('sp', sp), ('v_cmd', info['v_cmd']),
                         ('v_des', info['v_des']), ('p_ref', info['p_ref']), ('status', info['status'])]:
            L[key].append(val)
        for _ in range(n_sub):
            a = plant.wn ** 2 * (sp - p) - 2.0 * plant.zeta * plant.wn * v
            v = v + a * h
            p = p + v * h
    L = {key: np.array(val) for key, val in L.items()}
    L.update(p_j=runner, start=start, end=end, through=through, p_final=p.copy(), mode=cfg.mode)
    L['q'] = normalized_separation(L['p_i'], runner, P)
    L['q_zone'] = normalized_separation(L['p_i'], runner, P, include_margin=True)
    return L


def summarize_pass_through(L, P, finish_tol=0.15):
    """Pass/fail numbers for one run of simulate_pass_through."""
    k = int(np.argmin(L['q']))
    return {
        'mode': L['mode'],
        'q_min': float(L['q'][k]),
        't_q_min': float(L['t'][k]),
        'passed': bool(L['q'][k] >= 1.0),
        'finished_path': bool(np.linalg.norm(L['p_final'] - L['end']) < finish_tol),
        'final_error': float(np.linalg.norm(L['p_final'] - L['end'])),
        'pct_avoiding': float(np.mean(L['status'] != 'free') * 100.0),
        'max_speed': float(np.max(np.linalg.norm(L['v_i'], axis=1))),
    }


###################################
# SIMULATION

@dataclass
class Scenario:
    name: str
    description: str
    p_i0: np.ndarray
    runner: Callable        # t -> (p_j, v_j)
    desired: Callable       # (t, p_i, v_i, p_j, v_j) -> v_des
    T: float
    plot_axes: tuple = (0, 1)   # which two velocity components to show in the VO plot


def _const_runner(p0, v):
    p0, v = np.asarray(p0, float), np.asarray(v, float)
    return lambda t: (p0 + v * t, v.copy())


def _circle_runner(center, radius, speed, phase=0.0):
    center = np.asarray(center, float)
    w = speed / radius
    def f(t):
        th = phase + w * t
        p = center + radius * np.array([np.cos(th), np.sin(th), 0.0])
        v = speed * np.array([-np.sin(th), np.cos(th), 0.0])
        return p, v
    return f


def _goal_seek(goal, gain=1.0, v_cruise=0.8):
    goal = np.asarray(goal, float)
    return lambda t, p_i, v_i, p_j, v_j: _clip_norm(gain * (goal - p_i), v_cruise)


def _pursue(standoff=0.6, gain=1.0, v_cruise=0.9):
    """
    Aim at a point `standoff` meters from j on the line of sight, not at j itself.
    Aiming at j puts v_des inside the VO almost all the time, so the chosen
    velocity chatters along the edge of the exclusion zone. standoff should be
    larger than R_i + R_j + margin.
    """
    def f(t, p_i, v_i, p_j, v_j):
        los = p_j - p_i
        n = np.linalg.norm(los)
        target = p_j - standoff * los / n if n > 1e-9 else p_j
        return _clip_norm(gain * (target - p_i) + v_j, v_cruise)
    return f


def make_scenarios():
    return {
        'head_on': Scenario(
            'head_on', 'Drone i flies +x to a goal; drone j flies straight at it (-x).',
            np.array([-2.0, 0.0, 1.0]),
            _const_runner([2.0, 0.0, 1.0], [-0.5, 0.0, 0.0]),
            _goal_seek([2.0, 0.0, 1.0]), 8.0),
        'crossing': Scenario(
            'crossing', 'Drone i flies +x; drone j crosses its path flying +y.',
            np.array([-2.0, 0.0, 1.0]),
            _const_runner([0.0, -2.0, 1.0], [0.0, 0.7, 0.0]),
            _goal_seek([2.0, 0.0, 1.0]), 8.0),
        'chase': Scenario(
            'chase', 'Drone i pursues drone j, which flies a circle (not constant velocity).',
            np.array([-2.0, -2.0, 1.0]),
            _circle_runner([0.0, 0.0, 1.0], 1.0, 0.5),
            _pursue(), 20.0),
        'diagonal_3d': Scenario(
            'diagonal_3d', 'Drone i flies +x at constant altitude; drone j descends toward it.',
            np.array([-2.0, 0.0, 0.5]),
            _const_runner([2.0, 0.0, 1.5], [-0.5, 0.0, -0.25]),
            _goal_seek([2.0, 0.0, 0.5]), 8.0, plot_axes=(0, 2)),
    }


def simulate(scn, P, vel_lag=0.0):
    """
    Run one scenario. Drone i follows select_velocity; drone j follows its script.
    vel_lag = 0 means drone i achieves the commanded velocity instantly. A value
    like 0.1-0.2 s adds a first-order lag, which is closer to a real drone.
    """
    n = int(round(scn.T / P.dt))
    p_i = np.array(scn.p_i0, float)
    v_i = np.zeros(3)
    L = {k: [] for k in ['t', 'p_i', 'v_i', 'p_j', 'v_j', 'v_des', 'v_cmd', 'status', 'ttc_des']}
    for k in range(n):
        t = k * P.dt
        p_j, v_j = scn.runner(t)
        v_des = scn.desired(t, p_i, v_i, p_j, v_j)
        v_cmd, info = select_velocity(p_i, v_i, p_j, v_j, v_des, P)
        for key, val in [('t', t), ('p_i', p_i), ('v_i', v_i), ('p_j', p_j), ('v_j', v_j),
                         ('v_des', v_des), ('v_cmd', v_cmd), ('status', info['status']),
                         ('ttc_des', info['ttc_des'])]:
            L[key].append(val)
        v_new = v_i + (v_cmd - v_i) * (1.0 - np.exp(-P.dt / vel_lag)) if vel_lag > 0 else v_cmd
        p_i = p_i + 0.5 * (v_i + v_new) * P.dt
        v_i = v_new
    L = {k: np.array(v) for k, v in L.items()}
    p_rel = L['p_j'] - L['p_i']
    L['dist'] = np.linalg.norm(p_rel, axis=1)                              # Euclidean
    L['sep'] = np.linalg.norm(_scale_z(p_rel, P.z_stretch), axis=1)         # ellipsoid-normalized
    L['speed'] = np.linalg.norm(L['v_i'], axis=1)
    L['accel'] = np.r_[np.linalg.norm(np.diff(L['v_i'], axis=0), axis=1) / P.dt, np.nan]
    return L


def summarize(scn, P, L):
    k = int(np.argmin(L['sep']))
    q = L['sep'] / (P.R_i + P.R_j)            # normalized separation (envelopes touch at q = 1)
    avoiding = np.mean(L['status'] != 'free') * 100.0
    print(f'--- {scn.name}: {scn.description}')
    print(f'    min q             : {q.min():.2f} at t = {L["t"][k]:.2f} s '
          f'(envelopes overlap if q < 1: {"OK" if q.min() >= 1.0 else "OVERLAP"})')
    print(f'    min separation    : {L["sep"].min():.3f} (defended zone radius incl. margin {P.R:.3f} m: '
          f'{"OK" if L["sep"].min() >= P.R - 1e-3 else "ENTERED"}); min Euclidean distance {L["dist"].min():.3f} m')
    print(f'    max speed         : {L["speed"].max():.2f} m/s (limit {P.v_max:.2f})')
    print(f'    max acceleration  : {np.nanmax(L["accel"]):.2f} m/s^2 (limit {P.a_max:.2f})')
    print(f'    time avoiding     : {avoiding:.0f}% of steps; '
          f'cornered/inside: {int(np.sum(np.isin(L["status"], ["cornered", "inside"])))} steps')


###################################
# PLOTS

def plot_scenario(scn, P, L, path, show=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse

    t, pi, pj = L['t'], L['p_i'], L['p_j']
    active = L['status'] != 'free'
    snaps = np.linspace(0, len(t) - 1, 6).astype(int)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'{scn.name}: {scn.description}', fontsize=11)

    # Top-down view
    a = ax[0, 0]
    a.plot(pi[:, 0], pi[:, 1], 'C0', label='drone i (avoids)')
    a.plot(pj[:, 0], pj[:, 1], 'C1', label='drone j (not controlled)')
    a.plot(pi[active, 0], pi[active, 1], '.', color='red', ms=4, label='i is avoiding')
    for k in snaps:
        a.add_patch(Circle(pj[k, :2], P.R, fill=False, ls='--', ec='C1', alpha=0.6))
        a.plot(*pi[k, :2], 'o', color='C0')
        a.plot(*pj[k, :2], 'o', color='C1')
        a.annotate(f'{t[k]:.1f}s', pi[k, :2], textcoords='offset points', xytext=(4, 4), fontsize=7)
    a.set_xlabel('x (m)'); a.set_ylabel('y (m)'); a.set_title('Top view (dashed: exclusion zone of j)')
    a.set_aspect('equal', adjustable='datalim'); a.grid(alpha=0.3); a.legend(fontsize=8)

    # Side view
    a = ax[0, 1]
    a.plot(pi[:, 0], pi[:, 2], 'C0'); a.plot(pj[:, 0], pj[:, 2], 'C1')
    a.plot(pi[active, 0], pi[active, 2], '.', color='red', ms=4)
    for k in snaps:
        a.add_patch(Ellipse(pj[k, [0, 2]], 2 * P.R, 2 * P.R * P.z_stretch, fill=False,
                            ls='--', ec='C1', alpha=0.6))
        a.plot(pi[k, 0], pi[k, 2], 'o', color='C0'); a.plot(pj[k, 0], pj[k, 2], 'o', color='C1')
    a.set_xlabel('x (m)'); a.set_ylabel('z (m)'); a.set_title('Side view')
    a.set_aspect('equal', adjustable='datalim'); a.grid(alpha=0.3)

    # Separation
    a = ax[1, 0]
    a.plot(t, L['sep'], 'k', label='separation')
    a.axhline(P.R, color='C3', ls='--', label=f'zone radius R_i+R_j+margin = {P.R:.2f} m')
    a.axhline(P.R_i + P.R_j, color='C3', ls=':', label=f'collision radius R_i+R_j = {P.R_i + P.R_j:.2f} m')
    a.fill_between(t, 0, 1, where=active, transform=a.get_xaxis_transform(), color='red', alpha=0.1,
                   label='i is avoiding')
    a.set_xlabel('time (s)'); a.set_ylabel('separation (m)'); a.set_title('Separation vs. time')
    a.set_ylim(bottom=0); a.grid(alpha=0.3); a.legend(fontsize=8)

    # Speed and acceleration
    a = ax[1, 1]
    a.plot(t, L['speed'], 'C0', label='speed of i')
    a.axhline(P.v_max, color='C0', ls='--', alpha=0.6)
    a.set_xlabel('time (s)'); a.set_ylabel('speed (m/s)', color='C0'); a.grid(alpha=0.3)
    b = a.twinx()
    b.plot(t, L['accel'], 'C2', alpha=0.8, label='acceleration of i')
    b.axhline(P.a_max, color='C2', ls='--', alpha=0.6)
    b.set_ylabel('acceleration (m/s$^2$)', color='C2')
    a.set_title('Speed and acceleration (dashed: limits)')

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'    saved {path}')
    if show:
        plt.show()
    plt.close(fig)


def plot_velocity_space(scn, P, L, path, show=False, grid_n=241):
    """
    Velocity-space picture at one moment (like the schematic in the report):
    the VO is shaded, the reachable set and speed limit are circles, and the
    desired and chosen velocities are marked. The plot is a 2D slice of the 3D
    velocity space: the third component is held at drone i's current value.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    idx = np.flatnonzero(L['status'] == 'avoiding')
    k = 0 if len(idx) == 0 else min(idx[0] + int(0.5 / P.dt), len(L['t']) - 1)
    ax0, ax1 = scn.plot_axes
    third = 3 - ax0 - ax1
    p_i, v_i, p_j, v_j = L['p_i'][k], L['v_i'][k], L['p_j'][k], L['v_j'][k]

    lim = P.v_max * 1.2
    u = np.linspace(-lim, lim, grid_n)
    U, W = np.meshgrid(u, u)
    V = np.zeros((U.size, 3))
    V[:, ax0], V[:, ax1], V[:, third] = U.ravel(), W.ravel(), v_i[third]
    mask = in_velocity_obstacle(V, p_i, p_j, v_j, P).reshape(U.shape)

    fig, a = plt.subplots(figsize=(7, 7))
    a.contourf(U, W, mask.astype(float), levels=[0.5, 1.5], colors=['red'], alpha=0.25)
    a.contour(U, W, mask.astype(float), levels=[0.5], colors=['red'], linewidths=1)
    a.add_patch(Circle((0, 0), P.v_max, fill=False, ec='k', ls='--'))
    a.add_patch(Circle((v_i[ax0], v_i[ax1]), P.a_max * P.dt, fill=False, ec='C0'))
    a.plot(v_j[ax0], v_j[ax1], 'x', color='C1', ms=9, label='v_j (drone j)')
    a.plot(v_i[ax0], v_i[ax1], 'o', color='C0', label='v_i (current)')
    a.plot(L['v_des'][k][ax0], L['v_des'][k][ax1], '*', color='gold', mec='k', ms=14, label='v_des (desired)')
    a.plot(L['v_cmd'][k][ax0], L['v_cmd'][k][ax1], 's', color='green', label='v_cmd (chosen)')
    names = 'xyz'
    a.set_xlabel(f'v_{names[ax0]} (m/s)'); a.set_ylabel(f'v_{names[ax1]} (m/s)')
    a.set_title(f'{scn.name}: velocity space at t = {L["t"][k]:.2f} s\n'
                f'red = velocities of i inside VO (tau = {P.tau:g} s); '
                f'slice with v_{names[third]} = {v_i[third]:.2f}', fontsize=9)
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect('equal'); a.grid(alpha=0.3)
    a.legend(loc='lower left', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f'    saved {path}')
    if show:
        plt.show()
    plt.close(fig)


###################################
# MAIN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Velocity obstacle collision avoidance simulation')
    parser.add_argument('scenario', nargs='?', default='all',
                        help='head_on, crossing, chase, diagonal_3d, or all (default)')
    parser.add_argument('--vel-lag', type=float, default=0.0,
                        help='first-order lag of drone i velocity in seconds (default 0)')
    parser.add_argument('--z-stretch', type=float, default=None,
                        help='vertical/horizontal ratio of the envelope (default: VOParams default, 0.5)')
    parser.add_argument('--outdir', default='.', help='where to save figures')
    parser.add_argument('--show', action='store_true', help='open figure windows')
    args = parser.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use('Agg')

    P = VOParams() if args.z_stretch is None else VOParams(z_stretch=args.z_stretch)
    scenarios = make_scenarios()
    if args.scenario != 'all' and args.scenario not in scenarios:
        parser.error(f'unknown scenario "{args.scenario}"; choose from {list(scenarios)} or all')
    names = list(scenarios) if args.scenario == 'all' else [args.scenario]

    os.makedirs(args.outdir, exist_ok=True)
    print(f'R = {P.R:.2f} m, tau = {P.tau:g} s, v_max = {P.v_max:g} m/s, a_max = {P.a_max:g} m/s^2, '
          f'dt = {P.dt:g} s, vel_lag = {args.vel_lag:g} s, z_stretch = {P.z_stretch:g}')
    for name in names:
        scn = scenarios[name]
        L = simulate(scn, P, vel_lag=args.vel_lag)
        summarize(scn, P, L)
        plot_scenario(scn, P, L, os.path.join(args.outdir, f'ca_{name}.png'), show=args.show)
        plot_velocity_space(scn, P, L, os.path.join(args.outdir, f'ca_{name}_velocity_space.png'),
                            show=args.show)