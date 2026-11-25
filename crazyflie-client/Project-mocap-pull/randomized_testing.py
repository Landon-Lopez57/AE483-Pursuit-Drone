"""
randomized_testing.py

Seeded Monte Carlo testing for the pursuit + collision-avoidance simulation in
virtual_testing.ipynb. Put this file in the same folder as match_config.py, tracking.py
and pursuit_avoidance.py.

WHAT IS RANDOMIZED (see RandomizationSpec; every range can be changed or pinned)
  * pursuer start position
  * runner behavior (acceleration-limited, so it cannot reverse or turn instantly): stationary,
    straight line (bouncing off the walls), circle, random
    polygon, random waypoints, figure eight, a runner aimed at the pursuer's start
    ("approach"), and two reactive runners that steer relative to the pursuer
    ("charge" runs at it, "flee" runs away)
  * how the simulated drone responds to commands (first-order lag, response_time)
  * sensing: mocap latency, position noise, velocity noise
  * mocap dropouts: random windows during which one body's data stop refreshing

REPRODUCIBILITY
  A trial is fully determined by one integer seed: sample_trial(seed) draws the scenario
  and simulate_trial(trial) draws the sensor noise from a second stream tied to the same seed.
  run_batch(n, seed=...) derives the n trial seeds from the master seed, so a whole batch
  is reproducible, and any single trial can be replayed with replay(seed).

WHAT IT MEASURES
  Safety:      q_min < 1 (envelopes overlapped), entering the margin zone, commands outside the
               flight box or above the step limit
  Performance: unexpected stops (an "Avoidance stop" that was not caused by a dropout), the
               fraction of time the pursuer is in the scoring zone (ASSUMED definition below)

The simulation core mirrors simulate() in virtual_testing.ipynb. With latency = 0, no noise and a
single dropout that never ends, it reproduces that function; the additions are the randomized
parts above. One deliberate difference: during a pursuer data dropout the pursuer's velocity
measurement is frozen along with its position (the notebook keeps updating it).
"""
from dataclasses import dataclass, field, asdict
import numpy as np

from match_config import CONFIG
from tracking import travel_direction, trailing_point
import pursuit_avoidance as _pa
from pursuit_avoidance import PursuitAvoidance

normalized_separation = _pa.normalized_separation     # q, from collision_avoidance2


###################################
# WHAT TO RANDOMIZE

@dataclass
class RandomizationSpec:
    """
    Ranges are (low, high) and sampled uniformly. Pin a value with (x, x). The defaults are
    ASSUMPTIONS (the plant lag, latency, and noise have not been measured on your drone):
    narrow them once you have data.
    """
    runner_kinds: dict = field(default_factory=lambda: {
        'stationary': 1, 'line': 2, 'circle': 1, 'polygon': 1, 'waypoints': 2,
        'figure_eight': 1, 'approach': 2, 'charge': 1, 'flee': 1})   # relative weights
    runner_speed: tuple = (0.05, 0.5)       # m/s (pursuer limit is config.horizontal_speed)
    runner_accel: float = 1.0               # m/s^2: the runner cannot reverse or turn faster than this
    response_time: tuple = (0.25, 0.8)      # s, first-order lag of the simulated drone
    latency: tuple = (0.0, 0.08)            # s, delay of mocap measurements
    pos_noise: tuple = (0.0, 0.004)         # m, std of mocap position noise
    vel_noise: tuple = (0.0, 0.05)          # m/s, std of noise on the velocities the controller sees
    dropout_prob: float = 0.25              # probability that a trial has dropouts at all
    dropout_count: tuple = (1, 2)           # number of dropout windows if it does
    dropout_duration: tuple = (0.05, 0.6)   # s
    dropout_start: tuple = (1.0, 25.0)      # s, when a window may begin
    min_start_separation: float = 0.6       # m, initial pursuer-runner distance
    wall_clearance: float = 0.1             # m, runner and pursuer start this far inside the flight box
    duration: float = 30.0                  # s, report baseline pursuit trial duration


# Presets. STRESS (the default of RandomizationSpec) uses wide, pessimistic ranges for lag and noise.
# NOMINAL assumes a drone no slower than the notebook's 0.5 s and sub-millimeter-to-2 mm mocap noise.
# Neither is measured: replace them with numbers from your own flight data.
STRESS = RandomizationSpec()
NOMINAL = RandomizationSpec(response_time=(0.2, 0.5), latency=(0.0, 0.05), pos_noise=(0.0, 0.002), vel_noise=(0.0, 0.02))


@dataclass
class Trial:
    """Everything needed to rebuild one scenario. Plain numbers and lists only."""
    seed: int
    kind: str
    pursuer0: list
    runner_params: dict
    response_time: float
    latency: float
    pos_noise: float
    vel_noise: float
    dropouts: list          # [[body, t_start, t_end], ...]
    duration: float


###################################
# RUNNER MODELS  (each is called as runner(t, p_pursuer) -> 2-vector)

def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def _fold(x, lo, hi):
    """Reflect x into [lo, hi] (a runner that bounces off the walls)."""
    L = hi - lo
    y = np.mod(x - lo, 2 * L)
    return lo + L - np.abs(y - L)


class _Stationary:
    def __init__(self, p): self.p = np.array(p, float)
    def __call__(self, t, p_pursuer): return self.p.copy()


class _Line:
    def __init__(self, p0, v, lo, hi):
        self.p0, self.v, self.lo, self.hi = np.array(p0, float), np.array(v, float), np.array(lo, float), np.array(hi, float)
    def __call__(self, t, p_pursuer): return _fold(self.p0 + self.v * t, self.lo, self.hi)


class _Circle:
    def __init__(self, center, radius, omega, phase):
        self.c, self.r, self.w, self.ph = np.array(center, float), radius, omega, phase
    def __call__(self, t, p_pursuer):
        a = self.ph + self.w * t
        return self.c + self.r * np.array([np.cos(a), np.sin(a)])


class _FigureEight:
    def __init__(self, center, A, B, omega, phase):
        self.c, self.A, self.B, self.w, self.ph = np.array(center, float), A, B, omega, phase
    def __call__(self, t, p_pursuer):
        a = self.w * t + self.ph
        return self.c + np.array([self.A * np.sin(a), self.B * np.sin(2 * a)])


class _Polyline:
    """Closed loop through the points at constant speed."""
    def __init__(self, points, speed):
        pts = np.array(points, float)
        self.pts = np.vstack([pts, pts[:1]])
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.cum = np.r_[0.0, np.cumsum(seg)]
        self.total, self.speed = self.cum[-1], speed
    def __call__(self, t, p_pursuer):
        s = (self.speed * t) % self.total
        i = min(int(np.searchsorted(self.cum, s, side='right')) - 1, len(self.pts) - 2)
        seg = self.cum[i + 1] - self.cum[i]
        f = 0.0 if seg < 1e-12 else (s - self.cum[i]) / seg
        return self.pts[i] + f * (self.pts[i + 1] - self.pts[i])


class _Smoothed:
    """
    Follows a path with limited acceleration, so the runner cannot reverse or turn instantly
    (an instant reversal at a wall or a polygon corner is not something a real drone can do, and
    it creates collisions that no pursuer could avoid).
    """
    def __init__(self, base, accel, lo, hi, gain=3.0, v_cap=0.7):
        self.base, self.accel, self.gain, self.v_cap = base, accel, gain, v_cap
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        self.p = None; self.v = np.zeros(2); self.t_prev = None; self.ref_prev = None
    def __call__(self, t, p_pursuer):
        ref = np.asarray(self.base(t, p_pursuer), float)
        if self.p is None:
            self.p, self.ref_prev, self.t_prev = ref.copy(), ref, t
            return self.p.copy()
        dt = t - self.t_prev
        if dt > 0:
            v_des = (ref - self.ref_prev) / dt + self.gain * (ref - self.p)
            n = np.linalg.norm(v_des)
            if n > self.v_cap:
                v_des *= self.v_cap / n
            dv = v_des - self.v
            n = np.linalg.norm(dv)
            if n > self.accel * dt:
                dv *= self.accel * dt / n
            self.v = self.v + dv
            self.p = np.clip(self.p + self.v * dt, self.lo, self.hi)
            self.ref_prev, self.t_prev = ref, t
        return self.p.copy()


class _Reactive:
    """
    Kinematic runner that steers relative to the pursuer: 'charge' runs at it, 'flee' runs away
    (drifting toward the arena center so it does not just pin itself against a wall).
    Speed and acceleration limited; stays inside [lo, hi]. This is a crude opponent model.
    """
    def __init__(self, p0, speed, accel, mode, lo, hi):
        self.p, self.v = np.array(p0, float), np.zeros(2)
        self.speed, self.accel, self.mode = speed, accel, mode
        self.lo, self.hi = np.array(lo, float), np.array(hi, float)
        self.center, self.half = 0.5 * (self.lo + self.hi), 0.5 * (self.hi - self.lo)
        self.t_prev = None
    def __call__(self, t, p_pursuer):
        dt = 0.0 if self.t_prev is None else t - self.t_prev
        self.t_prev = t
        if dt > 0:
            u = _unit(self.p - np.asarray(p_pursuer, float))            # away from the pursuer
            if self.mode == 'charge':
                u = -u
            else:
                u = _unit(u + 0.5 * (self.center - self.p) / self.half)
            dv = self.speed * u - self.v
            n = np.linalg.norm(dv)
            if n > self.accel * dt:
                dv *= self.accel * dt / n
            self.v = self.v + dv
            self.p = np.clip(self.p + self.v * dt, self.lo, self.hi)
        return self.p.copy()


def make_runner(kind, params, config=CONFIG, wall_clearance=0.1):
    lo, hi = config.lower[:2] + wall_clearance, config.upper[:2] - wall_clearance
    P = params
    if kind == 'stationary':
        return _Stationary(P['p'])
    if kind in ('charge', 'flee'):
        return _Reactive(P['p0'], P['speed'], P['accel'], kind, lo, hi)
    if kind in ('line', 'approach'):
        base = _Line(P['p0'], P['v'], lo, hi)
    elif kind == 'circle':
        base = _Circle(P['center'], P['radius'], P['omega'], P['phase'])
    elif kind == 'figure_eight':
        base = _FigureEight(P['center'], P['A'], P['B'], P['omega'], P['phase'])
    elif kind in ('polygon', 'waypoints'):
        base = _Polyline(P['points'], P['speed'])
    else:
        raise ValueError(f'unknown runner kind {kind!r}')
    return _Smoothed(base, P['accel'], lo, hi) if P.get('accel', 0) > 0 else base   # no 'accel' = raw path


###################################
# SAMPLING

def _U(rng, rng_pair):
    return float(rng.uniform(rng_pair[0], rng_pair[1]))


def _sample_runner(kind, rng, spec, lo, hi, pursuer0):
    params = _sample_runner_path(kind, rng, spec, lo, hi, pursuer0)
    if kind not in ('stationary', 'charge', 'flee'):
        params['accel'] = spec.runner_accel
    return params


def _sample_runner_path(kind, rng, spec, lo, hi, pursuer0):
    speed = _U(rng, spec.runner_speed)
    if kind == 'stationary':
        return {'p': rng.uniform(lo, hi).tolist()}
    if kind == 'line':
        a = rng.uniform(0, 2 * np.pi)
        return {'p0': rng.uniform(lo, hi).tolist(), 'v': (speed * np.array([np.cos(a), np.sin(a)])).tolist()}
    if kind == 'approach':
        p0 = rng.uniform(lo, hi)
        d = _unit(np.asarray(pursuer0) - p0)
        a = np.arctan2(d[1], d[0]) + rng.normal(0, 0.1)         # aimed at where the pursuer starts
        return {'p0': p0.tolist(), 'v': (speed * np.array([np.cos(a), np.sin(a)])).tolist()}
    if kind == 'circle':
        r = rng.uniform(0.2, 0.7)
        c = rng.uniform(lo + r, hi - r)
        return {'center': c.tolist(), 'radius': float(r), 'omega': float(rng.choice([-1, 1]) * speed / r),
                'phase': float(rng.uniform(0, 2 * np.pi))}
    if kind == 'figure_eight':
        A, B = rng.uniform(0.3, 0.8), rng.uniform(0.2, 0.5)
        c = rng.uniform(lo + [A, B], hi - [A, B])
        return {'center': c.tolist(), 'A': float(A), 'B': float(B), 'omega': float(speed / (A + 2 * B)),
                'phase': float(rng.uniform(0, 2 * np.pi))}
    if kind == 'polygon':
        n, R = int(rng.integers(3, 6)), rng.uniform(0.3, 0.8)
        c = rng.uniform(lo + R, hi - R)
        ang = np.sort(rng.uniform(0, 2 * np.pi, n))
        pts = c + R * np.column_stack([np.cos(ang), np.sin(ang)])
        return {'points': pts.tolist(), 'speed': speed}
    if kind == 'waypoints':
        n = int(rng.integers(3, 8))
        return {'points': rng.uniform(lo, hi, size=(n, 2)).tolist(), 'speed': speed}
    if kind in ('charge', 'flee'):
        return {'p0': rng.uniform(lo, hi).tolist(), 'speed': speed, 'accel': spec.runner_accel}
    raise ValueError(kind)


def sample_trial(seed, spec=None, config=CONFIG):
    """Draw one scenario. The same seed and spec always give the same trial."""
    spec = spec or RandomizationSpec()
    rng = np.random.default_rng([int(seed), 0])
    lo, hi = config.lower[:2] + spec.wall_clearance, config.upper[:2] - spec.wall_clearance
    kinds = list(spec.runner_kinds)
    weights = np.array([spec.runner_kinds[k] for k in kinds], float)
    kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
    pursuer0 = rng.uniform(lo, hi)
    for _ in range(500):
        params = _sample_runner(kind, rng, spec, lo, hi, pursuer0)
        p0 = make_runner(kind, params, config, spec.wall_clearance)(0.0, pursuer0)
        if np.linalg.norm(p0 - pursuer0) >= spec.min_start_separation:
            break
    else:
        raise RuntimeError('could not place the runner far enough from the pursuer; relax min_start_separation')
    dropouts = []
    if rng.uniform() < spec.dropout_prob:
        for _ in range(int(rng.integers(spec.dropout_count[0], spec.dropout_count[1] + 1))):
            t0 = _U(rng, spec.dropout_start)
            dropouts.append([str(rng.choice(['pursuer', 'runner'])), t0, t0 + _U(rng, spec.dropout_duration)])
    return Trial(seed=int(seed), kind=kind, pursuer0=pursuer0.tolist(), runner_params=params,
                 response_time=_U(rng, spec.response_time), latency=_U(rng, spec.latency),
                 pos_noise=_U(rng, spec.pos_noise), vel_noise=_U(rng, spec.vel_noise),
                 dropouts=dropouts, duration=spec.duration)


###################################
# SIMULATION  (mirrors simulate() in virtual_testing.ipynb)

def simulate_trial(trial, config=CONFIG, pursuer_max_age=0.3, runner_max_age=0.2, wall_clearance=0.1,
                   runner_velocity_deadband=0.0):
    """
    runner_velocity_deadband (m/s): if > 0, a runner velocity estimate smaller than this is treated as zero
    before it reaches travel_direction() and the avoidance step. 0 reproduces the current flight.py.
    """
    period, height = config.period, config.height
    max_step = config.horizontal_speed * period
    rng = np.random.default_rng([trial.seed, 1])            # sensor noise stream
    runner = make_runner(trial.kind, trial.runner_params, config, wall_clearance)

    p_pursuer = np.array(trial.pursuer0, float)
    command = np.array([*p_pursuer, height])
    avoidance = PursuitAvoidance(command, config.lower, config.upper, height, period, max_step,
                                 config.follow_distance, config=config)
    params = avoidance.params
    lag_steps = int(round(trial.latency / period))
    response_fraction = 1.0 - np.exp(-period / trial.response_time)

    measured_pursuer = p_pursuer.copy()
    measured_runner = np.array(runner(0.0, p_pursuer), float)
    runner_velocity = np.zeros(2)
    pursuer_velocity = np.zeros(2)            # true, from the plant
    measured_pursuer_velocity = np.zeros(2)   # what the controller sees (frozen in a dropout)
    runner_direction = _unit(measured_runner - p_pursuer)
    last_update = {'pursuer': 0.0, 'runner': 0.0}
    hist_p, hist_r = [], []
    records, reason, stop_time, stop_reason = [], 'Completed', trial.duration, ''

    for k, t in enumerate(np.arange(0.0, trial.duration, period)):
        p_runner = np.array(runner(t, p_pursuer), float)
        hist_p.append(p_pursuer.copy()); hist_r.append(p_runner.copy())
        i = max(0, len(hist_p) - 1 - lag_steps)             # mocap latency
        seen_p, seen_r = hist_p[i], hist_r[i]
        for body in ('pursuer', 'runner'):
            missing = any(b == body and t0 <= t < t1 for b, t0, t1 in trial.dropouts)
            if missing:
                continue
            previous_update, last_update[body] = last_update[body], t
            if body == 'pursuer':
                measured_pursuer = seen_p + rng.normal(0, trial.pos_noise, 2)
                measured_pursuer_velocity = pursuer_velocity.copy()
            else:
                new = seen_r + rng.normal(0, trial.pos_noise, 2)
                dt = t - previous_update
                if dt > 0:
                    velocity = (new - measured_runner) / dt
                    runner_velocity = 0.3 * velocity + 0.7 * runner_velocity
                measured_runner = new

        if t - last_update['pursuer'] > pursuer_max_age or t - last_update['runner'] > runner_max_age:
            reason, stop_time, stop_reason = 'Stale motion capture', t, 'Stale motion capture'
            break

        rv_in = runner_velocity + rng.normal(0, trial.vel_noise, 2)
        if runner_velocity_deadband > 0 and np.linalg.norm(rv_in) < runner_velocity_deadband:
            rv_in = np.zeros(2)
        pv_in = measured_pursuer_velocity + rng.normal(0, trial.vel_noise, 2)
        runner_direction = travel_direction(rv_in, runner_direction)
        follow_point = trailing_point(measured_runner, runner_direction, config.follow_distance)
        next_command, target, status = avoidance.step(measured_pursuer, pv_in, measured_runner, rv_in,
                                                      runner_direction)
        if status == 'stop':
            reason, stop_time, stop_reason = 'Avoidance stop', t, avoidance.last_reason
            break
        previous_command = command[:2].copy()
        command[:] = next_command
        pi, pj = np.array([*p_pursuer, height]), np.array([*p_runner, height])
        records.append({
            'time': t, 'status': status, 'runner': p_runner.copy(), 'pursuer': p_pursuer.copy(),
            'command': command[:2].copy(), 'target': target.copy(), 'follow_point': follow_point.copy(),
            'direction': runner_direction.copy(), 'step': np.linalg.norm(command[:2] - previous_command),
            'separation': np.linalg.norm(p_runner - p_pursuer), 'error': np.linalg.norm(target - p_pursuer),
            'q': float(normalized_separation(pi, pj, params)),
            'q_zone': float(normalized_separation(pi, pj, params, include_margin=True)),
        })
        displacement = response_fraction * (command[:2] - p_pursuer)
        pursuer_velocity = displacement / period
        p_pursuer = p_pursuer + displacement

    keys2 = ('runner', 'pursuer', 'command', 'target', 'follow_point', 'direction')
    keys1 = ('time', 'step', 'separation', 'error', 'q', 'q_zone', 'status')
    result = {key: (np.array([r[key] for r in records]) if records else np.zeros((0, 2))) for key in keys2}
    result.update({key: (np.array([r[key] for r in records]) if records else np.zeros(0)) for key in keys1})
    result.update(reason=reason, stop_time=float(stop_time), stop_reason=stop_reason, trial=trial, params=params)
    return result


###################################
# METRICS

def summarize_trial(trial, result, config=CONFIG, pursuer_max_age=0.3, runner_max_age=0.2):
    period = config.period
    n_total = int(round(trial.duration / period))
    n = len(result['time'])
    q = result['q'] if n else np.array([np.inf])
    qz = result['q_zone'] if n else np.array([np.inf])
    max_age = {'pursuer': pursuer_max_age, 'runner': runner_max_age}
    dropout_related = any(t0 <= result['stop_time'] <= t1 + max_age[b] + period for b, t0, t1 in trial.dropouts)
    stopped = result['reason'] != 'Completed'
    box_ok = (not n) or bool(np.all(result['command'] >= config.lower[:2] - 1e-9)
                             and np.all(result['command'] <= config.upper[:2] + 1e-9))
    step_ok = (not n) or bool(np.all(result['step'] <= config.horizontal_speed * period + 1e-9))
    score = 0.0
    if n:   # ASSUMED scoring zone: behind the runner, distance in [scoring_min, scoring_max], within scoring_angle
        rel = result['pursuer'] - result['runner']
        d = np.maximum(np.linalg.norm(rel, axis=1), 1e-12)
        cosang = np.clip(np.sum(rel * -result['direction'], axis=1) / d, -1, 1)
        in_zone = ((d >= config.scoring_min) & (d <= config.scoring_max)
                   & (np.degrees(np.arccos(cosang)) <= config.scoring_angle_degrees))
        score = float(in_zone.sum() / n_total)
    return {
        'seed': trial.seed, 'kind': trial.kind, 'reason': result['reason'], 'stop_reason': result['stop_reason'],
        'stop_time': result['stop_time'],
        'q_min': float(q.min()), 'q_zone_min': float(qz.min()),
        'collided': bool(q.min() < 1.0), 'zone_entered': bool(qz.min() < 1.0),
        'dropout_related': bool(stopped and dropout_related),
        'unexpected_stop': bool(stopped and not dropout_related),
        'command_ok': bool(box_ok and step_ok),
        'score_fraction': score,
        'mean_error': float(result['error'].mean()) if n else float('nan'),
        # fraction of steps where the chosen command was not the controller's preferred one (its status is
        # 'avoiding' for any other candidate, so this is high even when nothing is close)
        'frac_not_preferred': float(np.mean(result['status'] != 'free')) if n else 0.0,
        'response_time': trial.response_time, 'latency': trial.latency,
        'pos_noise': trial.pos_noise, 'vel_noise': trial.vel_noise,
        'n_dropouts': len(trial.dropouts), 'start': trial.pursuer0,
    }


def run_batch(n, seed=0, spec=None, config=CONFIG, progress=True, **sim_kwargs):
    """
    Run n random trials. Returns a list of summary dicts (each has its seed for replay).
    Extra keyword arguments (e.g. runner_velocity_deadband=0.1) go to simulate_trial; pass a modified
    config (dataclasses.replace(CONFIG, max_lead=0.6)) to test a different controller setting.
    """
    spec = spec or RandomizationSpec()
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]
    rows = []
    for i, s in enumerate(seeds):
        trial = sample_trial(s, spec, config)
        rows.append(summarize_trial(trial, simulate_trial(trial, config, wall_clearance=spec.wall_clearance, **sim_kwargs), config))
        if progress and (i + 1) % max(1, n // 10) == 0:
            print(f'  {i + 1}/{n} trials', flush=True)
    return rows


def replay(seed, spec=None, config=CONFIG, **sim_kwargs):
    """Re-run one trial from its seed. Returns (trial, result). Use with the same spec as the batch."""
    spec = spec or RandomizationSpec()
    trial = sample_trial(seed, spec, config)
    return trial, simulate_trial(trial, config, wall_clearance=spec.wall_clearance, **sim_kwargs)


def rule_of_three(n):
    """If 0 failures were seen in n independent trials, the failure rate is below about 3/n (95% confidence)."""
    return 3.0 / n


def summary_markdown(rows):
    """Table of outcomes by runner kind (plus an overall row) as markdown."""
    kinds = sorted({r['kind'] for r in rows})
    head = ('| Runner | Trials | Collisions (q<1) | Margin zone entered | Unexpected stops | Dropout stops | '
            'Min q (worst) | Min q (5th pct) | Score zone (mean %) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    def row(name, rs):
        q = np.array([r['q_min'] for r in rs])
        return (f"| {name} | {len(rs)} | {sum(r['collided'] for r in rs)} | {sum(r['zone_entered'] for r in rs)} | "
                f"{sum(r['unexpected_stop'] for r in rs)} | {sum(r['dropout_related'] for r in rs)} | "
                f"{q.min():.2f} | {np.percentile(q, 5):.2f} | {100 * np.mean([r['score_fraction'] for r in rs]):.1f} |")
    lines = [head] + [row(k, [r for r in rows if r['kind'] == k]) for k in kinds] + [row('**All**', rows)]
    return '\n'.join(lines)


###################################
# PLOTTING HELPER

def plot_trial(result, trial, ax=None, config=CONFIG, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    lo, hi = config.lower[:2], config.upper[:2]
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ls='--', color='gray'))
    if len(result['time']):
        r, p = result['runner'], result['pursuer']
        ax.plot(r[:, 0], r[:, 1], color='tab:blue', lw=1.5, label='runner')
        ax.plot(p[:, 0], p[:, 1], color='tab:red', lw=1.5, label='pursuer')
        ax.plot(*r[0], 'o', color='tab:blue'); ax.plot(*p[0], 'o', color='tab:red')
        ax.plot(*r[-1], 'x', color='tab:blue', ms=9); ax.plot(*p[-1], 'x', color='tab:red', ms=9)
        k = int(np.argmin(result['q']))                     # closest approach, with the envelope drawn
        A = result['params'].R_i + result['params'].R_j
        ax.add_patch(Circle(r[k], A, fill=False, ec='k', lw=1.2))
        ax.add_patch(Circle(r[k], result['params'].R, fill=False, ec='k', ls='--', lw=0.8))
        ax.plot(*p[k], 's', color='tab:red', mfc='none', ms=7)
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=0.2)
    ax.set_xlim(lo[0] - 0.1, hi[0] + 0.1); ax.set_ylim(lo[1] - 0.1, hi[1] + 0.1)
    qmin = result['q'].min() if len(result['time']) else float('nan')
    ax.set_title(title or f"{trial.kind}  seed {trial.seed}\nq_min {qmin:.2f}  {result['reason']} @ {result['stop_time']:.1f}s", fontsize=8)
    return ax
