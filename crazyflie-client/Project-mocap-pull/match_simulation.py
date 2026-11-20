"""Deterministic, simultaneous two-drone match with a second-order plant."""
import numpy as np
from match_config import CONFIG
from pursuit_avoidance import PursuitAvoidance, normalized_separation
from tracking import travel_direction, trailing_point


def scoring_mask(positions, headings, config=CONFIG):
    relative = positions - positions[::-1]
    distances = np.linalg.norm(relative, axis=1)
    rear = -headings[::-1]
    cosine = np.sum(relative[:, :2]*rear, axis=1) / np.maximum(
        np.linalg.norm(relative[:, :2], axis=1), 1e-12)
    return ((distances >= config.scoring_min) & (distances <= config.scoring_max)
            & (cosine >= np.cos(np.deg2rad(config.scoring_angle_degrees))))


def simulate_match(config=CONFIG, starts=None, initial_headings=None):
    positions = np.array(starts if starts is not None else
                         [[-1.1, 0.65, config.height], [1.1, -0.65, config.height]], float)
    velocities = np.zeros((2, 3))
    heading = positions[::-1, :2]-positions[:, :2]
    if initial_headings is not None:
        heading = np.asarray(initial_headings, float).copy()
    heading /= np.linalg.norm(heading, axis=1)[:, None]
    controllers = [PursuitAvoidance(p, config=config) for p in positions]
    scores = np.zeros(2)
    log = {k: [] for k in ['time', 'positions', 'velocities', 'commands',
                         'command_velocity', 'headings', 'targets', 'scores', 'statuses', 'q']}
    min_q = np.inf
    outcome = 'Time limit'
    alpha = 1-np.exp(-config.period/config.velocity_filter_time)
    filtered = velocities.copy()
    for frame in range(round(config.duration/config.period)+1):
        t = frame*config.period
        heading = np.array([travel_direction(filtered[i], heading[i], config.heading_min_speed)
                            for i in range(2)])
        targets = np.array([trailing_point(positions[1-i], heading[1-i], config.follow_distance)
                            for i in range(2)])
        commands, statuses = [], []
        if frame < round(config.duration/config.period):
            for i in range(2):
                command, _, status = controllers[i].step(positions[i], filtered[i],
                    positions[1-i], filtered[1-i], heading[1-i])
                commands.append(command)
                statuses.append(status)
        else:
            commands = [c.sp.copy() for c in controllers]
            statuses = ['finished']*2
        q = float(normalized_separation(positions[0], positions[1], controllers[0].params))
        min_q = min(min_q, q)
        for key, value in [('time', t), ('positions', positions.copy()),
                ('velocities', velocities.copy()), ('commands', np.array(commands)),
                ('command_velocity', np.array([c.v_cmd for c in controllers])),
                ('headings', heading.copy()), ('targets', targets), ('scores', scores.copy()),
                ('statuses', statuses), ('q', q)]:
            log[key].append(value)
        if 'stop' in statuses:
            outcome = '; '.join(f'Drone {i+1}: {controllers[i].last_reason}'
                                for i in range(2) if statuses[i]=='stop')
            break
        if frame == round(config.duration/config.period):
            break
        # Integrate at 100 Hz; score occupied time and inspect separation between frames.
        h = config.period/10
        for _ in range(10):
            acceleration = 25*(np.array(commands)-positions)-10*velocities
            velocities += acceleration*h
            positions += velocities*h
            scores += scoring_mask(positions, heading, config)*h
            min_q = min(min_q, float(normalized_separation(
                positions[0], positions[1], controllers[0].params)))
        filtered += alpha*(velocities-filtered)
    log = {k: np.array(v) for k,v in log.items()}
    if outcome == 'Time limit':
        outcome = 'Draw' if abs(scores[0]-scores[1]) < config.period else f'Drone {np.argmax(scores)+1} wins'
    log.update(outcome=outcome, min_q=min_q, final_scores=scores)
    return log


def validate_match(log, config=CONFIG):
    assert np.isclose(log['time'][-1], config.duration), log['outcome']
    assert log['min_q'] >= 1.0, log['min_q']
    for key in ('positions', 'commands'):
        assert np.all(np.isfinite(log[key]))
        assert np.all(log[key] >= config.lower-1e-9)
        assert np.all(log[key] <= config.upper+1e-9)
    v = log['command_velocity']
    assert np.max(np.linalg.norm(v[..., :2], axis=2)) <= config.horizontal_speed+1e-9
    a = np.diff(np.concatenate([np.zeros((1,2,3)), v]), axis=0)/config.period
    assert np.max(np.linalg.norm(a[..., :2], axis=2)) <= config.horizontal_acceleration+1e-9
    assert np.max(np.abs(v[..., 2])) <= config.vertical_speed+1e-9
    assert np.max(np.abs(a[..., 2])) <= config.vertical_acceleration+1e-9
    return {'duration': float(log['time'][-1]), 'min_q': log['min_q'],
            'max_command_speed': float(np.linalg.norm(v[..., :2], axis=2).max()),
            'max_command_acceleration': float(np.linalg.norm(a[..., :2], axis=2).max()),
            'scores': log['final_scores'].tolist(), 'outcome': log['outcome']}
