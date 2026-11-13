"""
flight_collision_avoidance.py

Flight script for the collision avoidance pass-through test. It is built from your
flight.py (same structure, same safety shutdown, same data saving) and is meant to be
run ONCE PER DRONE, in two terminals:

  1. Runner:   set role = 'runner'   -> takes off from where it is placed and hovers.
     Start this first and wait until it is hovering.
  2. Pursuer:  set role = 'pursuer'  -> checks that the runner is hovering, takes off,
     then flies a straight line that passes THROUGH the runner's position, using the
     avoidance controller from collision_avoidance.py (the same one the notebook
     simulates). It logs the normalized separation q and saves it.

For the other flight, swap the two drones (change uri, me, other, marker_deck_ids and role).

Files needed in the ae483 package folder: clients.py, myclients.py, clients_battle.py,
collision_avoidance.py.

BEFORE FLYING: read the SAFETY section below and set BOUNDS for your lab.
"""

###################################
# IMPORTS
import json
import os
import time
from dataclasses import asdict

import numpy as np

from ae483.clients_battle import QualisysClient
from ae483.myclients import MyCrazyflieClient
from ae483.collision_avoidance2 import (VOParams, TestConfig, LineTrajectory, AvoidanceController,
                                       normalized_separation, extend_through, min_q_from_mocap)

###################################
# PARAMETERS -- WHICH DRONE IS THIS, AND WHAT IS IT DOING?

# 'pursuer' flies the pass-through trajectory. 'runner' hovers.
role = 'pursuer'

# The uri of THIS drone (if your radio channel is X, 'radio://0/X/2M/E7E7E7E7E7')
uri = 'radio://0/36/2M/E7E7E7E7E7'

# The mocap name of THIS drone's marker deck, and of the OTHER drone's.
me = 'marker_deck_30'
other = 'marker_deck_16'

# The marker IDs of THIS drone's marker deck ([X + 1, X + 2, X + 3, X + 4])
marker_deck_ids = [31, 32, 33, 34]

###################################
# PARAMETERS -- THE SAME AS FLIGHT.PY

use_controller = True
use_observer = False
gains_filename = 'gains.json' if use_controller else None
data_filename = f'{role}_test_1.json'     # CHANGE THE NUMBER FOR EACH FLIGHT

variables = [
    # WHEN RUNNING DEFAULT CONTROLLER
    'stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
    'stateEstimate.yaw', 'stateEstimate.pitch', 'stateEstimate.roll',
    'ctrltarget.x', 'ctrltarget.y', 'ctrltarget.z',
    # WHEN RUNNING CUSTOM CONTROLLER
    'ae483log.p_x', 'ae483log.p_y', 'ae483log.p_z',
    'ae483log.psi', 'ae483log.theta', 'ae483log.phi',
    'ae483log.v_x', 'ae483log.v_y', 'ae483log.v_z',
    'ae483log.w_x', 'ae483log.w_y', 'ae483log.w_z',
    'ae483log.p_x_des', 'ae483log.p_y_des', 'ae483log.p_z_des',
    'ae483log.m_1', 'ae483log.m_2', 'ae483log.m_3', 'ae483log.m_4',
]

###################################
# PARAMETERS -- THE TEST

# Collision avoidance parameters. KEEP THESE THE SAME AS IN THE NOTEBOOK.
# Envelope per drone (report): a = b = 0.0762 m (3 in), c = 0.0381 m (1.5 in)  ->  z_stretch = c / a = 0.5.
# q uses the pairwise semi-axes A = B = R_i + R_j = 0.1524 m and C = z_stretch * A = 0.0762 m.
# margin, vz_weight, tau, v_max and a_max are placeholders: set them from your parameter table.
# vz_weight > 1 makes the controller dodge sideways instead of passing just above/below the other
# drone (the envelope is thin vertically, so otherwise it prefers up/down, into the downwash).
P = VOParams(R_i=0.0762, R_j=0.0762, z_stretch=0.5, margin=0.02, vz_weight=10.0,
             tau=2.0, v_max=1.0, a_max=2.0, dt=0.05)

# 'vo', 'envelope', or 'both'. ('none' = tracking only exists for the simulation: it flies
# straight into the runner, so this script refuses it unless you set ALLOW_NO_AVOIDANCE.)
# 'envelope' alone can stop at the surface of the ellipsoid and never get around the runner
# if the path goes exactly through its center, so the pass may not finish.
MODE = 'both'
ALLOW_NO_AVOIDANCE = False

RUNNER_HOVER_Z = 0.5        # altitude at which the runner hovers (m)
RUNNER_HOVER_TIME = 45.0    # how long the runner hovers before landing by itself (s)
PURSUER_SPEED = 0.5         # speed along the pass-through path (m/s)
MIN_START_DISTANCE = 1.0    # the pursuer must start at least this far from the runner (m)
STALE_LIMIT = 0.5           # abort if mocap data is missing for this long (s)

# ---------------------------------------------------------------------------------------
# SAFETY: THE FLIGHT BOUNDS BELOW ARE PLACEHOLDERS. Set them from your lab dimensions, the
# capture volume of the mocap system, and your constraints table. Setpoints are clipped to
# this box, the end of the path is shortened to fit inside it, and the flight is aborted
# (landing) if the measured position leaves it by more than GEOFENCE_ABORT_MARGIN.
# (xmin, xmax, ymin, ymax, zmin, zmax) in meters, in the mocap frame.
BOUNDS = (-2.0, 2.0, -1.5, 1.5, 0.15, 1.2)
GEOFENCE_ABORT_MARGIN = 0.3
# ---------------------------------------------------------------------------------------


###################################
# HELPERS

def takeoff(drone_client, x, y, z_target, z_start=0.20, step=0.15):
    """Climb in steps, 1 s per step, like the graceful takeoff in flight.py."""
    z = z_start
    while True:
        drone_client.move(x, y, z, 0.0, 1.0)
        if z >= z_target - 1e-6:
            break
        z = min(z + step, z_target)


def land(drone_client, x, y, z_from, z_end=0.20, step=0.15):
    """Descend in steps, then stop the motors."""
    z = z_from
    while z > z_end + 1e-6:
        z = max(z - step, z_end)
        drone_client.move(x, y, z, 0.0, 1.0)
    drone_client.stop(1.0)


def outside_bounds(p, bounds, margin):
    lo = np.array([bounds[0], bounds[2], bounds[4]]) - margin
    hi = np.array([bounds[1], bounds[3], bounds[5]]) + margin
    return bool(np.any(p < lo) or np.any(p > hi))


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def run_pursuit(drone_client, mocap_client, controller, t_end):
    """
    The control loop. Runs at 1 / P.dt Hz. Each iteration reads the latest pose of both
    drones from the mocap client, asks the controller for a setpoint, and sends it.
    Returns (log, aborted, last_setpoint). aborted is None, or a string saying why the
    loop ended early.
    """
    log = {k: [] for k in ['t', 'p_i', 'v_i', 'p_j', 'v_j', 'sp', 'v_cmd', 'status', 'q']}
    sp = controller.sp.copy()
    stale = 0.0
    aborted = None
    t0 = time.time()
    k = 0
    while True:
        t = time.time() - t0
        if t > t_end:
            break
        pose_i, v_i, _ = mocap_client.get_latest(me, max_age=0.2)
        pose_j, v_j, _ = mocap_client.get_latest(other, max_age=0.2)
        if pose_i is None or pose_j is None:
            # No fresh data: keep sending the previous setpoint (the drone hovers there)
            stale += P.dt
            if stale > STALE_LIMIT:
                aborted = 'mocap data lost'
                break
        else:
            stale = 0.0
            p_i, p_j = np.array(pose_i[:3]), np.array(pose_j[:3])
            v_i = np.zeros(3) if v_i is None else np.asarray(v_i)
            v_j = np.zeros(3) if v_j is None else np.asarray(v_j)
            if outside_bounds(p_i, BOUNDS, GEOFENCE_ABORT_MARGIN):
                aborted = 'drone left the flight bounds'
                break
            sp, info = controller.step(t, p_i, v_i, p_j, v_j)
            log['t'].append(t)
            log['p_i'].append(p_i); log['v_i'].append(v_i)
            log['p_j'].append(p_j); log['v_j'].append(v_j)
            log['sp'].append(sp); log['v_cmd'].append(info['v_cmd'])
            log['status'].append(info['status'])
            log['q'].append(float(normalized_separation(p_i, p_j, P)))
        # Constant yaw (0), as the task requires
        drone_client.cf.commander.send_position_setpoint(float(sp[0]), float(sp[1]), float(sp[2]), 0.0)
        k += 1
        delay = t0 + k * P.dt - time.time()
        if delay > 0:
            time.sleep(delay)
    return log, aborted, sp


def fly_runner(drone_client, mocap_client):
    """Take off where the drone is placed, hover, then land."""
    pose, _, _ = mocap_client.get_latest(me, max_age=None)
    x0, y0 = float(pose[0]), float(pose[1])
    print(f'Runner: hovering at x = {x0:.2f}, y = {y0:.2f}, z = {RUNNER_HOVER_Z:.2f} for {RUNNER_HOVER_TIME:.0f} s')
    print('Runner: let it time out and land by itself. Ctrl-C stops the motors immediately.')
    drone_client.stop(1.0)
    takeoff(drone_client, x0, y0, RUNNER_HOVER_Z)
    drone_client.move(x0, y0, RUNNER_HOVER_Z, 0.0, RUNNER_HOVER_TIME)
    land(drone_client, x0, y0, RUNNER_HOVER_Z)
    return None, None


def check_runner_is_hovering(mocap_client):
    """Average the runner's pose for half a second and make sure it is actually in the air."""
    samples = []
    t0 = time.time()
    while time.time() - t0 < 0.5:
        pose, _, _ = mocap_client.get_latest(other, max_age=0.2)
        if pose is not None:
            samples.append(pose[:3])
        time.sleep(0.02)
    if not samples:
        raise Exception(f'Pursuer: no mocap data for the runner ({other}).')
    runner = np.mean(samples, axis=0)
    if runner[2] < 0.3:
        raise Exception(
            f'Pursuer: the runner is at z = {runner[2]:.2f} m, so it does not look like it is hovering. '
            f'Start the runner first and wait until it is in the air.'
        )
    return runner


def plan_pursuit(mocap_client, runner):
    """
    Work out the straight path through the runner and check it. Called BEFORE arming, so a
    bad setup raises an exception while the drone is still on the ground.
    """
    pose, _, _ = mocap_client.get_latest(me, max_age=None)
    start = np.array([float(pose[0]), float(pose[1]), float(runner[2])])
    horizontal = float(np.linalg.norm(runner[:2] - start[:2]))
    if horizontal < MIN_START_DISTANCE:
        raise Exception(f'Pursuer: it starts only {horizontal:.2f} m from the runner. '
                        f'Place it at least {MIN_START_DISTANCE:.1f} m away.')
    if outside_bounds(start, BOUNDS, 0.0) or outside_bounds(runner, BOUNDS, 0.0):
        raise Exception('Pursuer: the start position or the runner is outside BOUNDS.')
    end = extend_through(start, runner, BOUNDS)
    beyond = float(np.linalg.norm(end - runner))
    print(f'Pursuer: path from {np.round(start, 2)} through the runner at {np.round(runner, 2)} to {np.round(end, 2)}')
    if beyond < 0.5:
        print(f'Pursuer: WARNING -- the path only continues {beyond:.2f} m past the runner (BOUNDS are tight)')
    return start, end


def fly_pursuer(drone_client, mocap_client, runner, start, end):
    """Take off, fly the line through the runner's position with avoidance, land."""
    x0, y0, z_path = float(start[0]), float(start[1]), float(start[2])
    cfg = TestConfig(mode=MODE, v_nom=PURSUER_SPEED, bounds=BOUNDS)
    traj = LineTrajectory(start, end, PURSUER_SPEED)

    drone_client.stop(1.0)
    takeoff(drone_client, x0, y0, z_path)
    drone_client.move(x0, y0, z_path, 0.0, 2.0)       # settle before the pass

    pose, _, _ = mocap_client.get_latest(me, max_age=None)
    controller = AvoidanceController(P, cfg, traj, p0=np.array(pose[:3]))
    print(f'Pursuer: starting the pass (mode = {MODE}, speed = {PURSUER_SPEED} m/s)')
    log, aborted, sp = run_pursuit(drone_client, mocap_client, controller, traj.duration + 3.0)
    if aborted is not None:
        print(f'Pursuer: ABORTED -- {aborted}. Landing.')
    drone_client.move(sp[0], sp[1], sp[2], 0.0, 1.0)
    land(drone_client, float(sp[0]), float(sp[1]), float(sp[2]))
    test = {'config': asdict(cfg), 'path_start': start, 'path_end': end, 'runner_position': runner,
            'aborted': aborted, 'log': log}
    return test, aborted


###################################
# MAIN

def main():
    if role not in ('pursuer', 'runner'):
        raise ValueError("role must be 'pursuer' or 'runner'")
    if role == 'pursuer' and MODE not in ('vo', 'envelope', 'both', 'none'):
        raise ValueError("MODE must be 'vo', 'envelope', or 'both'")
    if role == 'pursuer' and MODE == 'none' and not ALLOW_NO_AVOIDANCE:
        raise ValueError("MODE = 'none' flies the pursuer straight into the runner. "
                         "Use it in the simulation only, or set ALLOW_NO_AVOIDANCE = True if you really mean it.")

    # Read control gains, if this flight uses a custom controller
    gains = None
    gains_created = None
    if gains_filename is not None:
        with open(gains_filename, 'r') as f:
            gains_file = json.load(f)
        gains = gains_file['gains']
        gains_created = gains_file['created']
        print(f'Using {len(gains)} gains from {gains_filename}, written {gains_created}')

    print(f'Role: {role}. This drone: {me}. Other drone: {other}. Bounds: {BOUNDS}')

    drone_client = MyCrazyflieClient(
        uri,
        use_controller=use_controller,
        use_observer=use_observer,
        marker_deck_ids=marker_deck_ids,
        variables=variables,
    )
    mocap_client = None
    test = None

    # Everything from here until "finally" is what happens during the flight. If anything
    # goes wrong, including Ctrl-C, the "finally" block stops the motors and disconnects.
    try:
        drone_client.wait_until_ready()
        drone_client.set_gains(gains)

        # Only THIS drone's pose is sent to THIS drone (callback). The other drone is only
        # read (callback None): sending its pose to this drone would corrupt the estimate.
        bodies = [{'name': me, 'callback': drone_client.send_pose}]
        if role == 'pursuer':
            bodies.append({'name': other, 'callback': None})
        mocap_client = QualisysClient(bodies)
        mocap_client.wait_for_data(me)

        runner = start = end = None
        if role == 'pursuer':
            mocap_client.wait_for_data(other)
            runner = check_runner_is_hovering(mocap_client)
            start, end = plan_pursuit(mocap_client, runner)

        drone_client.arm()

        if role == 'runner':
            fly_runner(drone_client, mocap_client)
        else:
            test, _ = fly_pursuer(drone_client, mocap_client, runner, start, end)

    except KeyboardInterrupt:
        print('\nInterrupted - stopping the motors and saving whatever data were collected.')

    finally:
        try:
            drone_client.close()
        except Exception as e:
            print(f'Error while closing the connection to the drone: {e}')

        if mocap_client is not None:
            try:
                mocap_client.close()
            except Exception as e:
                print(f'Error while closing the connection to the motion capture system: {e}')

        data = {}
        data['gains'] = drone_client.gains
        data['gains_created'] = gains_created if drone_client.gains is not None else None
        data['use_controller'] = use_controller
        data['drone'] = drone_client.data
        data['mocap'] = mocap_client.data.get(me, {}) if mocap_client is not None else {}
        data['bodies'] = mocap_client.data if mocap_client is not None else {}

        # Minimum normalized separation from the 100 Hz mocap log (pursuer flights only)
        q_min = None
        if mocap_client is not None and role == 'pursuer' and other in mocap_client.data:
            try:
                q_min, _ = min_q_from_mocap(mocap_client.data[me], mocap_client.data[other], P)
            except Exception as e:
                print(f'Could not compute q from the mocap log: {e}')
        data['test'] = {'role': role, 'me': me, 'other': other, 'params': asdict(P), 'mode': MODE,
                        'q_min': q_min, **(test or {'config': {'mode': MODE}})}
        if q_min is not None:
            if test is None or test.get('aborted') is not None:
                print(f'Minimum normalized separation q in this partial trial: {q_min:.3f} '
                      f'(the flight did not complete, so it does not count as a pass)')
            else:
                print(f'Minimum normalized separation q in this trial: {q_min:.3f} '
                      f'({"PASS" if q_min >= 1.0 else "FAIL"}, needs q >= 1)')

        # Write to a temporary file, then rename, so an interruption cannot leave a half-written file
        temporary_filename = data_filename + '.partial'
        with open(temporary_filename, 'w') as outfile:
            json.dump(to_jsonable(data), outfile, sort_keys=False)
        os.replace(temporary_filename, data_filename)
        print(f'Wrote flight data to {data_filename}')


if __name__ == '__main__':
    main()