###################################
# IMPORTS

import time
import numpy as np
import json
import os
from clients_tracking import QualisysClient
# from ae483.clients import QualisysClient
from ae483.myclients import MyCrazyflieClient
from ae483.mytools import *
from tracking import travel_direction
from pursuit_avoidance import PursuitAvoidance
from match_config import CONFIG


###################################
# PARAMETERS

# -- PROBABLY THE SAME FOR EVERY FLIGHT IN LABS 1-10 --

# Specify the uri of the drone to which you want to connect (if your radio
# channel is X, the uri should be 'radio://0/X/2M/E7E7E7E7E7')
uri = 'radio://0/36/2M/E7E7E7E7E7'

# Specify the name of the rigid body that corresponds to your active marker
# deck in the motion capture system. If your marker deck number is X, this name
# should be 'marker_deck_X'.
pursuer = 'marker_deck_30'
runner = 'Reference'

# Specify the marker IDs that correspond to your active marker deck in the
# motion capture system. If your marker deck number is X, these IDs should be
# [X + 1, X + 2, X + 3, X + 4]. They are listed in clockwise order (viewed
# top-down), starting from the front.
marker_deck_ids = [31, 32, 33, 34]

# -- MAY CHANGE FROM FLIGHT TO FLIGHT --

# Specify whether or not to use the motion capture system
use_mocap = True

# Specify whether or not to use a custom controller
use_controller = False

# Specify whether or not to use a custom observer
use_observer = False

# Specify the name of the file in which to save flight data
data_filename = 'mocap_test.json'

# Specify the name of the file from which to read control gains, or None if
# this flight does not use a custom controller. This file is written by your
# design notebook.
gains_filename = 'gains.json' if use_controller else None

# Specify the variables you want to log at 100 Hz from the drone
variables = [
    # WHEN RUNNING DEFAULT CONTROLLER
    # State estimates
    'stateEstimate.x',
    'stateEstimate.y',
    'stateEstimate.z',
    'stateEstimate.yaw',
    'stateEstimate.pitch',
    'stateEstimate.roll',
    # Desired position
    'ctrltarget.x',
    'ctrltarget.y',
    'ctrltarget.z',
    # WHEN RUNNING CUSTOM CONTROLLER
    # State
    'ae483log.p_x',
    'ae483log.p_y',
    'ae483log.p_z',
    'ae483log.psi',
    'ae483log.theta',
    'ae483log.phi',
    'ae483log.v_x',
    'ae483log.v_y',
    'ae483log.v_z',
    'ae483log.w_x',
    'ae483log.w_y',
    'ae483log.w_z',
    # Desired position
    'ae483log.p_x_des',
    'ae483log.p_y_des',
    'ae483log.p_z_des',
    # Motor power commands
    'ae483log.m_1',
    'ae483log.m_2',
    'ae483log.m_3',
    'ae483log.m_4',
]

# Follow 0.5 m behind the runner's horizontal travel direction.
follow_distance = CONFIG.follow_distance

# Height at which the pursuer flies while tracking (meters)
tracking_height = CONFIG.height

# How long to track the runner (seconds)
tracking_duration = CONFIG.duration

# The pursuer's setpoint is always kept inside this box (meters)
box_min = CONFIG.lower
box_max = CONFIG.upper

# Largest change in the setpoint per loop (meters). At 10 Hz, 0.05 m per loop
# limits the setpoint to moving at 0.5 m/s.
max_step = CONFIG.horizontal_speed * CONFIG.period

# Minimum horizontal separation for takeoff. In pursuit, the shared VO
# controller uses its larger buffered ellipsoidal envelope.
min_separation = 0.1

# If the pursuer's own motion capture data are older than this (seconds), its
# state estimate is no longer being corrected, so stop tracking and land
pursuer_max_age = 0.3

# Before takeoff, the drone's state estimate must agree with motion capture to
# within this distance (meters), or the flight is aborted
max_estimate_error = 0.1

###################################
# FLIGHT CODE

# Read control gains, if this flight uses a custom controller. The date is
# printed so that you notice if you are about to fly with gains from an
# earlier design that you forgot to export.
gains = None
gains_created = None
if gains_filename is not None:
    with open(gains_filename, 'r') as f:
        gains_file = json.load(f)
    gains = gains_file['gains']
    gains_created = gains_file['created']
    print(f'Using {len(gains)} gains from {gains_filename}, written {gains_created}')

# Create and start the client that will connect to the drone
drone_client = MyCrazyflieClient(
    uri,
    use_controller=use_controller,
    use_observer=use_observer,
    marker_deck_ids=marker_deck_ids if use_mocap else None,
    variables=variables,
)

# Create this now so that it always exists, even if we never get far enough to
# connect to the motion capture system
mocap_client = None

# Everything from here until "finally" is what happens during your flight. If
# anything goes wrong - including if you press Ctrl-C - the code in the
# "finally" block still runs, stopping the motors, disarming the drone, and
# disconnecting.
try:
    # Wait until the client is fully connected to the drone and until the state
    # estimate has had time to converge
    drone_client.wait_until_ready()

    # Send the control gains to the drone and confirm they arrived. This does
    # nothing if gains is None. It happens before the motion capture system is
    # started so that as little time as possible passes between the start of
    # motion capture data and the start of flight - the smaller that gap, the
    # easier it is to align the two sets of data afterward.
    drone_client.set_gains(gains)

    # Create and start the client that will connect to the motion capture system
    if use_mocap:
        mocap_client = QualisysClient([{'name': pursuer, 'callback': drone_client.send_pose},
                                       {'name': runner, 'callback' : None}])

    # Arm the drone. Brushless drones will not spin their motors until they are
    # armed. Brushed drones do not need to be armed, but arming them does no
    # harm, so the same flight code works for both.
    drone_client.arm()

    drone_client.stop(1.0)


    # Make sure both drones are visible to the motion capture system before
    # flying. This raises an exception (and so never takes off) if either
    # rigid body is missing.
    mocap_client.wait_for_data(pursuer)
    mocap_client.wait_for_data(runner)

    posi_p, _, _ = mocap_client.get_latest(pursuer, max_age=pursuer_max_age)
    if posi_p is None:
        raise RuntimeError('No MoCap data found for pursuer')
    position = np.array(posi_p[:3], dtype=float)
    if np.any(position[:2] < box_min[:2]) or np.any(position[:2] > box_max[:2]):
        raise RuntimeError('Initial position is outside the flight box')
    if not box_min[2] <= tracking_height <= box_max[2]:
        raise ValueError('Tracking height is outside the match')
    posi_r, runner_velocity, _ = mocap_client.get_latest(runner)
    if posi_r is None or np.linalg.norm(position[:2] - posi_r[:2]) < min_separation:
        raise RuntimeError('Runner is missing or too close for takeoff')

    # Until the runner moves, use the direction from pursuer to runner.
    runner_direction = np.array(posi_r[:2]) - position[:2]
    runner_direction /= np.linalg.norm(runner_direction)
    command = position.copy()
    landing_z = position[2]  # Return to the measured starting surface height.
    period = CONFIG.period

    # Helper to move pursuer drone
    def send(position):
        drone_client.cf.commander.send_position_setpoint(*position, 0.0)

    def change_altitude(goal):
        start = command[2]
        distance = abs(goal - start)
        if distance < 1e-9:
            return
        duration = max(1.875 * distance / CONFIG.vertical_speed,
                       np.sqrt(5.774 * distance / CONFIG.vertical_acceleration), period)
        steps = int(np.ceil(duration / period))
        for k in range(1, steps + 1):
            pp, _, _ = mocap_client.get_latest(pursuer, max_age=pursuer_max_age)
            pr, _, _ = mocap_client.get_latest(runner)
            if pp is None or pr is None:
                raise RuntimeError('Lost mocap during altitude transition')
            if np.linalg.norm(np.array(pp[:2])-pr[:2]) < min_separation:
                raise RuntimeError('Minimum separation during altitude transition')
            u = k / steps
            command[2] = start + (goal-start)*(10*u**3-15*u**4+6*u**5)
            send(command)
            time.sleep(period)

    # Lower smoothly to the measured starting surface.
    def descend():
        change_altitude(landing_z)
        drone_client.stop(1.0)

    try:
        change_altitude(tracking_height)

        avoidance = PursuitAvoidance(command, box_min, box_max, tracking_height,
                                     period, max_step, follow_distance)
        start_time = time.time()
        while time.time() - start_time < tracking_duration:
            posi_p, pursuer_velocity, _ = mocap_client.get_latest(pursuer, max_age=pursuer_max_age)
            posi_r, runner_velocity, _ = mocap_client.get_latest(runner)
            if posi_p is None or posi_r is None:
                print('Lost motion capture')
                break
            p_pursuer = np.array(posi_p[:2])
            p_runner = np.array(posi_r[:2])
            runner_direction = travel_direction(runner_velocity, runner_direction)
            next_command, target, status = avoidance.step(
                posi_p[:3], pursuer_velocity, posi_r[:3], runner_velocity, runner_direction)
            if status == 'stop':
                print(f'Avoidance stop: {avoidance.last_reason} - landing')
                break
            command[:] = next_command
            send(command)
            time.sleep(period)
    finally:
        descend()


except KeyboardInterrupt:
    print('\nInterrupted - stopping the motors and saving whatever data were collected.')

finally:
    # Stop the motors, disarm, and disconnect from the drone. Each step is
    # guarded so that a failure in one of them cannot prevent the others - and,
    # in particular, cannot prevent your flight data from being saved.
    try:
        drone_client.close()
    except Exception as e:
        print(f'Error while closing the connection to the drone: {e}')

    # Disconnect from the motion capture system
    if mocap_client is not None:
        try:
            mocap_client.close()
        except Exception as e:
            print(f'Error while closing the connection to the motion capture system: {e}')

    # Assemble flight data from both clients. The gains are saved along with the
    # data, so that every flight says for itself which controller flew it.
    data = {}
    data['gains'] = drone_client.gains
    data['gains_created'] = gains_created if drone_client.gains is not None else None
    data['use_controller'] = use_controller
    data['drone'] = drone_client.data
    data['mocap'] = mocap_client.data.get(pursuer, {}) if use_mocap and mocap_client is not None else {}
    data['bodies'] = mocap_client.data if use_mocap and mocap_client is not None else {}

    # Write flight data to a file. We write to a temporary file first and then
    # rename it, which is an operation the operating system does all at once.
    # That way, if anything interrupts the writing, you are left with your
    # previous data file rather than with a half-written one that cannot be
    # read at all.
    temporary_filename = data_filename + '.partial'
    with open(temporary_filename, 'w') as outfile:
        json.dump(data, outfile, sort_keys=False)
    os.replace(temporary_filename, data_filename)
    print(f'Wrote flight data to {data_filename}')


# uv run ae483-reboot "radio://0/36/2M/E7E7E7E7E7"

# uv run ae483-flash "radio://0/36/2M/E7E7E7E7E7" --brushless
