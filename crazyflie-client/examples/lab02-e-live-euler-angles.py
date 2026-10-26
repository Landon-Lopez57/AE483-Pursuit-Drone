"""
Watch the ZYX Euler angles change as you rotate the drone by hand.

Run this from the crazyflie-client directory:

    uv run examples/lab02-e-live-euler-angles.py

Then pick the drone up and turn it around. Nothing is ever armed, so the motors
will not spin no matter what you do. Close the plot window (or press Ctrl-C in
the terminal) to stop.

On the left, the drone is drawn as its body frame B, along with the world frame
W and the three axes about which the three rotations happen:

    yaw   (psi)   about z_W
    pitch (theta) about y_1, the y axis after yawing
    roll  (phi)   about x_B, the x axis after yawing and pitching

Watching where z_W sits relative to the body frame answers questions like "if
the yaw angle increases, about what axis does the drone rotate?" - the answer
is z_B when the drone is level, but not in general.

On the right, each row shows one of the three rotations: a diagram of that
rotation alone (as on the slide from class) and a plot of that angle over time.

The angles come from the drone, converted from degrees to radians with the sign
of pitch flipped, exactly as all the other code in this course does (the
firmware negates pitch "for the legacy CF2 body coordinate system" - see
kalman_core.c and estimator_complementary.c).

Near pitch = +/- 90 degrees the angles stop being well defined, and you will
see yaw and roll jump around. The body frame in the left-hand plot never jumps,
however - why do you suppose that is?

By default this uses the complementary filter rather than the kalman filter we
use for flight - see the comment on use_complementary_filter below for why.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from ae483.clients import CrazyflieClient


###################################
# PARAMETERS

# Specify the uri of the drone to which you want to connect
uri = 'radio://0/80/2M/E7E7E7E7E7'

# Specify the variables to log
variables = [
    'stateEstimate.yaw',
    'stateEstimate.pitch',
    'stateEstimate.roll',
]

# Specify how often to redraw the plot (seconds)
dt = 0.05

# Specify how many seconds of history to show in the plots of angle over time
history = 20.

# Specify whether to use the complementary filter instead of the kalman filter
# that we use for flight.
#
# The kalman filter estimates position and velocity as well as orientation, and
# it resets itself if either of those leaves a sensible range (see
# kalmanSupervisorIsStateWithinBounds in the firmware). When you hold the drone
# nose-up, the flow deck and the z-ranger report nonsense - their measurements
# assume a roughly level drone above the floor - so the filter infers a huge
# velocity, resets, and the orientation you are watching jumps back to zero
# about once a second.
#
# The complementary filter estimates orientation only, from the rate gyroscope
# and the accelerometer. It has no position or velocity to leave any range, so
# there is nothing to reset. That is all we need here, and this demo never arms
# the drone or asks it to fly.
use_complementary_filter = True


###################################
# APPEARANCE

# One color per axis, the same everywhere: x is red, y is green, z is blue. In
# the drawing of the drone, the body axes are solid and the three axes of
# rotation are dashed - so the axis you roll about is a dashed red line, the
# axis you pitch about is a dashed green line, and the axis you yaw about is a
# dashed blue line.
COLOR = {'x': 'tab:red', 'y': 'tab:green', 'z': 'tab:blue'}

# One color per frame in the diagrams on the right, matching the slide from
# class: the world frame is black, frame 1 is blue, frame 2 is green, and the
# body frame is black again.
FRAME_COLOR = {'W': 'black', '1': 'tab:blue', '2': 'tab:green', 'B': 'black'}

# Each row on the right: the name of the angle, its symbol, the axis it turns
# about, and the two axes of the plane you are looking at, before and after.
ROTATIONS = [
    ('yaw',   r'\psi',   'z_W', ('x_W', 'y_W'), ('x_1', 'y_1'), 'W', '1'),
    ('pitch', r'\theta', 'y_1', ('z_1', 'x_1'), ('z_2', 'x_2'), '1', '2'),
    ('roll',  r'\phi',   'x_B', ('y_2', 'z_2'), ('y_B', 'z_B'), '2', 'B'),
]


###################################
# DRAWING

def make_figure():
    """
    Create the figure and everything in it that does not change: one big
    drawing of the drone on the left, and one row per rotation on the right.
    Returns the axes and the artists that do change.
    """
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(3, 3, width_ratios=[1.85, 0.95, 1.05],
                            wspace=0.12, hspace=0.08,
                            left=0.01, right=0.98, top=0.98, bottom=0.07)

    ax_drone = fig.add_subplot(grid[:, 0], projection='3d')
    ax_diagram = [fig.add_subplot(grid[k, 1]) for k in range(3)]
    ax_time = [fig.add_subplot(grid[k, 2]) for k in range(3)]

    # Each plot of angle over time holds one line and one readout. The readout
    # is inside the plot, in a fixed-width font with a fixed number of digits,
    # so that nothing moves as the number changes.
    lines, readouts = [], []
    for k, (name, symbol, axis, *_) in enumerate(ROTATIONS):
        ax = ax_time[k]
        color = COLOR[axis[0]]
        lines.append(ax.plot([], [], color=color, linewidth=2)[0])
        readouts.append(ax.text(
            0.02, 0.94, '', transform=ax.transAxes, color=color,
            fontsize=14, family='monospace', fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2),
        ))
        ax.set_ylim(-190., 190.)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.grid(alpha=0.3)
        # Only the bottom plot needs its time axis labeled - all three share it
        ax.tick_params(labelsize=11, labelbottom=(k == len(ROTATIONS) - 1))
    ax_time[-1].set_xlabel('time (seconds)', fontsize=12)

    return fig, ax_drone, ax_diagram, ax_time, lines, readouts


def draw_drone(ax, psi, theta, phi):
    """Draw the world frame, the body frame, and the three axes of rotation."""
    ax.cla()

    R_1inW = Rotation.from_euler('Z', psi).as_matrix()
    R_BinW = Rotation.from_euler('ZYX', [psi, theta, phi]).as_matrix()

    # World frame, thin, because it never moves
    for axis, label in zip(np.eye(3), ['x_W', 'y_W', 'z_W']):
        ax.quiver(0, 0, 0, *axis, color='black', linewidth=1)
        ax.text(*(1.4 * axis), f'${label}$', color='black', fontsize=12)

    # Body frame, solid and thick
    for axis, name, label in zip(R_BinW.T, 'xyz', ['x_B', 'y_B', 'z_B']):
        ax.quiver(0, 0, 0, *axis, color=COLOR[name], linewidth=4)
        ax.text(*(1.15 * axis), f'${label}$', color=COLOR[name],
                fontsize=15, fontweight='bold')

    # The three axes of rotation, dashed, in the same colors
    for axis, name, label in [
        (np.eye(3)[:, 2], 'z', r'yaw ($\psi$) about $z_W$'),
        (R_1inW[:, 1], 'y', r'pitch ($\theta$) about $y_1$'),
        (R_BinW[:, 0], 'x', r'roll ($\phi$) about $x_B$'),
    ]:
        ax.quiver(0, 0, 0, *(1.7 * axis), color=COLOR[name], linewidth=2,
                  linestyle='dashed', arrow_length_ratio=0.06, label=label)

    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_zlim(-1.8, 1.8)
    # The zoom fills the panel - a 3d axes leaves wide margins - but not so
    # much that the drawing is clipped at the left edge of the figure
    ax.set_box_aspect([1, 1, 1], zoom=1.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=35)
    ax.legend(loc='upper left', fontsize=13, bbox_to_anchor=(-0.02, 1.02),
              framealpha=0.9)


def draw_rotation(ax, angle, before, after, color_before, color_after, symbol):
    """
    Draw one of the three rotations on its own, in the plane it happens in -
    the same picture as on the slide from class.
    """
    ax.cla()

    # The two axes you are looking at, before and after the rotation. Drawing
    # them in these coordinates makes a positive angle turn counterclockwise.
    for (h, v), labels, color, width in [
        ((1., 0.), before, color_before, 2),
        ((np.cos(angle), np.sin(angle)), after, color_after, 3),
    ]:
        for (x, y), label in [((h, v), labels[0]), ((-v, h), labels[1])]:
            ax.annotate('', xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=width))
            ax.text(1.22 * x, 1.22 * y, f'${label}$', color=color,
                    fontsize=14, fontweight='bold', ha='center', va='center')

    # An arc showing the angle itself
    arc = np.linspace(0., angle, 50)
    ax.plot(0.55 * np.cos(arc), 0.55 * np.sin(arc), color='black', linewidth=1.5)
    ax.text(0.78 * np.cos(angle / 2), 0.78 * np.sin(angle / 2), f'${symbol}$',
            fontsize=16, fontweight='bold', ha='center', va='center')

    # Just large enough to hold the labels, so the diagram fills its panel
    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.32, 1.32)
    ax.set_aspect('equal'); ax.axis('off')


def draw_diagrams(ax_diagram, angles):
    """Draw all three rotations, one per row."""
    for k, (name, symbol, axis, before, after, f0, f1) in enumerate(ROTATIONS):
        draw_rotation(ax_diagram[k], angles[k], before, after,
                      FRAME_COLOR[f0], FRAME_COLOR[f1], symbol)


def draw_angles(ax_time, lines, readouts, times, angle_history):
    """Update the three plots of angle over time, and their readouts."""
    for k, (name, *_) in enumerate(ROTATIONS):
        lines[k].set_data(times, angle_history[k])
        ax_time[k].set_xlim(max(0., times[-1] - history),
                            max(history, times[-1]))
        readouts[k].set_text(
            f'{name.upper():>5} = {angle_history[k][-1]:6.1f} deg')


###################################
# GETTING DATA

def get_angles(drone_client):
    """
    Return the yaw, pitch, and roll angles in radians, or None if the drone has
    not sent anything yet.

    The angles are converted as we always do: degrees to radians, and the sign
    of pitch is flipped to undo what the firmware does to it.
    """
    data = drone_client.data
    if not all(data[v]['data'] for v in variables):
        return None
    return np.array([
        np.deg2rad(data['stateEstimate.yaw']['data'][-1]),
        -np.deg2rad(data['stateEstimate.pitch']['data'][-1]),
        np.deg2rad(data['stateEstimate.roll']['data'][-1]),
    ])


def add_to_history(times, angle_history, t, angles):
    """Add one sample, and forget anything older than the plots will show."""
    times.append(t)
    for k in range(len(ROTATIONS)):
        angle_history[k].append(np.rad2deg(angles[k]))
    while times[-1] - times[0] > history:
        times.pop(0)
        for k in range(len(ROTATIONS)):
            angle_history[k].pop(0)


###################################
# DEMO

if __name__ == '__main__':
    drone_client = CrazyflieClient(uri, variables=variables)

    try:
        drone_client.wait_until_ready()

        if use_complementary_filter:
            print('CrazyflieClient: Switching to the complementary filter')
            drone_client.cf.param.set_value('stabilizer.estimator', 1)
            time.sleep(1.)

        plt.ion()
        fig, ax_drone, ax_diagram, ax_time, lines, readouts = make_figure()

        print('Pick up the drone and turn it around. Close the window to stop.')

        times, angle_history = [], [[] for _ in ROTATIONS]
        t_start = time.time()

        while plt.fignum_exists(fig.number):
            angles = get_angles(drone_client)
            if angles is None:
                plt.pause(dt)
                continue

            add_to_history(times, angle_history, time.time() - t_start, angles)

            draw_drone(ax_drone, *angles)
            draw_diagrams(ax_diagram, angles)
            draw_angles(ax_time, lines, readouts, times, angle_history)

            plt.pause(dt)

    except KeyboardInterrupt:
        print('\nStopping.')

    finally:
        drone_client.close()
        plt.close('all')
