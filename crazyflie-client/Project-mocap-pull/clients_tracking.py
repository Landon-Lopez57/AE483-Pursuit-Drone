"""
clients_battle.py

A modified copy of QualisysClient from clients.py that makes live motion capture
data easy to read from your own code (e.g., for one drone to chase another).

CrazyflieClient and queue_handler are NOT copied - they are imported from the
original clients.py, so the labs' code is left alone. Put this file in the same
folder as clients.py.

Typical use:

    from clients_battle import QualisysClient

    bodies = [{'name': 'marker_deck_40', 'callback': None},
              {'name': 'marker_deck_90', 'callback': None}]
    qualisys = QualisysClient(bodies)
    try:
        qualisys.wait_for_data('marker_deck_40')
        qualisys.monitor(duration=10)          # print live data
        pose, vel, age = qualisys.get_latest('marker_deck_40')
    finally:
        qualisys.close()
"""

import time
import asyncio
import xml.etree.ElementTree as ET
from threading import Thread, Lock
from multiprocessing import SimpleQueue

import numpy as np
import qtm_rt as qtm
from scipy.spatial.transform import Rotation

# Reused, unchanged, from the course code
from ae483.clients import CrazyflieClient, queue_handler  # noqa: F401 (CrazyflieClient re-exported for convenience)


###################################
# NEW: THREAD-SAFE STORAGE FOR THE LATEST POSE

class LatestPose:
    """
    Holds the most recent pose of one rigid body, safe to read from any thread.

    pose is [x, y, z, qx, qy, qz, qw] (meters, unit quaternion). Velocity is a
    low-pass filtered finite difference of position, computed with the mocap
    timestamps (not python's receive time), so it is only as good as that
    filtering - treat it as approximate.
    """

    def __init__(self):
        self._lock = Lock()
        self._pose = None
        self._t_mocap = None    # qualisys clock, seconds
        self._t_local = None    # time.time() when the pose arrived
        self._vel = None        # filtered velocity [vx, vy, vz], m/s

    def update(self, pose, t_mocap, alpha=0.3):
        with self._lock:
            if self._pose is not None:
                dt = t_mocap - self._t_mocap
                if dt > 0:
                    v = (np.array(pose[:3]) - np.array(self._pose[:3])) / dt
                    if self._vel is None:
                        self._vel = v
                    else:
                        self._vel = alpha * v + (1 - alpha) * self._vel
            self._pose = list(pose)
            self._t_mocap = t_mocap
            self._t_local = time.time()

    def get(self, max_age=0.2):
        """
        Return (pose, velocity, age).

        age is the number of seconds since the last valid pose arrived. If no
        pose has ever arrived, or if the latest one is older than max_age,
        pose and velocity are None. Pass max_age=None to always get the last
        known pose, however old (age tells you how stale it is).
        """
        with self._lock:
            if self._pose is None:
                return None, None, None
            age = time.time() - self._t_local
            if max_age is not None and age > max_age:
                return None, None, age
            vel = None if self._vel is None else self._vel.copy()
            return list(self._pose), vel, age


###################################
# CLIENT FOR QUALISYS (modified copy)

class QualisysClient(Thread):
    def __init__(self, bodies_to_track, ip_address='128.174.245.64', version='1.24'):
        """
        The argument

            bodies_to_track

        must be a list. Each element of this list must be a dictionary with these keys:

            'name': a string with the name of a rigid body to track
            'callback': either a function with one argument (pose) that you want to be called each time a
                        new pose of the rigid body is available, or None (no function will be called); the
                        'pose' argument will be a list with seven elements: [x, y, z, qx, qy, qz, qw]

        and, optionally (NEW):

            'direct_callback': a function with two arguments (pose, t) that is called for EVERY valid
                        pose, directly in the motion capture thread. It must be fast and must never
                        block (no radio sends, no sleeps).

        The position and orientation of each named rigid body in bodies_to_track will be logged at
        100 Hz as x, y, z, yaw, pitch, roll, along with a time stamp, regardless of whether or not
        a callback is provided.

        NEW: the latest pose of every tracked body is also available at any time through

            qualisys.get_latest(name)   ->  (pose, velocity, age)

        with no callback needed. See also wait_for_data() and monitor().

        Examples:

            [{'name': 'marker_deck_90', 'callback': None}]

            [{'name': 'marker_deck_90', 'callback': None},
             {'name': 'marker_deck_40', 'callback': drone_client.send_pose}]
        """
        Thread.__init__(self, daemon=True)
        self.ip_address = ip_address
        self.bodies_to_track = bodies_to_track # <-- NOTE: modified in place (see original clients.py)
        self.version = version
        self.data = {}
        self.latest = {}   # NEW: name -> LatestPose
        for body in self.bodies_to_track:
            if body['callback'] is None:
                body['queue'] = None
                body['thread'] = None
            else:
                body['queue'] = SimpleQueue()
                body['thread'] = Thread(
                    target=queue_handler,
                    args=[body['queue'], body['callback']],
                    daemon=True,
                )
                body['thread'].start()
            # NEW: one LatestPose per body
            body['latest'] = LatestPose()
            self.latest[body['name']] = body['latest']
            self.data[body['name']] = {
                'time': [],
                'x': [],
                'y': [],
                'z': [],
                'yaw': [],
                'pitch': [],
                'roll': [],
            }
        self.num_bodies = 0
        self.connection = None
        self._stay_open = True
        self._is_closed = False
        self.start()

    #
    # NEW: reading live data
    #

    def get_latest(self, name, max_age=0.2):
        """
        Return (pose, velocity, age) for the rigid body called name.
        pose = [x, y, z, qx, qy, qz, qw]; velocity = numpy array [vx, vy, vz]
        (or None until two poses have arrived). See LatestPose.get for max_age.
        """
        if name not in self.latest:
            raise KeyError(f'QualisysClient: "{name}" is not in bodies_to_track')
        return self.latest[name].get(max_age)

    def wait_for_data(self, name, timeout=10.):
        """
        Block until a valid pose has arrived for name. Raises an exception after
        timeout seconds, which usually means the name is wrong, the body is
        occluded, or the connection to the mocap system failed.
        """
        start_time = time.time()
        while True:
            pose, _, _ = self.get_latest(name, max_age=None)
            if pose is not None:
                return
            if time.time() - start_time > timeout:
                raise Exception(
                    f'QualisysClient: No valid pose for "{name}" within {timeout:.1f} seconds. '
                    f'Check the name, that the body is visible to the cameras, and the console '
                    f'for warnings.'
                )
            time.sleep(0.05)

    def monitor(self, names=None, duration=10., rate_hz=5.):
        """
        Print the live position, yaw, velocity, and data age of the named bodies
        (default: all of them) for duration seconds. Runs in the calling thread.
        """
        if names is None:
            names = [body['name'] for body in self.bodies_to_track]
        end_time = time.time() + duration
        while time.time() < end_time:
            for name in names:
                pose, vel, age = self.get_latest(name, max_age=None)
                if pose is None:
                    print(f'{name}: no data yet')
                    continue
                yaw = Rotation.from_quat(pose[3:]).as_euler('ZYX', degrees=True)[0]
                vel_str = 'n/a' if vel is None else f'[{vel[0]:+.2f} {vel[1]:+.2f} {vel[2]:+.2f}]'
                print(
                    f'{name}: pos=[{pose[0]:+.3f} {pose[1]:+.3f} {pose[2]:+.3f}] m  '
                    f'yaw={yaw:+.1f} deg  vel={vel_str} m/s  age={age * 1e3:.0f} ms'
                )
            time.sleep(1. / rate_hz)

    #
    # Closing (unchanged from clients.py)
    #

    def close(self, timeout=5.):
        if self._is_closed:
            return
        self._is_closed = True

        for body in self.bodies_to_track:
            if body['queue'] is not None:
                body['queue'].put('END')

        self._stay_open = False

        self.join(timeout)
        if self.is_alive():
            print(
                f'QualisysClient: WARNING -- The motion capture thread did not stop within '
                f'{timeout:.1f} seconds. Flight data have still been collected. If this keeps '
                f'happening, tell us.'
            )

        for body in self.bodies_to_track:
            if body['thread'] is not None:
                body['thread'].join(timeout)
                if body['thread'].is_alive():
                    print(
                        f'QualisysClient: WARNING -- The callback thread for '
                        f'{body["name"]} did not stop within {timeout:.1f} seconds.'
                    )

    def run(self):
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while self._stay_open:
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        print('QualisysClient: Connect to motion capture system')
        self.connection = await qtm.connect(self.ip_address, version=self.version)
        params = await self.connection.get_parameters(parameters=['6d'])
        xml = ET.fromstring(params)
        names = [name.text.strip() for index, name in enumerate(xml.findall('*/Body/Name'))]

        self.num_bodies = len(names)
        if self.num_bodies == 0:
            print(f'QualisysClient: WARNING -- No rigid bodies were found in params')
            return
        for body in self.bodies_to_track:
            try:
                body['index'] = names.index(body['name'])
            except ValueError:
                body['index'] = None
                # CHANGED: double quotes inside the f-string (works on all Python 3 versions)
                print(f'QualisysClient: WARNING -- Rigid body {body["name"]} was not found in params')
        await self.connection.stream_frames(
            components=['6d'],
            on_packet=self._on_packet,
        )

    def _on_packet(self, packet):
        header, bodies = packet.get_6d()

        if bodies is None:
            print(f'QualisysClient: ERROR -- No rigid bodies were found in packet')
            return

        if len(bodies) != self.num_bodies:
            print(f'QualisysClient: ERROR -- The wrong number of rigid bodies were found in packet')
            return

        # Get time in seconds, with respect to the qualisys clock
        t = packet.timestamp / 1e6

        for body in self.bodies_to_track:
            # Only proceed if the body has an index
            if body.get('index') is None:
                continue

            # Log the time
            self.data[body['name']]['time'].append(t)

            # Get the position and orientation
            position, orientation = bodies[body['index']]

            # Only proceed if the position and orientation are valid
            if not np.all(np.isfinite(position)):
                self.data[body['name']]['x'].append(np.nan)
                self.data[body['name']]['y'].append(np.nan)
                self.data[body['name']]['z'].append(np.nan)
                self.data[body['name']]['yaw'].append(np.nan)
                self.data[body['name']]['pitch'].append(np.nan)
                self.data[body['name']]['roll'].append(np.nan)
                continue

            # Express the position in units of meters
            x, y, z = np.array(position) / 1e3

            # Express the orientation as ZYX body-fixed Euler angles in units of radians
            R = Rotation.from_matrix(np.reshape(orientation.matrix, (3, -1), order='F'))
            yaw, pitch, roll = R.as_euler('ZYX', degrees=False)

            # Log the position and orientation
            self.data[body['name']]['x'].append(x)
            self.data[body['name']]['y'].append(y)
            self.data[body['name']]['z'].append(z)
            self.data[body['name']]['yaw'].append(yaw)
            self.data[body['name']]['pitch'].append(pitch)
            self.data[body['name']]['roll'].append(roll)

            # CHANGED (moved up): express the orientation as a unit quaternion and
            # collect position and orientation together as a pose. This used to
            # happen only after the "has a queue" check, so bodies without a
            # callback never got a pose.
            qx, qy, qz, qw = R.as_quat(scalar_first=False)
            pose = [x, y, z, qx, qy, qz, qw]

            # Only proceed if the pose is valid
            if not np.all(np.isfinite(pose)):
                continue

            # NEW: store the latest pose so it can be read with get_latest()
            body['latest'].update(pose, t)

            # NEW: optional fast path for consumers that must see every valid
            # pose and never block. Runs in the mocap thread, so keep it tiny.
            direct = body.get('direct_callback')
            if direct is not None:
                try:
                    direct(pose, t)
                except Exception as e:
                    print(f'QualisysClient: Error in direct_callback ({e})')

            # Only proceed if the body has a queue (i.e., if a callback was specified)
            if body['queue'] is None:
                continue

            # Only proceed if the queue is empty, so we never create a backlog of poses
            if not body['queue'].empty():
                continue

            # Put pose in queue
            body['queue'].put(pose)

    async def _close(self):
        if self.connection is None:
            return
        try:
            await self.connection.stream_frames_stop()
        except Exception as e:
            print(f'QualisysClient: Error while stopping the stream ({e})')
        try:
            self.connection.disconnect()
        except Exception as e:
            print(f'QualisysClient: Error while disconnecting ({e})')


###################################
# QUICK TEST: python clients_battle.py
# Prints live data from one rigid body for 15 seconds. No drone needed.

if __name__ == '__main__':
    BODY_NAME = 'marker_deck_40'   # <-- change to the name of your second drone's body

    qualisys = QualisysClient([{'name': BODY_NAME, 'callback': None}])
    try:
        qualisys.wait_for_data(BODY_NAME)
        qualisys.monitor(duration=15.)
    finally:
        qualisys.close()