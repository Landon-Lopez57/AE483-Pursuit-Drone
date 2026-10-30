"""Functions that you write.

Add your own code here, not in tools.py. Keeping them separate means you can
`git pull` our updates all semester without conflicts.
"""

import numpy as np
from scipy.spatial.transform import Rotation

# Makes everything in tools.py available here, just as if your code were at the
# bottom of that file.
from ae483.tools import *


def transform_data_mocap(raw_data):
    # Copy raw data
    data = {}
    for key, val in raw_data.items():
        data[key] = val.copy()

    # Define parameters
    d_1 = 0.018
    d_2 = 0.010
    
    # Pose of drone body frame in active marker frame
    R_inA_ofB = np.eye(3)                   # <-- FIXME
    p_inA_ofB = np.array([0., 0., -d_1])      # <-- FIXME

    ####################################
    # START OF ANALYSIS AT TIME STEP 0
    #
    
    # Pose of drone world frame in active marker frame
    R_inA_ofW = np.eye(3)                   # <-- FIXME
    p_inA_ofW = np.array([0., 0., -1*(d_1+d_2)])      # <-- FIXME

    # Get measurements of (x, y, z) and (psi, theta, phi) from mocap
    x, y, z = data['x'][0], data['y'][0], data['z'][0]
    psi, theta, phi = data['yaw'][0], data['pitch'][0], data['roll'][0]

    # Pose of active marker frame in mocap world frame
    R_inQ_ofA = Rotation.from_euler(
        'ZYX',
        [psi, theta, phi],
        degrees = False,
    ).as_matrix()              # <-- FIXME
    p_inQ_ofA = np.array([x,y,z])      # <-- FIXME
    
    # Pose of drone world frame in mocap world frame
    R_inQ_ofW = Rotation.from_euler(
        'ZYX',
        [psi, theta, phi],
        degrees = False,
    ).as_matrix()                   # <-- FIXME
    p_inQ_ofW = np.array([x,y,z-(d_1+d_2)])      # <-- FIXME
    
    # Pose of mocap world frame in drone world frame
    R_inW_ofQ = R_inQ_ofW.T                   # <-- FIXME
    p_inW_ofQ = np.matmul(-1*(R_inQ_ofW.T), p_inQ_ofW)     # <-- FIXME

    #
    # END OF ANALYSIS AT TIME STEP 0
    ####################################

    for i in range(len(data['time'])):

        ####################################
        # START OF ANALYSIS AT TIME STEP i
        #

        # Get measurements of (x, y, z) and (psi, theta, phi) from mocap
        x, y, z = data['x'][i], data['y'][i], data['z'][i]
        psi, theta, phi = data['yaw'][i], data['pitch'][i], data['roll'][i]

        # Pose of active marker deck in mocap world frame
        R_inQ_ofA = Rotation.from_euler(
        'ZYX',
        [psi, theta, phi],
        degrees = False,
        ).as_matrix()                # <-- FIXME

        p_inQ_ofA = np.array([x, y, z])  # <-- FIXME

        # Pose of drone body frame in mocap world frame
        R_inQ_ofB = np.matmul(R_inQ_ofA, R_inA_ofB)              # <-- FIXME
        p_inQ_ofB = p_inQ_ofA + np.matmul(R_inQ_ofA, p_inA_ofB)

        # Pose of drone body frame in drone world frame
        R_inW_ofB = np.matmul(R_inW_ofQ, R_inQ_ofB)               # <-- FIXME
        p_inW_ofB = p_inW_ofQ + np.matmul(R_inW_ofQ, p_inQ_ofB)  # <-- FIXME

        # Replace measurements of (x, y, z) and (phi, theta, psi) from mocap
        data['x'][i], data['y'][i], data['z'][i] = p_inW_ofB[0], p_inW_ofB[1], p_inW_ofB[2]               # <-- FIXME
        data['yaw'][i], data['pitch'][i], data['roll'][i] = Rotation.from_matrix(R_inW_ofB).as_euler('ZYX')     # <-- FIXME

        #
        # END OF ANALYSIS AT TIME STEP i
        ####################################
    
    # Return the result
    return data

import math


def quaternion_to_euler(x, y, z, w):
  # Roll (X-axis rotation)
  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = math.atan2(sinr_cosp, cosr_cosp)

  # Pitch (Y-axis rotation)
  sinp = 2.0 * (w * y - z * x)
  if abs(sinp) >= 1:
    pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
  else:
    pitch = math.asin(sinp)

  # Yaw (Z-axis rotation)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = math.atan2(siny_cosp, cosy_cosp)

  return roll, pitch, yaw


def mocap_transform(pose):

    x, y, z = pose[0], pose[1], pose[2]
    phi, theta, psi = quaternion_to_euler(pose[3], pose[4], pose[5], pose[6])

    # Define parameters
    d_1 = 0.018
    d_2 = 0.010
    
    # Pose of drone body frame in active marker frame
    R_inA_ofB = np.eye(3)                   # <-- FIXME
    p_inA_ofB = np.array([0., 0., -d_1])      # <-- FIXME

    ####################################
    # START OF ANALYSIS AT TIME STEP 0
    #
    
    # Pose of drone world frame in active marker frame
    R_inA_ofW = np.eye(3)                   # <-- FIXME
    p_inA_ofW = np.array([0., 0., -1*(d_1+d_2)])      # <-- FIXME


    # Pose of active marker frame in mocap world frame
    R_inQ_ofA = Rotation.from_euler(
        'ZYX',
        [psi, theta, phi],
        degrees = False,
    ).as_matrix()              # <-- FIXME
    p_inQ_ofA = np.array([x,y,z])      # <-- FIXME
    
    # Pose of drone world frame in mocap world frame
    R_inQ_ofW = Rotation.from_euler(
        'ZYX',
        [psi, theta, phi],
        degrees = False,
    ).as_matrix()                   # <-- FIXME
    p_inQ_ofW = np.array([x,y,z-(d_1+d_2)])      # <-- FIXME
    
    # Pose of mocap world frame in drone world frame
    R_inW_ofQ = R_inQ_ofW.T                   # <-- FIXME
    p_inW_ofQ = np.matmul(-1*(R_inQ_ofW.T), p_inQ_ofW)     # <-- FIXME

    #
    # END OF ANALYSIS AT TIME STEP 0
    ####################################



    ####################################
    # START OF ANALYSIS AT TIME STEP i
    #

    # Get measurements of (x, y, z) and (psi, theta, phi) from mocap
    x, y, z = data['x'][i], data['y'][i], data['z'][i]
    psi, theta, phi = data['yaw'][i], data['pitch'][i], data['roll'][i]

    # Pose of active marker deck in mocap world frame
    R_inQ_ofA = Rotation.from_euler(
    'ZYX',
    [psi, theta, phi],
    degrees = False,
    ).as_matrix()                # <-- FIXME

    p_inQ_ofA = np.array([x, y, z])  # <-- FIXME

    # Pose of drone body frame in mocap world frame
    R_inQ_ofB = np.matmul(R_inQ_ofA, R_inA_ofB)              # <-- FIXME
    p_inQ_ofB = p_inQ_ofA + np.matmul(R_inQ_ofA, p_inA_ofB)

    # Pose of drone body frame in drone world frame
    R_inW_ofB = np.matmul(R_inW_ofQ, R_inQ_ofB)               # <-- FIXME
    p_inW_ofB = p_inW_ofQ + np.matmul(R_inW_ofQ, p_inQ_ofB)  # <-- FIXME

    # Replace measurements of (x, y, z) and (phi, theta, psi) from mocap
    data['x'][i], data['y'][i], data['z'][i] = p_inW_ofB[0], p_inW_ofB[1], p_inW_ofB[2]               # <-- FIXME
    data['yaw'][i], data['pitch'][i], data['roll'][i] = Rotation.from_matrix(R_inW_ofB).as_euler('ZYX')     # <-- FIXME



def precise_time_shift(raw_data_mocap, t, z_drone, t_shift_guess,
                       do_transform=True, t_min_offset=0.,
                       t_shift_rng=1., t_shift_res=0.05):
    """
    Find a value of t_shift that precisely aligns mocap data with
    drone data by sampling candidate values in the interval

        [t_shift_guess - t_shift_rng, t_shift_guess + t_shift_rng]
    
    at the resolution t_shift_res and returning the value that
    produces the minimum RMSE between z_drone and z_mocap.
    """

    # FIXME - your code here, to...
    #   1) Create an array with a range of time shifts over which to search
    t_shifts = np.linspace(
    t_shift_guess - t_shift_rng,                        # lower bound
    t_shift_guess + t_shift_rng,                        # upper bound
    int(1 + np.ceil((2 * t_shift_rng) / t_shift_res)),  # number of samples
)
    #   2) Print a warning if computation time is likely to be long
    if len(t_shifts) > 50:
        print(f'========================================================================')
        print(f'WARNING (sync_data_mocap)\n')
        print(f'Searching the interval\n')
        print(f'  t_shift_bnds=[{min(t_shifts):4.2f}, {max(t_shifts):4.2f}]\n')
        print(f'with the resolution\n')
        print(f'  t_shift_res={t_shift_res:4.2f}\n')
        print(f'means computing RMSE for {len(t_shifts)} possible time shifts, which is likely')
        print(f'to be slow. You may want to decrease the size of the search interval')
        print(f'or increase the resolution.')
        print(f'========================================================================\n')

    #   3) Create an array to hold the RMSE for each time shift
    RMSEs = np.empty_like(t_shifts)
    #   4) Find the RMSE for each time shift
    for i, t_shift in enumerate(t_shifts):
        # Resample mocap data with time shift
        resampled_data_mocap = resample_data_mocap(
            raw_data_mocap,
            t,
            t_shift=t_shift,
            t_min_offset=t_min_offset,
        )

        # Transform mocap data
        if do_transform:
            transformed_data_mocap = transform_data_mocap(resampled_data_mocap)
        else:
            transformed_data_mocap = resampled_data_mocap

        # Get z estimate from mocap data
        z_mocap = transformed_data_mocap['z']

        # Find RMSE between z_mocap and z_drone
        RMSEs[i] = np.sqrt(np.mean((z_mocap - z_drone)**2))

    #   5) Find the index of the minimum RMSE
    i_min = np.argmin(RMSEs) 
    #   6) Find the minimum RMSE
    RMSE_min = RMSEs[i_min] 
    #   7) Find the time shift that gives the minimum RMSE
    t_shift_min = t_shifts[i_min]
    
    # Return the result
    return t_shift_min

def sync_data_mocap(raw_data_mocap, t, z_drone,
                    do_transform=True, t_min_offset=0.,
                    t_shift_rng=1., t_shift_res=0.05):

    # Get rough guess at time shift
    t_shift_guess = rough_time_shift(
        raw_data_mocap,
        t,
        z_drone,
        t_min_offset=t_min_offset,
    )

    # Refine the rough guess to get the time shift that
    # minimizes RMSE between z_drone and z_mocap
    t_shift_min = precise_time_shift(
        raw_data_mocap,
        t,
        z_drone,
        t_shift_guess=t_shift_guess,
        do_transform=do_transform,
        t_min_offset=t_min_offset,
        t_shift_rng=t_shift_rng,
        t_shift_res=t_shift_res,
    )

    # Resample mocap data with this time shift
    resampled_data_mocap = resample_data_mocap(
        raw_data_mocap,
        t,
        t_shift=t_shift_min,
        t_min_offset=t_min_offset,
    )
    
    # Transform mocap data
    if do_transform:
        transformed_data_mocap = transform_data_mocap(resampled_data_mocap)
    else:
        transformed_data_mocap = resampled_data_mocap

    # Return the result
    return transformed_data_mocap
