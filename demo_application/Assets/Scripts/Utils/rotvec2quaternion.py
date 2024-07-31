import pickle
import json

import sys
import subprocess
import logging

def install_scipy():
    try:
        from scipy.spatial.transform import Rotation as R
    except ModuleNotFoundError as e:
        python_exe = sys.executable
        subprocess.call([python_exe, "-m", "ensurepip"])
        ret = subprocess.run([python_exe, "-m", "pip", "install", "scipy"])
        if ret.returncode != 0:
            logging.error(f"Failed to install scipy")
        else:
            logging.info(f"Install scipy!")
    except Exception as e:
        raise e
    
def install_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as e:
        python_exe = sys.executable
        subprocess.call([python_exe, "-m", "ensurepip"])
        ret = subprocess.run([python_exe, "-m", "pip", "install", "numpy"])
        if ret.returncode != 0:
            logging.error(f"Failed to install scipy")
        else:
            logging.info(f"Install scipy!")
    except Exception as e:
        raise e


install_scipy()
install_numpy()
from scipy.spatial.transform import Rotation as R
import numpy as np

pkl_path = "./Assets/Resources/Animation/pkl/013_normed.pkl"
json_path = "./Assets/Resources/Animation/json/013.json"

def read_pkl_file(pkl_path):
    with open(pkl_path, "rb") as pkl_file:
        data = pickle.load(pkl_file)
    return data

def blender2unity(blender_rot):
    # blender_euler_radians = np.radians(blender_euler_degrees)
    # Convert Blender Euler angles (XYZ) to a quaternion
    # blender_quat = blender_rot.as_quat()
    # adjusted_quat = np.array([blender_quat[3], -blender_quat[0], blender_quat[2], -blender_quat[1]])
    # Convert adjusted quaternion to Unity's Euler angles (XYZ)

    # unity_euler = blender_rot.as_euler('ZXY', degrees=True)
    # unity_euler = [unity_euler[0], unity_euler[2], -unity_euler[1]]
    # unity_euler = blender_rot.as_euler('xyz', degrees=True)

    blender_euler = blender_rot.as_euler('xyz', degrees = False)
    z_axis = R.from_euler('z', blender_euler[2], degrees = False)
    x_aixs = z_axis.inv() * R.from_euler('x', blender_euler[0], degrees = False)

    # # x_aixs = R.from_euler('x', tmp[0], degrees = False) * z_axis.inv()
    # # unity_euler = R.from_euler('y', tmp[1], degrees = False) * x_aixs.inv()
    unity_euler = x_aixs.inv() * R.from_euler('y', blender_euler[1], degrees = False)
    unity_euler = unity_euler.as_euler('xyz', degrees = True)


    return unity_euler
    # return blender_quat

def rotvec2quaternion(data):
    smpl_poses = data['smpl_poses']
    n_frame = len(smpl_poses)
    n_joints = int(len(smpl_poses[0]) / 3)

    smpl_poses_quaternion = []

    with open("./Assets/Resources/013.txt", "w") as f:

        for frame in range(n_frame):
            frame_quaternion = []

            for joint_idx in range(n_joints):
                rot = smpl_poses[frame][joint_idx * 3 : joint_idx * 3 + 3]
                # print(f'{frame}, {joint_idx} rot {rot}')
                rot_scipy = R.from_rotvec([rot[0], rot[1], rot[2]])
                # rot = R.from_rotvec([rot[0], -rot[2], rot[1]])
                # rot = rot.as_euler('xyz', degrees = True)
                # print(f'euler {rot}')
                # unity_euler = rot.as_quat(canonical=False, scalar_first=False)
                # unity_euler = rot.as_quat()
                unity_euler = blender2unity(rot_scipy)
                f.write(f'{frame}_{joint_idx}:\n\tbefore: {rot_scipy.as_euler('xyz', degrees=True)}\n\tafter: {unity_euler}\n')
                # unity_euler = np.array([rot[0], -rot[2], rot[1]])
                # if (joint_idx == 0):
                #     unity_euler = np.array([rot[0] - 90, rot[1], rot[2]])
                # else:
                #     unity_euler = np.array([rot[0], rot[1], rot[2]])
                # print(f"before: {unity_euler}")

                # rotate_90_degrees = R.from_euler('xyz', [-90, 0, 0], degrees = True)
                # unity_euler = rotate_90_degrees.apply(unity_euler)
                # print(f"after: {unity_euler}")
                
                # unity_euler = np.array([rot[0], rot[1], rot[2]])
                # blender_rot = rot_scipy.as_euler('xyz', degrees=False)
                # f.write(f'{frame}_{joint_idx}:\n\tunity: {rot_scipy.as_euler('xyz', degrees=True)}\n\tblender: {rot_scipy.as_euler('xyz', degrees=False)}\n\toriginal: {rot}\n\tquaternion: {rot_scipy.as_quat()}\n')
                frame_quaternion.append(unity_euler[0])
                frame_quaternion.append(unity_euler[1])
                frame_quaternion.append(unity_euler[2])
                # frame_quaternion.append(unity_euler[3])

            smpl_poses_quaternion.append(frame_quaternion)

    data['smpl_poses'] = smpl_poses_quaternion
    return data

def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(element) for element in obj]
    if isinstance(obj, tuple):
        return tuple(convert_to_serializable(element) for element in obj)
    return obj

def write2json(data, json_path):
    serializable_data = convert_to_serializable(data)

    # Write the data to a .json file
    with open(json_path, "w") as json_file:
        json.dump(serializable_data, json_file, indent=4)

    print(f'write 2 json: {json_path}')

import scipy
print(scipy.__version__)
data = read_pkl_file(pkl_path)
data = rotvec2quaternion(data)
write2json(data, json_path)