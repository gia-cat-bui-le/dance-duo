import pickle

import numpy as np
import scipy.stats as stats
import smplx
import torch
from scipy.spatial.transform import Rotation as R

ALPHA = 7
BETA = 3


def romp2edge_smpl(romp_data):
    smpl_trans = torch.from_numpy(romp_data['cam_trans'])
    global_orient = romp_data['global_orient']
    body_pose = romp_data['body_pose']
    # smpl_pose = np.concatenate([global_orient, body_pose], axis=1)
    # smpl_pose = torch.from_numpy(smpl_pose)
    smpl_betas = torch.from_numpy(romp_data['smpl_betas'])
    # smpl_model = smplx.SMPL("../dance_data/romp/smpl", body_pose=smpl_pose, transl=smpl_trans, betas=smpl_betas)
    smpl_model = smplx.SMPL("./dance_data/romp/smpl", body_pose=torch.from_numpy(body_pose), transl=smpl_trans,
                            betas=smpl_betas)
    full_pose = smpl_model.forward(return_full_pose=True).full_pose.detach().cpu().numpy().reshape(-1, 24, 3)
    new_smpl_poses = np.concatenate([global_orient, smpl_model.body_pose.cpu().detach().numpy()], axis=1)
    result_data = {
        # 'smpl_trans': smpl_model.transl.cpu().detach().numpy(),
        'smpl_poses': new_smpl_poses,
        'full_pose': full_pose,
        # for test visualization only
        # 'smpl_scaling': [1],
    }
    return result_data


def extract_full_pose_from_generated_dance(frame, generated_data):
    ref_data = generated_data['full_pose'][frame]
    # print(f"full pose generated dance frame {frame}: {ref_data}")
    return generated_data['full_pose'][frame]


def normalized_generated_pose(generated_pose):
    tmp = generated_pose[0]
    for i in range(22):
        generated_pose[i] -= tmp
    return generated_pose


def furthest_dist(generated_pose):
    ret = 0

    for i in range(len(generated_pose) - 1):
        for j in range(i, len(generated_pose)):
            ret = max(ret, np.linalg.norm(generated_pose[i] - generated_pose[j]))

    return ret


# Mean Per Joint Position Error
def mpjpe(frame, capture_data, reference_data):
    generated_pose = extract_full_pose_from_generated_dance(frame, reference_data)
    capture_pose = capture_data['full_pose'].reshape(24, 3)
    # print(f"generated_pose: {generated_pose}")
    generated_pose = normalized_generated_pose(generated_pose)
    # print(f"after normalize: {generated_pose}")
    # print(f"capture_pose: {capture_pose}")
    num_joints = 22  # romp not capture hands
    total_error = 0
    norm_dist = furthest_dist(generated_pose)

    for i in range(0, num_joints):
        capture_joint, generated_joint = capture_pose[i], generated_pose[i]
        error = np.linalg.norm(capture_joint - generated_joint) / norm_dist
        # print(f"joint {i} error = {error}")
        total_error += error

    # print(f"total error: {total_error}")
    return total_error / num_joints


def extract_smpl_poses_from_generated_dance(frame, generated_dance):
    ref_data = generated_dance['smpl_poses'][frame]
    # print(f"ref_rot: {ref_data}")
    return ref_data


def get_euclidean_rot(rot_vec):
    rot = R.from_rotvec(rot_vec, degrees=False)
    euclidean_rot = rot.as_euler('xyz', degrees=False)
    return euclidean_rot


def unit_vector(vec):
    return vec / np.linalg.norm(vec)


# Mean Per Joint Angle Error
def mpjae(frame, capture_data, reference_data):
    total_error = 0
    num_joints = 22

    generated_pose = extract_smpl_poses_from_generated_dance(frame, reference_data)
    capture_pose = capture_data['smpl_poses']
    # print(f"generated pose: {generated_pose}")
    # print(f"capture pose: {capture_pose}")

    for i in range(0, num_joints):
        generated_rot = generated_pose[i:i + 3]
        capture_rot = capture_pose[0][i:i + 3]
        print(f"gen: {generated_rot}, cap: {capture_rot}")
        generated_rot = get_euclidean_rot(generated_pose[i:i + 3])
        capture_rot = get_euclidean_rot(capture_pose[0][i:i + 3])
        generated_rot_u = unit_vector(generated_rot)
        capture_rot_u = unit_vector(capture_rot)
        error = np.arccos(np.clip(np.dot(generated_rot_u, capture_rot_u), -1.0, 1.0)) / 2.0 / np.pi
        print(f"rot error joint {i}: {error}")
        total_error += error

    return total_error / num_joints


def accuracy_to_score(accuracy):
    # Map accuracy (0 to 1) to beta distribution
    return stats.beta.ppf(accuracy, ALPHA, BETA)


def calculate_score(music_genre):
    pkl_path = f"dance_data/{music_genre}.pkl"
    generated_dance = pickle.load(open(pkl_path, "rb"))
    capture_result_path = "vid_demo/output/video_results.npz"
    capture_result_data = np.load(capture_result_path, allow_pickle=True)['results'][()]
    frames = list(capture_result_data)
    num_capture_frames = len(frames)
    num_generated_frames = len(generated_dance['smpl_trans'])
    total_point = 0

    for f in range(num_generated_frames):
        corresponding_capture_frame = int(f * num_capture_frames / num_generated_frames)
        cur_frame = frames[corresponding_capture_frame]
        # print(f"cal frame {f}, corresponding frame: {corresponding_capture_frame}")
        # print(cur_frame)
        # print(type(capture_result_data))
        capture_data = capture_result_data[cur_frame]
        capture_data = romp2edge_smpl(capture_data)
        pose_error = 1 - mpjpe(f, capture_data, generated_dance)
        angle_error = 1 - mpjae(f, capture_data, generated_dance)
        accuracy = (pose_error + angle_error) / 2
        total_point += accuracy_to_score(accuracy)

    return round(total_point * 100 / num_generated_frames, 2)
