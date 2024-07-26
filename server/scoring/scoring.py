import pickle
import numpy as np
import smplx
import torch
from typing import Optional


def romp2edge_smpl(romp_data):
    smpl_trans = torch.from_numpy(romp_data['cam_trans'])
    global_orient = romp_data['global_orient']
    body_pose = romp_data['body_pose']
    smpl_pose = np.concatenate([global_orient, body_pose], axis=1)
    smpl_pose = torch.from_numpy(smpl_pose)
    smpl_betas = torch.from_numpy(romp_data['smpl_betas'])
    # smpl_model = smplx.SMPL("../dance_data/romp/smpl", body_pose=smpl_pose, transl=smpl_trans, betas=smpl_betas)
    smpl_model = smplx.SMPL("../dance_data/romp/smpl", body_pose=torch.from_numpy(body_pose), transl=smpl_trans, betas=smpl_betas)
    full_pose = smpl_model.forward(return_full_pose=True).full_pose.detach().cpu().numpy().reshape(-1, 24, 3)
    new_smpl_poses = np.concatenate([global_orient, smpl_model.body_pose.cpu().detach().numpy()], axis=1)
    result_data = {
        'smpl_trans': smpl_model.transl.cpu().detach().numpy(),
        'smpl_poses': new_smpl_poses,
        'full_pose': full_pose,
        'smpl_scaling': [1],
    }
    return result_data


def extract_frames_from_result(reference_path):
    result_data = np.load(reference_path, allow_pickle=True)['results'][()]
    frames = list(result_data)
    return frames


def extract_full_pose_from_generated_dance(frame, generated_data):
    ref_data = generated_data['full_pose'][frame]
    print(f"full pose generated dance frame {frame}: {ref_data}")
    return generated_data['full_pose'][frame]


def extract_full_pose_from_user(frame, user_dance_data):
    data = user_dance_data['']


# Mean Per Joint Position Error
def mpjpe(frame):
    total_point = 0

    return total_point


# Mean Per Joint Angle Error
def mpjae(frame):
    error = 0

    return error


def calculate_score(music_genre):
    pkl_path = f"dance_data/{music_genre}.pkl"
    frames = extract_frames_from_result(pkl_path)
    user_num_frames = len(frames)


# Load the .npz file
npz_file = np.load('../vid_demo/output/00000000.npz', allow_pickle=True)['results'][()]
data = romp2edge_smpl(npz_file)
print(data)
pickle.dump(data, open("../test.pkl", "wb"))

with open("../dance_data/gHO_sBM_cAll_d20_mHO5_ch02_normed_26_0.pkl", "rb") as pkl_file:
    data = pickle.load(pkl_file)
print(data)
print("Data has been written to output.txt")
