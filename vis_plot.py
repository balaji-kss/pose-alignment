import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.linalg import svd
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

def plot_3d_joints(ax, joints_3d, joints_2d, color, conf_thresh, shift=[0.0, 0.0], name="hmr"):

    JOINT_PAIRS_H36M = [
    (0, 1),
    (1, 4),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (1, 14),
    (4, 11),
    (11, 12),
    (12, 13),
    (14, 15),
    (15, 16),
    (8, 9),
    (9, 10),
    (14, 7),
    (7, 11),
    (14, 8),
    (8, 11),
    ]

    for i, connection in enumerate(JOINT_PAIRS_H36M):
        if joints_2d is not None:
            if joints_2d[connection[0], 2] < conf_thresh or joints_2d[connection[1], 2] < conf_thresh: dcolor = 'black'
            else:dcolor = color
        else:dcolor = color
        xs = [-joints_3d[connection[0], 0] + shift[0], -joints_3d[connection[1], 0] + shift[0]]
        ys = [joints_3d[connection[0], 1] + shift[1], joints_3d[connection[1], 1] + shift[1]]
        zs = [joints_3d[connection[0], 2], joints_3d[connection[1], 2]]
        if i == 0: ax.plot(xs, ys, zs, 'o-', color=dcolor, label=name)
        else:ax.plot(xs, ys, zs, 'o-', color=dcolor)

def get_3d_pose(pose3d_path, num_people, name="hmr"):

    with open(pose3d_path, 'rb') as f:
        poses3d = pickle.load(f)

    num_frames = len(poses3d)

    kpts3d_data = np.zeros((num_people, num_frames, 17, 3), dtype='float')
    kpts3d_data[:, :, :, :] = np.nan

    kpts2d_data = np.zeros((num_people, num_frames, 17, 2), dtype='float')
    kpts2d_data[:, :, :, :] = np.nan

    for i in range(len(poses3d)):
        for j in range(len(poses3d[i])):
            track_id = poses3d[i][j]['track_id']
            assert track_id >= 0
            kpts3d = poses3d[i][j]['keypoints_3d']
            kpts3d_data[track_id, i] = kpts3d
            if name == "hmr": kpts2d = poses3d[i][j]['keypoints_hmr'][:, :2]
            if name == "vp3d": kpts2d = poses3d[i][j]['keypoints'][:, :2]
            kpts2d_data[track_id, i] = kpts2d

    return kpts3d_data, kpts2d_data


def shift_pose_3d(joints_3d):

    xm, ym, zm = np.mean(joints_3d, axis = 0)

    joints_3d[:, 0] -= xm
    joints_3d[:, 1] -= ym
    joints_3d[:, 2] -= zm

    return joints_3d

def get_height(joints_3d):

    head = joints_3d[10]
    thorax = 0.5 * (joints_3d[11]  + joints_3d[14])

    lshoulder = joints_3d[11]
    rshoulder = joints_3d[14]

    lhip = joints_3d[4]
    rhip = joints_3d[1]

    lknee = joints_3d[5]
    rknee = joints_3d[2]

    lankle = joints_3d[6]
    rankle = joints_3d[3]

    head_neck = np.linalg.norm(head - thorax)
    lshoulder_hip = np.linalg.norm(lshoulder - lhip)
    rshoulder_hip = np.linalg.norm(rshoulder - rhip)
    # shoulder_hip = 0.5 * (lshoulder_hip + rshoulder_hip)
    shoulder_hip = max(lshoulder_hip, rshoulder_hip)

    lknee_hip = np.linalg.norm(lknee - lhip)
    rknee_hip = np.linalg.norm(rknee - rhip)
    # knee_hip = 0.5 * (lknee_hip + rknee_hip)
    knee_hip = max(lknee_hip, rknee_hip)

    lankle_knee = np.linalg.norm(lankle - lknee)
    rankle_knee = np.linalg.norm(rankle - rknee)
    # ankle_knee = 0.5 * (lankle_knee + rankle_knee)
    ankle_knee = max(lankle_knee, rankle_knee)

    tot_height = head_neck + shoulder_hip + knee_hip + ankle_knee

    return np.round(tot_height, 3)

def procrustes_local(data1, data2):
    def orthogonal_procrustes(A, B):
        # Compute the SVD of the matrix B^T A
        u, w, vt = svd(B.T.dot(A).T)
        R = u.dot(vt)
        scale = w.sum()
        return R, scale        
        # u, w, vt = svd(B.T @ A)
        # R = u @ vt
        # if np.linalg.det(R) < 0:
        #     # Reflection detected, adjust for proper rotation
        #     u[:, -1] *= -1
        #     R = u @ vt
        # scale = w.sum()
        # return R, scale    
    # Translate all the data to the origin
    mtx1 = data1 - np.mean(data1, 0)
    mtx2 = data2 - np.mean(data2, 0)

    translation1 = np.mean(data1, 0)
    translation2 = np.mean(data2, 0)
    translation = translation1 - translation2

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # Change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # Transform mtx2 to minimize disparity
    R, scale = orthogonal_procrustes(mtx1, mtx2)
    mtx2_transformed = np.dot(mtx2, R.T) * scale

    # Measure the dissimilarity between the two datasets
    disparity = np.sum(np.square(mtx1 - mtx2_transformed))

    return mtx1, mtx2_transformed, disparity, R, scale, translation

def transform_points(points, R, scale, translation):
    # Center the points
    mean_points = np.mean(points, axis=0)
    centered_points = points - mean_points

    centered_points /= np.linalg.norm(centered_points)

    # Apply rotation
    rotated_points = np.dot(centered_points, R.T)

    # Apply scaling
    scaled_points = rotated_points * scale

    # Apply translation
    transformed_points = scaled_points #+ translation

    return transformed_points

def vis_pose3d(pose3d_paths):

    num_people = 1
    kpts3d_data1, kpts2d_data1 = get_3d_pose(pose3d_paths[0], num_people, name="hmr")
    kpts3d_data2, kpts2d_data2 = get_3d_pose(pose3d_paths[1], num_people, name="hmr")

    num_frames = kpts3d_data1.shape[1]
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.view_init(elev=10, azim=0)

    for t in range(num_frames):
        
        ax.clear()
        gmins, gmaxs = None, None

        print('time t ', t)

        for idx in range(1):
            hmr_joints_3d1  = kpts3d_data1[idx, t]   
            hmr_joints_2d1  = kpts2d_data1[idx, t]

            hmr_joints_3d2  = kpts3d_data2[idx, t]
            hmr_joints_2d2  = kpts2d_data2[idx, t]    
            
            # standard_pose_ref_3d_np = np.array([standard_pose_ref_3d[k] for k in h36m_joint_indices.keys()])
            # procrustes_mask = np.array([True if "SHOULDER" in k or "HIP" in k or "ANKLE" in k else False for k in h36m_joint_indices.keys()])

            # c1_part, start_part_trans_local, _, r1, s1, t1 = procrustes_local(standard_pose_ref_3d_np[procrustes_mask], hmr_joints_3d1[procrustes_mask])
            # c2_part, end_part_trans_local,   _, r2, s2, t2  = procrustes_local(standard_pose_ref_3d_np[procrustes_mask], hmr_joints_3d2[procrustes_mask])

            # hmr_joints_3d1 = transform_points(hmr_joints_3d1, r1, s1, t1)
            # hmr_joints_3d2  = transform_points(hmr_joints_3d2, r2, s2, t2)

            # c1, _, _ = procrustes(standard_pose_ref_3d_np, hmr_joints_3d1)
            # canonical_ref = np.zeros_like(standard_pose_ref_3d_np)
            # canonical_ref = c1 # start_part_trans_local#  

            # plot_3d_joints(ax, canonical_ref, joints_2d=None, color='green', conf_thresh=None, shift=[0.0, 0.0], name="hmr")

            R, _ = orthogonal_procrustes(hmr_joints_3d1, hmr_joints_3d2)
            hmr_joints_3d2 = hmr_joints_3d2 @ R

            print(hmr_joints_2d1.shape)
            plot_3d_joints(ax, hmr_joints_3d1, joints_2d=None, color='blue', conf_thresh=None, shift=[0.0, 0.0], name="hmr")
            h1 = get_height(hmr_joints_3d1)
            plot_3d_joints(ax, hmr_joints_3d2, joints_2d=None, color='red', conf_thresh=None, shift=[0.0, 0.0], name="hmr")
            h2 = get_height(hmr_joints_3d2)

            print('h1 ', h1, ' h2 ', h2)

        gmin = np.min(hmr_joints_3d1) 
        gmax = np.max(hmr_joints_3d1) 

        ax.set_xlim([gmin - 0.1, gmax + 0.1])  # Adjust as necessary
        ax.set_ylim([gmin - 0.1, gmax + 0.1])  # Adjust as necessary
        ax.set_zlim([gmin - 0.05, gmax + 0.05]) 

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        plt.pause(0.0000000000001)

if __name__ == '__main__':

    json_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/Closing_Overhead_Bin/"
    json_paths = glob.glob(os.path.join(json_dir, '*.json'), recursive=False)
    json_paths.sort()

    pose_dir = "/home/tumeke-balaji/Documents/results/human-mesh-recovery/delta_data_videos_poses_res/Closing_Overhead_Bin/videos/" 

    standard_pose_ref_3d = {
            "HEAD": np.array([0, 0, 2.0]),
            "NECK": np.array([0, 0, 1.9]),
            "THORAX": np.array([0, 0, 1.88]),
            "PELV": np.array([0, 0, 1.1]),
            "RSHOULDER": np.array([0.3, 0.0, 1.88]),
            "LSHOULDER": np.array([-0.3, -0.0, 1.88]),
            "SPINE": np.array([0.0, 0.0, 1.49]),
            "RHIP":  np.array([0.3, 0.0, 1.1]),
            "LHIP": np.array([-0.3, -0.0, 1.1]),
            "RKNEE": np.array([0.3, 0.0, 0.6]),
            "LKNEE":  np.array([-0.3, -0.0, 0.6]),
            "RANKLE":  np.array([0.3, 0.0, 0.0]),
            "LANKLE":  np.array([-0.3, -0.0, 0.0]),
            "RELBOW": np.array([0.4, 0.0, 1.5]),
            "LELBOW": np.array([-0.4, -0.0, 1.5]),
            "RWRIST": np.array([0.4, 0.0, 1.1]),
            "LWRIST": np.array([-0.4, -0.0, 1.1])
        }     

    h36m_joint_indices = {
        "PELV": 0,
        "RHIP": 1,
        "RKNEE": 2,
        "RANKLE": 3,
        "LHIP": 4,
        "LKNEE": 5,
        "LANKLE": 6,
        "SPINE": 7,
        "THORAX": 8,
        "NECK": 9,
        "HEAD": 10,
        "LSHOULDER": 11,
        "LELBOW": 12,
        "LWRIST": 13,
        "RSHOULDER": 14,
        "RELBOW": 15,
        "RWRIST": 16
    }  
    
    for json_path in json_paths:
        name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
        if name1 not in ["baseline4", "baseline5", "baseline12", "baseline13", "baseline19"]:continue
        pose3d_path1 = os.path.join(pose_dir, name1 + '_pose3d.pkl')
        pose3d_path2 = os.path.join(pose_dir, name2 + '_pose3d.pkl')
        print(name1, name2)
        vis_pose3d([pose3d_path1, pose3d_path2])