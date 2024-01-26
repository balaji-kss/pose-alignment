from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
import mmcv
import os
import json
import cv2
import pickle
import numpy as np
from PIL import Image
import time
import math
from numpy.lib.stride_tricks import sliding_window_view
from numpy import linalg as LA
from scipy.spatial import procrustes
import fine_align
import utils
import render
import calc_angles as ca

def get_min_max(joint_3ds):

    gmin, gmax = 1000, -1000 
    for joints_3d in joint_3ds:
        min_ = np.min(joints_3d)
        max_ = np.max(joints_3d)
        gmin = min(gmin, min_)
        gmax = max(gmax, max_)

    return gmin, gmax

def draw_2d_skeletons_3d(frame, kpts_2d, color=(0, 0, 255), conf_thresh=0.35):
    radius = 10
    scale = 1

    for sk_id, sk in enumerate(connections):

        if kpts_2d[sk[0], 2] <= conf_thresh or kpts_2d[sk[1], 2] <= conf_thresh:continue

        pos1 = (int(kpts_2d[sk[0], 0]*scale), int(kpts_2d[sk[0], 1]*scale))
        pos2 = (int(kpts_2d[sk[1], 0]*scale), int(kpts_2d[sk[1], 1]*scale))
        cv2.line(frame, pos1, pos2, color, thickness=radius)

    return frame

def plot_3d(ax, joints_2d, joints_3d_org, color, shiftx = 0.0, idxs = None):

    joints_3d = joints_3d_org.copy()
    joints_3d[:, 0] += shiftx

    for connection in connections:
        if idxs and connection[0] not in idxs and connection[1] not in idxs:continue
        if joints_2d[connection[0], 2] < conf_thresh or joints_2d[connection[1], 2] < conf_thresh: continue
        xs = [-joints_3d[connection[0], 0], -joints_3d[connection[1], 0]]
        ys = [joints_3d[connection[0], 1], joints_3d[connection[1], 1]]
        zs = [joints_3d[connection[0], 2], joints_3d[connection[1], 2]]
        ax.plot(xs, ys, zs, 'o-', color=color)

def align_candidate_pose_low(base_3d_joints, cand_3d_joints):

    base_joints_aligned, cand_joints_aligned = base_3d_joints.copy(), cand_3d_joints.copy()
    # base_joints_aligned (17, 3)
    # cand_joints_aligned (17, 3)

    cand_joints_aligned = ca.align_hip(base_joints_aligned, cand_joints_aligned)

    # center origin
    base_joints_aligned[:, :] -= base_joints_aligned[0, :]
    cand_joints_aligned[:, :] -= cand_joints_aligned[0, :]

    return base_joints_aligned, cand_joints_aligned

def align_candidate_pose_up(base_3d_joints, cand_3d_joints):

    base_joints_aligned, cand_joints_aligned = base_3d_joints.copy(), cand_3d_joints.copy()
    # base_joints_aligned (17, 3)
    # cand_joints_aligned (17, 3)

    cand_joints_aligned = ca.align_shoulder(base_joints_aligned, cand_joints_aligned)

    bmid_shoulder = 0.5 * (base_joints_aligned[11, :] + base_joints_aligned[14, :])
    cmid_shoulder = 0.5 * (cand_joints_aligned[11, :] + cand_joints_aligned[14, :])

    base_joints_aligned[:, :] -= bmid_shoulder
    cand_joints_aligned[:, :] -= cmid_shoulder

    return base_joints_aligned, cand_joints_aligned

def cal_dev(base_joints_3d, cand_joints_3d, bjoints_2d, cjoints_2d, ax2, vis=False):

    if np.all(base_joints_3d == 0.0) or np.all(cand_joints_3d == 0.0):
        return [0] * 10

    base_joints_3d, cand_joints_3d, _ = procrustes(base_joints_3d, cand_joints_3d)

    if vis:
        plot_3d(ax2, bjoints_2d, base_joints_3d, colors[0], shiftx=0.25)
        plot_3d(ax2, cjoints_2d, cand_joints_3d, colors[1], shiftx=0.25)

        vis_vector(ax2, base_joints_3d, shiftx=0.25)
        vis_vector(ax2, cand_joints_3d, shiftx=0.25)
        
    trunk_dev = ca.get_trunk_dev(base_joints_3d, cand_joints_3d)   
    trunk_twist_dev = ca.get_trunk_twist_dev(base_joints_3d, cand_joints_3d)   

    # lower body align
    base_joints_aligned_low, cand_joints_aligned_low = align_candidate_pose_low(base_joints_3d, cand_joints_3d)

    if vis:
        pad = 0.15
        gminn, gmaxn = get_min_max(cand_joints_aligned_low)
        ax2.set_xlim([gminn - pad, gmaxn + pad])  # Adjust as necessary
        ax2.set_ylim([gminn - pad, gmaxn + pad])  # Adjust as necessary
        ax2.set_zlim([gminn - pad, gmaxn + pad])            

    lthigh_dev = ca.get_left_thigh_dev(base_joints_aligned_low, cand_joints_aligned_low)

    rthigh_dev = ca.get_right_thigh_dev(base_joints_aligned_low, cand_joints_aligned_low)

    lleg_dev = ca.get_left_leg_dev(base_joints_aligned_low, cand_joints_aligned_low)

    rleg_dev = ca.get_right_leg_dev(base_joints_aligned_low, cand_joints_aligned_low)

    if False:
        low_idxs = list(range(0, 7))
        plot_3d(ax2, bjoints_2d, base_joints_aligned_low, colors[0], shiftx=0.50, idxs = low_idxs)

        plot_3d(ax2, cjoints_2d, cand_joints_aligned_low, colors[1], shiftx=0.50, idxs = low_idxs)

        vis_vector(ax2, base_joints_aligned_low, shiftx=0.5)
        vis_vector(ax2, cand_joints_aligned_low, shiftx=0.5)

    # upper body align
    base_joints_aligned_up, cand_joints_aligned_up = align_candidate_pose_up(base_joints_3d, cand_joints_3d)

    larm_dev = ca.get_left_arm_dev(base_joints_aligned_up, cand_joints_aligned_up)

    rarm_dev = ca.get_right_arm_dev(base_joints_aligned_up, cand_joints_aligned_up)

    lfarm_dev = ca.get_left_farm_dev(base_joints_aligned_up, cand_joints_aligned_up)

    rfarm_dev = ca.get_right_farm_dev(base_joints_aligned_up, cand_joints_aligned_up)

    if vis:
        up_idxs = list(range(11, 17))
        plot_3d(ax2, bjoints_2d, base_joints_aligned_up, colors[0], shiftx=0.0, idxs = up_idxs)

        plot_3d(ax2, cjoints_2d, cand_joints_aligned_up, colors[1], shiftx=0.0, idxs = up_idxs)

        vis_vector(ax2, base_joints_aligned_up, shiftx=0.0)
        vis_vector(ax2, cand_joints_aligned_up, shiftx=0.0)

    return [trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev]

def vis_vector(ax, joints_3d_org, shiftx=0.0):

    joints_3d = joints_3d_org.copy()
    joints_3d[:, 0] += shiftx
    # _, pts = ca.get_right_forearm_vector(joints_3d)
    # s, e = pts

    # ax.plot([-s[0], -e[0]], [s[1], e[1]], [s[2], e[2]], 'o-', color='blue')

    _, pts = ca.get_left_arm_vector(joints_3d)
    s, e = pts

    ax.plot([-s[0], -e[0]], [s[1], e[1]], [s[2], e[2]], 'o-', color='blue')

def create_path_ids(tb, tc, pad):

    bs = list(range(tb - pad, tb + pad))
    cs = list(range(tc - pad, tc + pad))

    bs = np.expand_dims(bs, axis=1)
    cs = np.expand_dims(cs, axis=1)
    align = np.ones((len(cs), 1))

    pair_lst = np.hstack((bs, cs))
    pair_lst = np.hstack((pair_lst, align)).astype('int')

    return pair_lst.tolist()

def align_pose3d_dev(video_lst, poses_2d_list, poses_3d_list, path_ids, save_out_pkl, vis=False):

    if vis:
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 4, width_ratios=[0.5, 1, 1, 0.5])

        # Initialize Matplotlib 3D plot in the first subplot
        ax = fig.add_subplot(gs[0]) 
        ax1 = fig.add_subplot(gs[1], projection='3d')
        ax2 = fig.add_subplot(gs[2], projection='3d')
        ax3 = fig.add_subplot(gs[3]) 

        ax1.view_init(elev=10, azim=-140)
        ax2.view_init(elev=10, azim=-140)

    base_video, cand_video = video_lst[0], video_lst[1]
    bpose_3ds, cpose_3ds = poses_3d_list[0], poses_3d_list[1] 
    bpose_2ds, cpose_2ds = poses_2d_list[0], poses_2d_list[1] 
    deviations_lst = []

    # tb, tc = 131, 115 #Closing_Overhead_Bin
    # # tb, tc = 178, 174 #Lift_Galley_Carrier
    # # tb, tc = 231, 271 #Stow_Full_Cart
    # path_ids = create_path_ids(tb, tc, pad=5)

    for t, (b, c, isalign) in enumerate(path_ids):

        frame1 = base_video[b]
        frame2 = cand_video[c]
        
        if len(bpose_3ds[b]):
            bjoints_3d = bpose_3ds[b][0]['keypoints_3d']
            bjoints_2d = bpose_3ds[b][0]['keypoints']
            btid = bpose_3ds[b][0]['track_id']
            bface_2d = bpose_2ds[b][0]['keypoints_2d'][[0, 1, 2, 3, 4, 17]]        
            bsjoints_2d = np.concatenate((bjoints_2d, bface_2d), axis = 0)   
        else:
            bjoints_3d = np.zeros((17, 3), dtype="float")
            bjoints_2d = np.zeros((17, 3), dtype="float")
            btid = 0
            bsjoints_2d = np.zeros((23, 3), dtype="float")
        if len(cpose_3ds[c]):
            cjoints_3d = cpose_3ds[c][0]['keypoints_3d']
            cjoints_2d = cpose_3ds[c][0]['keypoints']
            ctid = cpose_3ds[c][0]['track_id']
            cface_2d = cpose_2ds[c][0]['keypoints_2d'][[0, 1, 2, 3, 4, 17]]           
            csjoints_2d = np.concatenate((cjoints_2d, cface_2d), axis = 0)
        else:
            cjoints_3d = np.zeros((17, 3), dtype="float")
            cjoints_2d = np.zeros((17, 3), dtype="float")
            ctid = 0
            csjoints_2d = np.zeros((23, 3), dtype="float")

        if not (btid == ctid and btid == req_tid): continue

        if vis:

            ax.clear()
            ax1.clear()
            ax2.clear()
            ax3.clear()

            ax1.set_title(f"3D Joints Frame {t}")

            ax.set_title(f"2D Joints Baseline Frame {b}")
            ax3.set_title(f"2D Joints Candidate Frame {c}")

            plot_3d(ax1, bjoints_2d, bjoints_3d, colors[0], shiftx=0.0)

            plot_3d(ax1, cjoints_2d, cjoints_3d, colors[1], shiftx=0.25)

        if isalign:
            if not vis:ax2=None
            deviations = cal_dev(bjoints_3d, cjoints_3d, bjoints_2d, cjoints_2d, ax2=ax2, vis=vis)
        else:
            deviations = [0] * 10

        deviations_lst.append({'deviations':deviations, 'cjoints_2d':csjoints_2d, 'bjoints_2d':bsjoints_2d})

        if vis:
            trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev = deviations
            ax2.set_title(f"trunk: {trunk_dev} \n trunk twist: {trunk_twist_dev} \n larm_dev: {larm_dev} \n rarm_dev: {rarm_dev} \n lfarm_dev: {lfarm_dev} \n rfarm_dev: {rfarm_dev} \n lthigh_dev: {lthigh_dev} \n rthigh_dev: {rthigh_dev} \n lleg_dev: {lleg_dev} \n rleg_dev: {rleg_dev} \n")

            frame1 = cv2.putText(frame1, "Frame: " + str(b), (500, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA)
            frame1 = draw_2d_skeletons_3d(frame1, bjoints_2d, (0, 0, 255), conf_thresh = conf_thresh)
            frame1 = cv2.resize(frame1, None, fx=0.5, fy=0.5)
            ax.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

            frame2 = cv2.putText(frame2, "Frame: " + str(c), (500, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA)
            frame2 = draw_2d_skeletons_3d(frame2, cjoints_2d, (0, 255, 0), conf_thresh = conf_thresh)
            frame2 = cv2.resize(frame2, None, fx=0.5, fy=0.5)
            ax3.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

            pad = 0.25
            gminn, gmaxn = get_min_max(cjoints_3d)
            ax1.set_xlim([gminn - pad, gmaxn + pad])  # Adjust as necessary
            ax1.set_ylim([gminn - pad, gmaxn + pad])  # Adjust as necessary
            ax1.set_zlim([gminn - pad, gmaxn + pad])

            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')

            plt.tight_layout()
            plt.pause(0.00001)
            # plt.pause(5)
            # plt.show()

    with open(save_out_pkl, 'wb') as f:
        pickle.dump(deviations_lst, f)

def draw_joints_2d(frame, joints_2d_hrnet, num_pts=18):

    for i in range(num_pts):
        x, y = joints_2d_hrnet[i, :2].astype('int')
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    return frame

if __name__ == "__main__":

    connections = [
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

    file_names = ['baseline', 'candidate1']
    act_name = "Incorrect_closing_OH_bin" #"Closing_Overhead_Bin" #"Lift_Galley_Carrier" #"Stow_Full_Cart" #"Lift_Luggage" # "Serving_from_Basket"
    # 'Removing_Item_from_Bottom_of_Cart' # #'Serving_from_Basket' # 'Pushing_cart' # 'Lower_Galley_Carrier' #Stowing_carrier
    root_dir = '/home/tumeke-balaji/Documents/results/delta/input_videos/delta_incorrect_data/'
    root_pose = root_dir + act_name + '/'
    align_path = root_pose + file_names[0] + "_" + file_names[1] + "-dtw_path.json"
    output_video_path = root_pose + act_name + '_dev1.mov'
    dtw_video_path = root_pose + file_names[0] + "_" + file_names[1] + "_alignment.mp4"
    colors = ['red', 'green', 'black', 'orange', 'blue']
    req_tid = 0
    conf_thresh = 0.35

    video_lst, poses_2ds, poses_3ds = [], [], []
    video_paths = []    

    for file_name in file_names:
        pose_dir = root_pose + '/poses/joints/'
        video_path = root_pose + 'videos/' + file_name + '.mov'
        video_paths.append(video_path)
        video = mmcv.VideoReader(video_path)

        pose_path = pose_dir + file_name + '_pose_3d.p'
        pose_path_2d = pose_dir + file_name + '.pkl'
        print('pose_path ', pose_path)

        with open(pose_path_2d, 'rb') as f:
            poses_2d = pickle.load(f)
            
        with open(pose_path, 'rb') as f:
            poses_3d = pickle.load(f)
        
        poses_3ds.append(poses_3d)
        poses_2ds.append(poses_2d)

        video_lst.append(video)

    out_pkl = root_pose + '/deviations.pkl'
    print('out_pkl ', out_pkl)

    # get alignment
    # path_pairs = utils.manual_video_align(act_name)
    path_pairs = utils.pose_embed_video_align(align_path)
    # path_pairs = fine_align.get_pmpjpe_align(video_lst[0], video_lst[1], poses_3ds[0], poses_3ds[1])
    
    align_pose3d_dev(video_lst, poses_2ds, poses_3ds, path_pairs, out_pkl, vis=False)

    with open(out_pkl, 'rb') as f:
        deviations = pickle.load(f)

    render.render_results(video_paths[0], video_paths[1], dtw_video_path, output_video_path, path_pairs, deviations, conf_thresh)