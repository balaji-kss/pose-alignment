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
import smooth_align as sa
np.set_printoptions(precision=3)

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
        return [0] * 12

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

    fleg_dev = ca.get_leg_dev(base_joints_aligned_low, cand_joints_aligned_low)

    if vis:
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

    farm_dev = ca.get_farm_dev(base_joints_aligned_up, cand_joints_aligned_up)

    if False:
        up_idxs = list(range(11, 17))
        plot_3d(ax2, bjoints_2d, base_joints_aligned_up, colors[0], shiftx=0.0, idxs = up_idxs)

        plot_3d(ax2, cjoints_2d, cand_joints_aligned_up, colors[1], shiftx=0.0, idxs = up_idxs)

        vis_vector(ax2, base_joints_aligned_up, shiftx=0.0)
        vis_vector(ax2, cand_joints_aligned_up, shiftx=0.0)

    return [trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev, farm_dev, fleg_dev]

def vis_vector(ax, joints_3d_org, shiftx=0.0):

    joints_3d = joints_3d_org.copy()
    joints_3d[:, 0] += shiftx
    _, pts = ca.get_left_hip_ankle_vector(joints_3d)
    s, e = pts

    ax.plot([-s[0], -e[0]], [s[1], e[1]], [s[2], e[2]], 'o-', color='blue')

    _, pts = ca.get_right_hip_ankle_vector(joints_3d)
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

def valid_dev(dev, bjoints_2d, cjoints_2d, idxs, thresh):

    for idx in idxs:
        if bjoints_2d[idx, 2] < thresh or cjoints_2d[idx, 2] < thresh:
            return np.nan
        
    return dev

def mask_dev(deviations, bjoints_2d, cjoints_2d, thresh):

    # NOTE: check mask for trunk
    # NOTE: check mask for inter_fore_arm

    trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev, farm_dev, fleg_dev = deviations

    larm_dev = valid_dev(larm_dev, bjoints_2d, cjoints_2d, [11, 12], thresh=thresh)
    rarm_dev = valid_dev(rarm_dev, bjoints_2d, cjoints_2d, [14, 15], thresh=thresh)

    lfarm_dev = valid_dev(lfarm_dev, bjoints_2d, cjoints_2d, [11, 12, 13], thresh=thresh)
    rfarm_dev = valid_dev(rfarm_dev, bjoints_2d, cjoints_2d, [14, 15, 16], thresh=thresh)

    lthigh_dev = valid_dev(lthigh_dev, bjoints_2d, cjoints_2d, [4, 5], thresh=thresh)
    rthigh_dev = valid_dev(rthigh_dev, bjoints_2d, cjoints_2d, [1, 2], thresh=thresh)
    
    lleg_dev = valid_dev(lleg_dev, bjoints_2d, cjoints_2d, [4, 5, 6], thresh=thresh)
    rleg_dev = valid_dev(rleg_dev, bjoints_2d, cjoints_2d, [1, 2, 3], thresh=thresh)

    # # Max across left arm
    max_larm = max(larm_dev, farm_dev)
    if max_larm > lfarm_dev:
        larm_dev, lfarm_dev = max_larm, max_larm
    else:
        larm_dev, lfarm_dev = max_larm, lfarm_dev
        
    # # Max across right arm
    max_rarm = max(rarm_dev, farm_dev)
    if max_rarm > lfarm_dev:
        rarm_dev, rfarm_dev = max_rarm, max_rarm
    else:
        rarm_dev, rfarm_dev = max_rarm, rfarm_dev

    max_lleg = max(lthigh_dev, fleg_dev)
    if max_lleg > lleg_dev:
        lthigh_dev, lleg_dev = max_lleg, max_lleg
    else:
        lthigh_dev, lleg_dev = max_lleg, lleg_dev

    max_rleg = max(rthigh_dev, fleg_dev)
    if max_rleg > rleg_dev:
        rthigh_dev, rleg_dev = max_rleg, max_rleg
    else:
        rthigh_dev, rleg_dev = max_rleg, rleg_dev

    return [trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev, farm_dev, fleg_dev]

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

    max_bidx, max_cidx = 0, 0
    for t, (b, c, isalign) in enumerate(path_ids):

        # if c < 91:continue

        max_cidx = max(max_cidx, c)
        max_bidx = max(max_bidx, b)
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

        # if not (btid == ctid and btid == req_tid): continue

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

        
        if not vis:ax2=None

        deviations = cal_dev(bjoints_3d, cjoints_3d, bjoints_2d, cjoints_2d, ax2=ax2, vis=vis)

        deviations = mask_dev(deviations, bjoints_2d, cjoints_2d, thresh=conf_thresh)

        deviations_lst.append({'deviations':deviations, 'cjoints_2d':csjoints_2d, 'bjoints_2d':bsjoints_2d})

        if vis:
            trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev, farm_dev, fleg_dev = deviations
            ax2.set_title(f"trunk: {trunk_dev} \n trunk twist: {trunk_twist_dev} \n larm_dev: {larm_dev} \n rarm_dev: {rarm_dev} \n lfarm_dev: {lfarm_dev} \n rfarm_dev: {rfarm_dev} \n lthigh_dev: {lthigh_dev} \n rthigh_dev: {rthigh_dev} \n lleg_dev: {lleg_dev} \n rleg_dev: {rleg_dev} \n farm_dev: {farm_dev} \n fleg_dev: {fleg_dev} \n")

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
            # plt.pause(1)
            # plt.show()

    deviations_lst = avg_non_align(path_ids, deviations_lst)
    base_dev_dict = create_dict_bs(path_ids, deviations_lst, max_bidx + 1)
    deviations_lst, cand_dev_dict = smooth_deviations(path_ids, deviations_lst, max_cidx + 1)
                      
    with open(save_out_pkl, 'wb') as f:
        pickle.dump(deviations_lst, f)

    return base_dev_dict, cand_dev_dict

def avg_non_align_segment(deviation_info_lst):
    
    deviations_np = create_np(deviation_info_lst, name="deviations")

    # avg_dev = np.nanmean(deviations_np, axis = 0).tolist()
    avg_dev = np.nanquantile(deviations_np, 0.85, axis = 0).tolist()
    
    avg_dev = np.round(avg_dev, 2)

    for deviation_info in deviation_info_lst:
        deviation_info["deviations"] = avg_dev

    return deviation_info_lst

def avg_non_align(path_ids, deviations_list):

    print('path_ids ', len(path_ids))
    print('deviations_list ', len(deviations_list))

    assert len(path_ids) == len(deviations_list)
    path_ids_np = np.array(path_ids)

    # get align and non align segments 
    segments = sa.find_segments(path_ids_np[:, 2]) # start and end included

    # get indices of non align segments
    non_align_se = np.where(segments[:, 0] == 0)[0]
    
    # calc avg for each non align segments
    for na_idx in non_align_se:
        # ignore non-aligns at the start and end of video 
        # because the activity might not have started
        if na_idx == 0 or na_idx == len(segments) - 1:continue 
        sidx = segments[na_idx][1]
        eidx = segments[na_idx][2]
        deviations_list[sidx:eidx + 1] = avg_non_align_segment(deviations_list[sidx:eidx + 1])

    return deviations_list

def moving_average(data, window_size):

    padding = (window_size - 1) // 2

    data = np.pad(data, (padding, padding), mode='edge')

    windowed_data = sliding_window_view(data, window_shape=window_size)

    moving_averages = np.nanmean(windowed_data, axis=1)

    return moving_averages

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def filter_nans(y):
    nans, x = nan_helper(y)
    if len(x(~nans)) == 0 or len(y[~nans]) == 0:return y

    y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    return y

def window_mean_angle_dev(deviations_np, window_size):

    N, D = deviations_np.shape[:2]

    for d in range(D):
        deviations_np[:, d] = filter_nans(deviations_np[:, d])
        deviations_np[:, d] = moving_average(deviations_np[:, d], window_size)

    return np.round(deviations_np, 2)

def create_np(info_lst, name):

    val_lst = []
    for info in info_lst:
        val_lst.append(info[name])
    return np.array(val_lst)

def update_dict_lst(dev_info_lst, cand_dev_dict, sm_dev_lst, path_ids, num_cand_frames):

    for i in range(num_cand_frames):
        cand_dev_dict[i][0] = sm_dev_lst[i]

    for i in range(len(dev_info_lst)):
        _, c, _ = path_ids[i]
        dev_info_lst[i]["deviations"] = sm_dev_lst[c]

    return dev_info_lst, cand_dev_dict

def create_dict_np(path_ids, deviations_info_lst, num_cand_frames):

    cand_dev_dict = {}
    pad_dev = [0] * 10
    csjoints_2d = np.zeros((23, 3), dtype="float")

    for i in range(num_cand_frames):
        cand_dev_dict[i] = [pad_dev, csjoints_2d]

    for i in range(len(path_ids)):
        _, c, _ = path_ids[i]
        cand_dev_dict[c][0] = deviations_info_lst[i]['deviations']
        cand_dev_dict[c][1] = deviations_info_lst[i]['cjoints_2d']

    dev_lst = []
    for i in range(num_cand_frames):
        dev_lst.append(cand_dev_dict[i][0])

    return cand_dev_dict, np.array(dev_lst)

def create_dict_bs(path_ids, deviations_info_lst, num_base_frames):

    base_dev_dict = {}
    pad_dev = [0] * 10
    bsjoints_2d = np.zeros((23, 3), dtype="float")

    for i in range(num_base_frames):
        base_dev_dict[i] = [pad_dev, bsjoints_2d]

    for i in range(len(path_ids)):
        b, _, _ = path_ids[i]
        base_dev_dict[b][0] = deviations_info_lst[i]['deviations']
        base_dev_dict[b][1] = deviations_info_lst[i]['bjoints_2d']
    
    return base_dev_dict

def smooth_deviations(path_ids, deviations_info_lst, num_cand_frames, window_size = 5):
    
    cand_dev_dict, dev_np = create_dict_np(path_ids, deviations_info_lst, num_cand_frames)
    smooth_devations = window_mean_angle_dev(dev_np, window_size)
    smooth_devations = smooth_devations.tolist()
    deviations_info_lst, cand_dev_dict = update_dict_lst(deviations_info_lst, cand_dev_dict, smooth_devations, path_ids, num_cand_frames)

    return deviations_info_lst, cand_dev_dict

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

    file_names = ['baseline14', 'candidate1']
    act_name = "Lowering_Crew_Bag" #"Incorrect_Lowering_crew_Bag" #"Closing_Overhead_Bin" #"Lift_Galley_Carrier" #"Stow_Full_Cart" #"Lift_Luggage" # "Serving_from_Basket"
    # 'Removing_Item_from_Bottom_of_Cart' # #'Serving_from_Basket' # 'Pushing_cart' # 'Lower_Galley_Carrier' #Stowing_carrier
    root_dir = '/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/'
    root_pose = root_dir + act_name + '/'
    align_path = root_pose + file_names[0] + "_" + file_names[1] + "-dtw_path.json"
    save_vname = root_pose + act_name + "_" + file_names[0] + "_" + file_names[1]
    output_video_path = save_vname + '_falign.mov'
    output_cand_video_path = save_vname + '.mov'
    output_compare_video_path = save_vname + '_fc.mov'
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
    
    base_dict, deviations_dict = align_pose3d_dev(video_lst, poses_2ds, poses_3ds, path_pairs, out_pkl, vis=False)

    with open(out_pkl, 'rb') as f:
        deviations = pickle.load(f)

    render.render_results(video_paths[0], video_paths[1], dtw_video_path, output_video_path, path_pairs, deviations, conf_thresh)
    # render.render_cand_video(video_paths[1], output_cand_video_path, deviations_dict, conf_thresh)
    # render.render_compare_video(video_paths[0], video_paths[1], output_compare_video_path, base_dict, deviations_dict, conf_thresh, isblur=False)