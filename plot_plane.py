import cv2.mat_wrapper
import mmcv
import numpy as np
import json
import os
import pickle
from scipy.spatial import procrustes

def filter_frame_pairs(frame_pairs, total_frames_A, total_frames_B, ratio):
    # Convert the list of tuples to a NumPy array
    frame_pairs = np.array(frame_pairs)

    # Calculate the 10% boundary frames for both videos
    start_A = int(ratio * total_frames_A)
    end_A = int((1 - ratio) * total_frames_A)
    start_B = int(ratio * total_frames_B)
    end_B = int((1 - ratio) * total_frames_B)
    
    # Filter the frame pairs using NumPy
    mask = (
        (frame_pairs[:, 0] >= start_A) & (frame_pairs[:, 0] <= end_A) &
        (frame_pairs[:, 1] >= start_B) & (frame_pairs[:, 1] <= end_B)
    )
    filtered_pairs = frame_pairs[mask]
    
    return filtered_pairs

def valid_indices(joints_2d1, joints_2d2, conf_thresh):

    indices1 = np.where(joints_2d1[:, 2] >= conf_thresh)[0]
    indices2 = np.where(joints_2d2[:, 2] >= conf_thresh)[0]

    set1 = set(indices1)
    set2 = set(indices2)

    # Find intersection
    val_ids = set1.intersection(set2)
    mask_ids = [2, 3, 5, 6, 9, 10]
    
    val_ids = val_ids - set(mask_ids)
    non_val_ids = set(range(17)) - val_ids
    
    val_ids = np.array(list(val_ids))
    non_val_ids = np.array(list(non_val_ids))

    return val_ids, non_val_ids

def calc_pmjpe(joints_3d1, joints_3d2, val_ids):

    joints_3d1_a = np.zeros_like(joints_3d1)
    joints_3d2_a = np.zeros_like(joints_3d2)

    joints_3d1, joints_3d2, disparity = procrustes(joints_3d1[val_ids], joints_3d2[val_ids])
    
    mean1 = np.mean(joints_3d1, axis = 0)
    mean2 = np.mean(joints_3d1, axis = 0)

    joints_3d1_a[:] = mean1
    joints_3d2_a[:] = mean2

    joints_3d1_a[val_ids] = joints_3d1[:, :]
    joints_3d2_a[val_ids] = joints_3d2[:, :]

    # joints_3d1_a[:, :] = joints_3d1[:, :]
    # joints_3d2_a[:, :] = joints_3d2[:, :]

    disp_sqrt = np.sqrt(disparity)
    disp_sqrt = disp_sqrt / len(val_ids)

    score = np.round(disp_sqrt , 3)

    return score

def convert_keypoint_definition(keypoints):
    
    keypoints_new = np.zeros((17, keypoints.shape[1]))
    # pelvis is in the middle of l_hip and r_hip
    keypoints_new[0] = np.mean(keypoints[[11, 12]], axis = 0)

    # thorax is in the middle of l_shoulder and r_shoulder
    keypoints_new[8] = np.mean(keypoints[[5, 6]], axis = 0)

    # spine is in the middle of thorax and pelvis
    keypoints_new[7] = np.mean(keypoints[[0, 8]], axis = 0)

    # calculate new neck-base - y of chin, x - avg of eyes and chin
    keypoints_new[9, 1:] = np.mean(keypoints[[17]], axis = 0)[1:]    
    keypoints_new[9, 0] = np.mean(keypoints[[1, 2, 17]], axis = 0)[0]

    # calculate head - mid of leye and reye
    keypoints_new[10] = np.mean(keypoints[[1, 2]], axis = 0)

    # rearrange other keypoints
    keypoints_new[[1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]] = \
        keypoints[[12, 14, 16, 11, 13, 15, 5, 7, 9, 6, 8, 10]]

    return keypoints_new

def get_pairs(json_path, tot_frames_b, tot_frames_c):

    with open(json_path, 'r') as file:
        data = json.load(file)

    data = np.array(data)
    
    indices = np.where((data[:, 3] == 1))[0]
    dist_vals = data[indices, 2]

    indices = np.where((data[:, 2] < np.quantile(dist_vals, 0.6)) & (data[:, 3] == 1))[0]

    pairs = filter_frame_pairs(data[indices], tot_frames_b, tot_frames_c, ratio = 0.15)

    # print('min ', min(dist_vals))
    # print('max ', max(dist_vals))
    # dist_vals.sort()
    # print('sorted ', dist_vals)
    # print('mean ', np.mean(dist_vals))
    # print('median ', np.median(dist_vals))

    return pairs

def compare_video(temp_vid_path, cand_vid_path):

    temp_video = mmcv.VideoReader(temp_vid_path)
    cand_video = mmcv.VideoReader(cand_vid_path)
    pairs = get_pairs(json_path, len(temp_video), len(cand_video))
    
    name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
    
    pose3d_path1 = os.path.join(pose_dir, name1 + '_pose3d.pkl')
    pose3d_path2 = os.path.join(pose_dir, name2 + '_pose3d.pkl')

    pose2d_path1 = os.path.join(pose_dir, name1 + '_pose2d.pkl')
    pose2d_path2 = os.path.join(pose_dir, name2 + '_pose2d.pkl')
    
    with open(pose3d_path1, 'rb') as f:
        poses3d1 = pickle.load(f)

    with open(pose3d_path2, 'rb') as f:
        poses3d2 = pickle.load(f)
    
    with open(pose2d_path1, 'rb') as f:
        poses2d1 = pickle.load(f)
    
    with open(pose2d_path2, 'rb') as f:
        poses2d2 = pickle.load(f)

    avg_pmpjpe = 0.0
    for bid, cid, score, flag in pairs:
        bid = int(bid)
        cid = int(cid)

        frame1 = temp_video[bid]
        frame2 = cand_video[cid]

        try:
            joints3d1 = poses3d1[bid][0]['keypoints_hmr'][:, :2]
            joints3d2 = poses3d2[cid][0]['keypoints_hmr'][:, :2]

            joints2d1 = poses3d1[bid][0]['keypoints']
            joints2d2 = poses3d2[cid][0]['keypoints']
            joints2d1 = convert_keypoint_definition(joints2d1)
            joints2d2 = convert_keypoint_definition(joints2d2)
        except:
            continue

        val_ids, non_val_ids = valid_indices(joints2d1, joints2d2, conf_thresh=0.35)
        print('val_ids ', val_ids)
        pmpjpe = calc_pmjpe(joints3d1, joints3d2, val_ids)
        avg_pmpjpe += pmpjpe

        print(bid, cid, np.round(score, 3), pmpjpe, flag)
        frame1 = cv2.resize(frame1, None, fx=0.25, fy=0.25)
        frame2 = cv2.resize(frame2, None, fx=0.25, fy=0.25)
        cv2.imshow('baseline', frame1)
        cv2.imshow('candidate', frame2)
        cv2.waitKey(-1)

    avg_pmpjpe = avg_pmpjpe / len(pairs)
    print('avg_pmpjpe ', avg_pmpjpe)

if __name__ == '__main__':

    task_name = "Turning_Bag_onto_Side"
    template_name = "baseline13"
    candidate_name = "candidate1"
    json_path = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/" + template_name + "_" + candidate_name + "-dtw_path.json"
    pose_dir = "/home/tumeke-balaji/Documents/results/human-mesh-recovery/delta_data_videos_poses_res/" + task_name + "/videos/" 
    video_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/videos/" 
    temp_vid_path = video_dir + template_name + ".mov"
    cand_vid_path = video_dir + candidate_name + ".mov"
    compare_video(temp_vid_path, cand_vid_path)