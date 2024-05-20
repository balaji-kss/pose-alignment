import cv2.mat_wrapper
import mmcv
import numpy as np
import json
import os
import pickle
from scipy.spatial import procrustes
from pick_baseline import valid_indices, filter_frame_pairs, calc_pmjpe 

def get_plane_pts(joints3d):

    right = np.copy(joints3d[14]) 
    # right[0] = np.mean(joints3d[[14], 0])

    left = np.copy(joints3d[11])
    # left[0] = np.mean(joints3d[[4, 5, 6, 11], 0])

    mid = np.nanmean(joints3d[[0, 1, 4, 7]], axis = 0)
    # mid = np.copy(joints3d[7])

    points = [left, right, mid]

    return np.array(points)


def angle_bw_planes_frame(joints3d1, joints3d2):

    points1 = get_plane_pts(joints3d1)
    points2 = get_plane_pts(joints3d2)

    def normal_vector(points):
        # Assume points is a Nx3 matrix where each row is a point
        # Calculate two vectors in the plane
        vector1 = points[1] - points[0]
        vector2 = points[2] - points[0]
        
        # Calculate the normal vector as the cross product of these two vectors
        normal = np.cross(vector1, vector2)

        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        return normal
    
    # Calculate the normal vectors for each plane
    normal1 = normal_vector(points1)
    normal2 = normal_vector(points2)
    
    # Use the dot product to calculate the angle between the two normals
    dot_product = np.dot(normal1, normal2)
    
    # Clip the dot product to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Calculate the angle in radians
    angle = np.arccos(dot_product)
    
    # Optionally, convert to degrees
    angle_degrees = np.degrees(angle)

    # cross_product = np.cross(normal1, normal2)
    # direction = np.sign(np.dot(cross_product, np.array([0, 0, 1])))
    # print('direction ', direction, angle_degrees)

    return angle_degrees
    
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

    print('tot_frames_b, tot_frames_c ', tot_frames_b, tot_frames_c)

    data = np.array(data)
    
    indices = np.where((data[:, 3] == 1))[0]
    dist_vals = data[indices, 2]
    
    indices = np.where((data[:, 2] < dist_thresh) & (data[:, 3] == 1))[0]

    pairs = filter_frame_pairs(data[indices], tot_frames_b, tot_frames_c, ratio = [0.15, 0.15])
    # indices = indices[:int(0.1* len(indices))]
    # pairs = data[indices]

    print('min ', min(dist_vals))
    print('max ', max(dist_vals))
    print('mean ', np.mean(dist_vals))
    print('median ', np.median(dist_vals))

    return pairs

def compare_video(temp_vid_path, cand_vid_path):

    temp_video = mmcv.VideoReader(temp_vid_path)
    cand_video = mmcv.VideoReader(cand_vid_path)
    pairs = get_pairs(json_path, len(temp_video), len(cand_video))
    print('num pairs ', len(pairs))

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

    pmpjpe_lst = []
    
    for bid, cid, score, flag in pairs:
        bid = int(bid)
        cid = int(cid)

        frame1 = temp_video[bid]
        frame2 = cand_video[cid]
        
        try:
            joints3d1 = poses3d1[bid][0]['keypoints_3d']
            joints3d2 = poses3d2[cid][0]['keypoints_3d']

            joints2d1 = poses2d1[bid][0]['keypoints']
            joints2d2 = poses2d2[cid][0]['keypoints']
            joints2d1 = convert_keypoint_definition(joints2d1)
            joints2d2 = convert_keypoint_definition(joints2d2)
        except:continue
        print(bid, cid)
        angle = angle_bw_planes_frame(joints3d1, joints3d2)
        print('angle bw planes ', angle)

        val_ids, non_val_ids = valid_indices(joints2d1, joints2d2, conf_thresh=0.35)
        pmpjpe = calc_pmjpe(joints3d1, joints3d2, val_ids, task_name)
        pmpjpe_lst.append(pmpjpe)

        print(bid, cid, np.round(score, 3), pmpjpe, flag)
        frame1 = cv2.resize(frame1, None, fx=0.25, fy=0.25)
        frame2 = cv2.resize(frame2, None, fx=0.25, fy=0.25)
        cv2.imshow('baseline', frame1)
        cv2.imshow('candidate', frame2)
        cv2.waitKey(-1)

    avg_pmpjpe = np.mean(pmpjpe_lst)
    print('avg_pmpjpe ', avg_pmpjpe)

if __name__ == '__main__':
    dist_thresh = 100 #4.114593505859375 #3.330575168132782
    task_name =  "Remove_Full_Cart"#"Turning_Bag_onto_Side"
    template_name = "baseline17"
    candidate_name = "candidate1"
    json_path = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/" + template_name + "_" + candidate_name + "-dtw_path.json"
    pose_dir = "/home/tumeke-balaji/Documents/results/human-mesh-recovery/delta_data_videos_poses_res/" + task_name + "/videos/" 
    video_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/videos/" 
    temp_vid_path = video_dir + template_name + ".mov"
    cand_vid_path = video_dir + candidate_name + ".mov"
    print('template_name ', template_name)
    print('candidate_name ', candidate_name)
    compare_video(temp_vid_path, cand_vid_path)