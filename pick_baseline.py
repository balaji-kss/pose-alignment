import numpy as np
import os
import glob
import json
import pickle
from scipy.spatial import procrustes
import mmcv
import cv2
from scipy.optimize import curve_fit
from vis_plot import angle_bw_planes, get_2d_pose, get_3d_pose

def exponential_decay(x, a, b):
    return a * np.exp(-b * x)

def fit_exponential_decay(hist_data, bins):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    popt, _ = curve_fit(exponential_decay, bin_centers, hist_data, maxfev=5000)
    return popt

def calc_hist_score(hist_data, bins):

    params = fit_exponential_decay(hist_data, bins)
    residuals = hist_data - exponential_decay(0.5 * (bins[:-1] + bins[1:]), *params)

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((hist_data - np.mean(hist_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return round(r_squared, 3)

def custom_clustering(numbers, max_diff=10):
    # Sort the list of numbers
    numbers.sort()
    
    # Initialize the first cluster
    clusters = []
    current_cluster = [numbers[0]]
    
    # Iterate through the sorted numbers
    for number in numbers[1:]:
        # Check if adding this number exceeds the max difference in the current cluster
        if number[0] - current_cluster[0][0] > max_diff:
            # If yes, start a new cluster
            clusters.append(current_cluster)
            current_cluster = [number]
        else:
            # Otherwise, add the number to the current cluster
            current_cluster.append(number)
    
    # Add the last cluster if not empty
    clusters.append(current_cluster)
    return clusters

def angle_bw_planes_frame(joints3d1, joints3d2):

    hip1 = np.mean(joints3d1[[0, 1, 4]], axis = 0)
    points1 = [joints3d1[11], joints3d1[14], hip1]

    hip2 = np.mean(joints3d2[[0, 1, 4]], axis = 0)
    points2 = [joints3d2[11], joints3d2[14], hip2]

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

    return angle_degrees

def get_pairs(json_path):

    with open(json_path, 'r') as file:
        data = json.load(file)

    data = np.array(data)

    indices = np.where((data[:, 3] == 1))[0]

    return data[indices, :2].astype('int')

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

def angle_bw_planes_video(pairs, pose3d_path1, pose3d_path2):

    num_people = 1
    poses_3d1, _ = get_3d_pose(pose3d_path1, num_people)
    poses_3d2, _ = get_3d_pose(pose3d_path2, num_people)
    angles_lst = []
    
    for bid, cid in pairs:

        joints_3d1  = poses_3d1[0, bid]   
        joints_3d2  = poses_3d2[0, cid]

        all_nan1 = np.isnan(joints_3d1).all()
        all_nan2 = np.isnan(joints_3d2).all()

        if all_nan1 or all_nan2:continue

        # angle = angle_bw_planes_frame(joints_3d1, joints_3d2)
        angle = angle_bw_planes(joints_3d1, joints_3d2)
        
        if np.isnan(angle):continue
        
        angles_lst.append(angle)

    avg_angles = np.median(angles_lst)
    
    return round(avg_angles, 3)

def cluster_baselines(json_paths, pose_dir, candidate_name):

    baseline_angles = []
    for json_path in json_paths:
        
        name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
        if name2 != candidate_name:continue

        pairs = get_pairs(json_path)
        pose3d_path1 = os.path.join(pose_dir, name1 + '_pose3d.pkl')
        pose3d_path2 = os.path.join(pose_dir, name2 + '_pose3d.pkl')

        angle_bw_planes = angle_bw_planes_video(pairs, pose3d_path1, pose3d_path2)
        baseline_angles.append([angle_bw_planes, json_path])
    
    clusters = custom_clustering(baseline_angles, max_diff=20)

    return clusters

def valid_indices(joints_2d1, joints_2d2, conf_thresh=0.35):

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
    
    if len(val_ids) == 0:return np.nan
    joints_3d1, joints_3d2, disparity = procrustes(joints_3d1[val_ids], joints_3d2[val_ids])
    mean1 = np.mean(joints_3d1, axis = 0)
    mean2 = np.mean(joints_3d2, axis = 0)

    joints_3d1_a[:] = mean1
    joints_3d2_a[:] = mean2

    joints_3d1_a[val_ids] = joints_3d1[:, :]
    joints_3d2_a[val_ids] = joints_3d2[:, :]

    disp_sqrt = np.sqrt(disparity)
    disp_sqrt = disp_sqrt / len(val_ids)

    score = np.round(disp_sqrt , 3)

    return score

def calc_pmjpe_video_pair(json_path):

    num_people = 1

    name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
    
    pose3d_path1 = os.path.join(pose_dir, name1 + '_pose3d.pkl')
    pose3d_path2 = os.path.join(pose_dir, name2 + '_pose3d.pkl')

    pose2d_path1 = os.path.join(pose_dir, name1 + '_pose2d.pkl')
    pose2d_path2 = os.path.join(pose_dir, name2 + '_pose2d.pkl')
    
    poses_3d1, _ = get_3d_pose(pose3d_path1, num_people)
    poses_3d2, _ = get_3d_pose(pose3d_path2, num_people)

    poses_2d1 = get_2d_pose(pose2d_path1, num_people)
    poses_2d2 = get_2d_pose(pose2d_path2, num_people)

    pairs = get_pairs(json_path)
    avg_pmpjpe = 0

    for bid, cid in pairs:

        joints_3d1  = poses_3d1[0, bid]   
        joints_3d2  = poses_3d2[0, cid]

        joints_2d1  = poses_2d1[0, bid]
        joints_2d2  = poses_2d2[0, cid]

        val_ids, non_val_ids = valid_indices(joints_2d1, joints_2d2)
        pmpjpe = calc_pmjpe(joints_3d1, joints_3d2, val_ids)
        if np.isnan(pmpjpe): continue
        avg_pmpjpe += pmpjpe
    
    avg_pmpjpe = avg_pmpjpe / len(pairs)

    return round(avg_pmpjpe, 3)

def calc_pmpjpe_clusters(clusters):

    num_clusters = len(clusters)
    gclusters_pmpjpe = []
    for i in range(num_clusters):
        clusters_pmpjpe = []
        for j in range(len(clusters[i])):
            angle_bw_plane, json_path = clusters[i][j]
            name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
            pmjpe = calc_pmjpe_video_pair(json_path)
            clusters_pmpjpe.append([angle_bw_plane, pmjpe, name1])
        sorted_clusters_pmpjpe = sorted(clusters_pmpjpe, key=lambda x: x[1])
        gclusters_pmpjpe.append(sorted_clusters_pmpjpe)
    
    return gclusters_pmpjpe

def create_mosaic(video_dir, clusters_pose, name='pmpjpe'):

    max_cols = max([len(a) for a in clusters_pose])
    max_rows = len(clusters_pose)
    cand_video_path = os.path.join(video_dir, candidate_name + '.mov')
    mosiac = np.zeros((max_rows * 480, max_cols * 270, 3), dtype='uint8')
    mosiac[:, :, :] = 128

    for i in range(max_rows):
        for j in range(len(clusters_pose[i])):
            angle_bw_planes,  score, vname = clusters_pose[i][j]
            print('angle_bw_planes: ', angle_bw_planes, name, score, ' vname: ', vname)
            vpath = os.path.join(video_dir, vname + '.mov')
            video = mmcv.VideoReader(vpath)
            frame = video[0]
            frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25)
            mosiac[i * 480 : (i + 1) * 480, j * 270 : (j + 1) * 270] = frame
        print('************')
    video = mmcv.VideoReader(cand_video_path)
    frame = video[0]

    max_height = max(frame.shape[0], mosiac.shape[0])
    total_width = frame.shape[1] + mosiac.shape[1]
    canvas = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    canvas[:, :, :] = 128

    # Stack the images side by side on the canvas
    canvas[:frame.shape[0], :frame.shape[1]] = frame
    canvas[:mosiac.shape[0], frame.shape[1]:] = mosiac

    mosiac_path = video_dir + candidate_name + '_' + name + '.png'
    print('mosiac_path ', mosiac_path)
    cv2.imwrite(mosiac_path, canvas)

def calc_hist_video(json_path):

    with open(json_path, 'r') as file:
        data = json.load(file)
    data = np.array(data)
    indices = np.where((data[:, 3] == 1))[0]
    filtered_values = data[indices, 2]
    num_bins = 10
    bin_edges = np.linspace(min(filtered_values), max(filtered_values), num_bins + 1)
    counts, bin_edges = np.histogram(filtered_values, bins=bin_edges)
    counts = counts[:5]
    bin_edges = bin_edges[:6]
    print('len counts ', len(counts))
    print('len bin_edges ', len(bin_edges))
    score = calc_hist_score(counts, bin_edges)

    return score

def calc_hist_clusters(clusters):

    num_clusters = len(clusters)
    gclusters_hist = []
    for i in range(num_clusters):
        clusters_hist = []
        for j in range(len(clusters[i])):
            angle_bw_plane, json_path = clusters[i][j]
            score = calc_hist_video(json_path)
            name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
            clusters_hist.append([angle_bw_plane, score, name1])
        sorted_clusters_hist = sorted(clusters_hist, key=lambda x: x[1], reverse=True)
        gclusters_hist.append(sorted_clusters_hist)
    
    return gclusters_hist

if __name__ == '__main__':

    task_name = "Lifting_Crew_Bag"
    json_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/"
    json_paths = glob.glob(os.path.join(json_dir, '*.json'), recursive=False)
    pose_dir = "/home/tumeke-balaji/Documents/results/human-mesh-recovery/delta_data_videos_poses_res/" + task_name + "/videos/" 
    video_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/videos/" 
    candidate_name = "baseline18"
    
    clusters = cluster_baselines(json_paths, pose_dir, candidate_name)  
    print('clusters ', clusters)
    
    clusters_pose = calc_pmpjpe_clusters(clusters)
    create_mosaic(video_dir, clusters_pose, name='pmpjpe')
    # clusters_hist = calc_hist_clusters(clusters)
    # create_mosaic(video_dir, clusters_hist, name='hist')