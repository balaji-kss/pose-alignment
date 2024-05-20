import numpy as np
import os
import glob
import json
import pickle
from scipy.spatial import procrustes
import mmcv
import cv2
from scipy.optimize import curve_fit
from vis_plot import get_2d_pose, get_3d_pose

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

def custom_clustering_side(data, max_diff):
    # Sort the data to make comparisons between consecutive elements
    sorted_data = data
    sorted_data.sort()
    
    # Initialize the first cluster
    clusters = []
    current_cluster = [sorted_data[0]]
    
    # Iterate through the sorted data
    for i in range(1, len(sorted_data)):
        # Check if the next element should be in the same cluster
        if abs(sorted_data[i][0] - sorted_data[i - 1][0]) <= max_diff:
            current_cluster.append(sorted_data[i])
        else:
            # If not, add the current cluster to the list of clusters and start a new one
            clusters.append(current_cluster)
            current_cluster = [sorted_data[i]]
    
    # Add the last cluster to the list if not empty
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters

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

def filter_frame_pairs(frame_pairs, total_frames_A, total_frames_B, ratio = [0.0, 0.0]):
    # Convert the list of tuples to a NumPy array
    frame_pairs = np.array(frame_pairs)

    # Calculate the 10% boundary frames for both videos
    start_A = int(ratio[0] * total_frames_A)
    end_A = int((1 - ratio[1]) * total_frames_A)
    start_B = int(ratio[0] * total_frames_B)
    end_B = int((1 - ratio[1]) * total_frames_B)
    
    # Filter the frame pairs using NumPy
    mask = (
        (frame_pairs[:, 0] >= start_A) & (frame_pairs[:, 0] <= end_A) &
        (frame_pairs[:, 1] >= start_B) & (frame_pairs[:, 1] <= end_B)
    )
    filtered_pairs = frame_pairs[mask]
    
    return filtered_pairs

def get_pairs_cluster(json_path):

    name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')

    vpath = os.path.join(video_dir, name1 + '.mov')
    base_video = mmcv.VideoReader(vpath)
    
    vpath = os.path.join(video_dir, name2 + '.mov')
    cand_video = mmcv.VideoReader(vpath)

    with open(json_path, 'r') as file:
        data = json.load(file)

    data = np.array(data)

    indices = np.where((data[:, 3] == 1))[0]

    if task_name in ["Serving_from_Basket"]:
        pairs = filter_frame_pairs(data[indices], len(base_video), len(cand_video), ratio = [0.1, 0.0])
    else:
        pairs = filter_frame_pairs(data[indices], len(base_video), len(cand_video), ratio = [0.15, 0.15])

    return pairs[:, :3]

def get_pairs_pmpjpe(json_path, gthresh):

    name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
    vpath = os.path.join(video_dir, name1 + '.mov')
    base_video = mmcv.VideoReader(vpath)
    
    vpath = os.path.join(video_dir, name2 + '.mov')
    cand_video = mmcv.VideoReader(vpath)

    with open(json_path, 'r') as file:
        data = json.load(file)

    data = np.array(data)

    indices = np.where((data[:, 3] == 1))[0]

    indices = np.where((data[:, 2] < gthresh) & (data[:, 3] == 1))[0]

    pairs = filter_frame_pairs(data[indices], len(base_video), len(cand_video), ratio = [0.15, 0.15])

    return pairs[:, :2].astype('int')

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
    
    if len(pairs) == 0:return 180

    for bid, cid, _ in pairs:
        bid, cid = int(bid), int(cid)
        joints_3d1  = poses_3d1[0, bid]   
        joints_3d2  = poses_3d2[0, cid]

        all_nan1 = np.isnan(joints_3d1).all()
        all_nan2 = np.isnan(joints_3d2).all()

        if all_nan1 or all_nan2:continue

        angle = angle_bw_planes_frame(joints_3d1, joints_3d2)
        
        if np.isnan(angle):continue
        
        angles_lst.append(angle)
    
    if len(angles_lst) == 0:return 180

    avg_angles = np.median(angles_lst)

    return round(avg_angles, 3)

def cluster_baselines(json_paths, pose_dir, candidate_name):

    baseline_angles = []
    for json_path in json_paths:
        
        name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
        if name2 != candidate_name:continue
        pairs = get_pairs_cluster(json_path)
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

def calc_pmjpe(joints_3d1, joints_3d2, val_ids, task_name):
    
    if len(val_ids) <= 1:return np.nan
    if np.all(np.isnan(joints_3d1) | (joints_3d1 == -1)):return np.nan
    if np.all(np.isnan(joints_3d2) | (joints_3d2 == -1)):return np.nan

    if task_name in ["Closing_Overhead_Bin", "Lift_Galley_Carrier", "Lifting_Crew_Bag", "Lower_Galley_Carrier", "Lowering_Crew_Bag", "Remove_Full_Cart", "Stow_Full_Cart"]:
        joints_3d1, joints_3d2, _ = procrustes(joints_3d1, joints_3d2)
        
    squared_diff = np.square(joints_3d1[val_ids] - joints_3d2[val_ids])

    # Sum the squared differences along each joint dimension
    sum_squared_diff = np.sum(squared_diff, axis=1)

    # Compute the L2 norm by taking the square root of the sum of squared differences
    disp_sqrt = np.sqrt(np.sum(sum_squared_diff))

    disp_sqrt = disp_sqrt / len(val_ids)

    return disp_sqrt

def calc_pmjpe_video_pair(json_path, gthresh):

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

    pairs = get_pairs_pmpjpe(json_path, gthresh)
    pmpjpe_lst = []
    # print('pairs ', pairs)

    for bid, cid in pairs:

        joints_3d1  = poses_3d1[0, bid]   
        joints_3d2  = poses_3d2[0, cid]

        joints_2d1  = poses_2d1[0, bid]
        joints_2d2  = poses_2d2[0, cid]

        val_ids, non_val_ids = valid_indices(joints_2d1, joints_2d2)
        pmpjpe = calc_pmjpe(joints_3d1, joints_3d2, val_ids, task_name)
        
        if np.isnan(pmpjpe): continue
        pmpjpe_lst.append(pmpjpe)
    
    if len(pmpjpe_lst):
        avg_pmpjpe = np.mean(pmpjpe_lst)
    else:
        avg_pmpjpe = 100

    ratio = np.round(len(pmpjpe_lst) / poses_3d2.shape[1], 3)

    if ratio <= ratio_thresh: avg_pmpjpe = 100

    return avg_pmpjpe, ratio

def calc_pmpjpe_clusters(clusters, gthresh):

    num_clusters = len(clusters)
    gclusters_pmpjpe = []
    for i in range(num_clusters):
        clusters_pmpjpe = []
        for j in range(len(clusters[i])):
            angle_bw_plane, json_path = clusters[i][j]
            name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
            # if name1 not in ["baseline7"]:continue
            pmjpe, num_pairs = calc_pmjpe_video_pair(json_path, gthresh)
            clusters_pmpjpe.append([angle_bw_plane, pmjpe, num_pairs, name1])
        
        if task_name in ["Pushing_Cart"]:
            sorted_clusters_pmpjpe = sorted(clusters_pmpjpe, key=lambda x: x[2], reverse=True)
        else:
            sorted_clusters_pmpjpe = sorted(clusters_pmpjpe, key=lambda x: x[1])
        gclusters_pmpjpe.append(sorted_clusters_pmpjpe)

    # Sort outer list based on the average magnitude of the first element
    gclusters_pmpjpe = sorted(gclusters_pmpjpe, key=lambda inner_list: sum(abs(x[0]) for x in inner_list) / len(inner_list))

    return gclusters_pmpjpe

def create_mosaic(video_dir, clusters_pose, name='pmpjpe'):

    max_cols = max([len(a) for a in clusters_pose])
    max_rows = len(clusters_pose)
    cand_video_path = os.path.join(video_dir, candidate_name + '.mov')
    mosiac = np.zeros((max_rows * 480, max_cols * 270, 3), dtype='uint8')
    mosiac[:, :, :] = 128

    for i in range(max_rows):
        for j in range(len(clusters_pose[i])):
            angle_bw_planes,  score, num_pairs, vname = clusters_pose[i][j]
            print('angle_bw_planes: ', angle_bw_planes, name, np.round(score, 6), num_pairs, ' vname: ', vname)
            vpath = os.path.join(video_dir, vname + '.mov')
            video = mmcv.VideoReader(vpath)
            frame = video[0]
            pos = (100, 200)
            frame = cv2.putText(frame, vname[8:], pos, cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 255), 5, cv2.LINE_AA)
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

    mosiac_path = mosiac_dir + candidate_name + '_' + name + '.png'
    print('mosiac_path ', mosiac_path)
    cv2.imwrite(mosiac_path, canvas)

def create_mosiac_video(video_path_lst, top_n = 5):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    mosiac_path = mosiac_dir + candidate_name + '_' + 'pmpjpe_5.mp4'
    video = mmcv.VideoReader(video_path_lst[0])
    
    mosaic_width = int(480 + 270 * (len(video_path_lst) - 1))
    mosaic_height = int(480)
    out = cv2.VideoWriter(mosiac_path, fourcc, int(video.fps), (mosaic_width, mosaic_height))
    print('mosaic_width ', mosaic_width)
    print('mosaic_height ', mosaic_height)

    video_lens = []
    video_indices = []
    video_reader_lst = []
    for video_path in video_path_lst:
        video = mmcv.VideoReader(video_path)
        video_lens.append(len(video))
        video_reader_lst.append(video)
    min_len = min(video_lens)

    for i in range(len(video_path_lst)):
        video_len = video_lens[i]
        indices = np.linspace(0, video_len - 1, min_len)
        indices = np.round(indices).astype(int)
        video_indices.append(indices)
    
    for i in range(min_len):
        frames = []
        for j in range(len(video_reader_lst)):
            vreader = video_reader_lst[j]
            index = video_indices[j][i]
            frame = vreader[index] 
            frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25)
            if j == 1:
                frame = cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 10)
            if j == 0:
                frame_pad = np.zeros((480, 480, 3), dtype=np.uint8)
                frame_pad[:, :, :] = 128
                frame_pad[:frame.shape[0], :frame.shape[1]] = frame
            else:
                frame_pad = frame
            frames.append(frame_pad)
        mosaic_frame = np.hstack(frames)
        print('mosaic_frame shape ', mosaic_frame.shape)
        out.write(mosaic_frame)

    out.release()

def create_topn(video_dir, clusters_pose, name='pmpjpe', num=5):

    cand_video_path = os.path.join(video_dir, candidate_name + '.mov')
    video_list = [cand_video_path]

    t = 0
    for i in range(len(clusters_pose)):
        for j in range(len(clusters_pose[i])):
            angle_bw_planes,  score, num_pairs, vname = clusters_pose[i][j]
            print('angle_bw_planes: ', angle_bw_planes, name, np.round(score, 6), num_pairs, ' vname: ', vname)
            vpath = os.path.join(video_dir, vname + '.mov')
            video_list.append(vpath)
            t += 1
            if t >= num - 1:break
        if t >= num - 1:break

    print('len video list ', len(video_list))
    create_mosiac_video(video_list)

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

def list_subdirectories(path):
    subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return subdirs

def get_dist_thresh(json_paths):

    gdist_values = []
    for json_path in json_paths:
        
        name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
        if name2 != candidate_name:continue
        # if name1 not in ["baseline12", "baseline20"]:continue
        pairs = get_pairs_cluster(json_path)
        gdist_values += pairs[:, 2].tolist()
    
    return np.quantile(gdist_values, quant_thresh)

task_thresh_dict = {'Closing_Overhead_Bin': [1.0, 0.1], 'Lift_Galley_Carrier': [0.25, 0.15], 'Lifting_Crew_Bag': [0.25, 0.15], 'Lower_Galley_Carrier': [0.5, 0.15], 'Lowering_Crew_Bag': [0.5, 0.15], 'Pushing_Cart': [0.25, 0.1], 'Remove_Full_Cart': [0.25, 0.1], 'Remove_Half_Cart': [0.15, 0.1], 'Removing_Item_from_Bottom_of_Cart': [0.15, 0.1], 'Serving_from_Basket': [0.25, 0.1], 'Shifting_Bag_in_OHB': [0.25, 0.1], 'Stow_Full_Cart': [0.1, 0.15], 'Stow_Half_Cart': [0.15, 0.15], 'Turning_Bag_onto_Side': [0.5, 0.0]}

if __name__ == '__main__':

    root_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/pick_baseline/"
    task_names = list_subdirectories(root_dir)
    task_names.sort()
    task_names.reverse()

    for task_name in task_names[7:8]:
        # task_name = "Turning_Bag_onto_Side"

        quant_thresh, ratio_thresh = task_thresh_dict[task_name]
        print('quant_thresh ', quant_thresh)
        print('ratio_thresh ', ratio_thresh)

        json_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/"
        json_paths = glob.glob(os.path.join(json_dir, '*.json'), recursive=False)
        pose_dir = "/home/tumeke-balaji/Documents/results/human-mesh-recovery/delta_data_videos_poses_res/" + task_name + "/videos/" 
        video_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/videos/" 
        mosiac_dir = root_dir + task_name + "/"
        # candidate_names = ["baseline14", "baseline18"]
        # candidate_names = ["candidate2"]
        # candidate_names = ["baseline1", "baseline18"]
        candidate_names = ["candidate1", "candidate2", "candidate3"]

        for candidate_name in candidate_names:
            print('candidate_name ', candidate_name)
            gthresh = get_dist_thresh(json_paths)
            print('gthresh ', gthresh)
            clusters = cluster_baselines(json_paths, pose_dir, candidate_name)  
            
            clusters_pose = calc_pmpjpe_clusters(clusters, gthresh)
            create_mosaic(video_dir, clusters_pose, name='pmpjpe')
            # clusters_hist = calc_hist_clusters(clusters)
            # create_mosaic(video_dir, clusters_hist, name='hist')
            create_topn(video_dir, clusters_pose)
