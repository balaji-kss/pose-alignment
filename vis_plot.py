import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from scipy.linalg import svd
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import json
import copy
from sklearn.decomposition import PCA

def plot_3d_joints(ax, joints_3d, joints_2d, dcolor, conf_thresh, shift=[0.0, 0.0], name="hmr"):

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
            if joints_2d[connection[0], 2] < conf_thresh or joints_2d[connection[1], 2] < conf_thresh: continue
        xs = [-joints_3d[connection[0], 0] + shift[0], -joints_3d[connection[1], 0] + shift[0]]
        ys = [joints_3d[connection[0], 1] + shift[1], joints_3d[connection[1], 1] + shift[1]]
        zs = [joints_3d[connection[0], 2], joints_3d[connection[1], 2]]
        if i == 0: ax.plot(xs, ys, zs, 'o-', color=dcolor, label=name)
        else:ax.plot(xs, ys, zs, 'o-', color=dcolor)

def get_idx(pose_dets_frame):

    tid0_idx, tid1_idx = None, None
    for i in range(len(pose_dets_frame)):
        tid = pose_dets_frame[i]['track_id']
        if tid == 0:
            return i
        if tid == 1:
            tid1_idx = i

    return tid1_idx

def get_3d_pose(pose3d_path, num_people, name="hmr"):

    with open(pose3d_path, 'rb') as f:
        poses3d = pickle.load(f)

    num_frames = len(poses3d)

    kpts3d_data = np.zeros((num_people, num_frames, 17, 3), dtype='float')
    kpts3d_data[:, :, :, :] = np.nan

    kpts2d_data = np.zeros((num_people, num_frames, 17, 2), dtype='float')
    kpts2d_data[:, :, :, :] = np.nan

    for i in range(len(poses3d)):
        j = get_idx(poses3d[i])
        if j is None:continue
        track_id = poses3d[i][j]['track_id']
        assert track_id >= 0
        track_id = 0
        kpts3d = poses3d[i][j]['keypoints_3d']
        kpts3d_data[track_id, i] = kpts3d
        if name == "hmr": kpts2d = poses3d[i][j]['keypoints_hmr'][:, :2]
        if name == "vp3d": kpts2d = poses3d[i][j]['keypoints'][:, :2]
        kpts2d_data[track_id, i] = kpts2d

    return kpts3d_data, kpts2d_data

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

def get_2d_pose(pose2d_path, num_people):

    with open(pose2d_path, 'rb') as f:
        poses2d = pickle.load(f)

    num_frames = len(poses2d)

    kpts2d_data = np.zeros((num_people, num_frames, 17, 3), dtype='float')
    kpts2d_data[:, :, :, :] = np.nan

    for i in range(len(poses2d)):
        j = get_idx(poses2d[i])
        if j is None:continue
        track_id = poses2d[i][j]['track_id']
        assert track_id >= 0
        track_id = 0
        kpts2d = poses2d[i][j]['keypoints']
        kpts2d = convert_keypoint_definition(kpts2d)
        kpts2d_data[track_id, i] = kpts2d

    return kpts2d_data

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

def valid_indices(joints_2d1, joints_2d2):

    indices1 = np.where(joints_2d1[:, 2] >= conf_thresh)[0]
    indices2 = np.where(joints_2d2[:, 2] >= conf_thresh)[0]

    set1 = set(indices1)
    set2 = set(indices2)

    # Find intersection
    common_elements = set1.intersection(set2)
    
    mask_ids = [2, 3, 5, 6, 9, 10]
    val_ids = common_elements - set(mask_ids)
    non_val_ids = set(range(17)) - val_ids
    
    val_ids = np.array(list(val_ids))
    non_val_ids = np.array(list(non_val_ids))

    return val_ids, non_val_ids

def fit_plane_and_find_normal(points):
    """ Fit a PCA plane to a set of points and return the normal to the plane. """
    pca = PCA(n_components=2)  # We are fitting a plane, so we use 2 components

    # mask nan joints
    points = points[~np.isnan(points).any(axis=1)]
    
    # mask  hand joints
    points = points[:11]

    pca.fit(points)
    normal = np.cross(pca.components_[0], pca.components_[1])  # Cross product of the principal components
    return pca.mean_, normal / np.linalg.norm(normal)  # Normalizing the normal vector

def calculate_angle_between_planes(normal1, normal2):
    """ Calculate the angle between two planes given their normals. """
    cosine_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
    cosine_angle = np.clip(cosine_angle, -1, 1)  # Numerical stability
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)  # Return the angle in degrees

def angle_bw_planes(joints3d1, joints3d2):

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

def angle_bw_planes_0(skeleton1, skeleton2, ax=None):

    mean1, normal1 = fit_plane_and_find_normal(skeleton1)
    mean2, normal2 = fit_plane_and_find_normal(skeleton2)

    # Calculate the angle between the two planes
    angle_between_planes = calculate_angle_between_planes(normal1, normal2)
    angle_between_planes_s = angle_bw_planes_shoulder(skeleton1, skeleton2)

    diff1 = np.abs(angle_between_planes - angle_between_planes_s)
    diff2 = np.abs(180 - angle_between_planes - angle_between_planes_s)
    if diff1 > diff2:
        angle_between_planes = 180 - angle_between_planes

    if ax is not None:
        # mean2[0] += 0.5
        # mean2[1] += 0.5
        ax.quiver(*mean1, *normal1, length=1, color='r', label='Normal 1')
        ax.quiver(*mean2, *normal2, length=1, color='b', label='Normal 2')
        # ax.scatter(-skeleton1[:, 0], skeleton1[:, 1], skeleton1[:, 2], color='r', s=50)
        # ax.scatter(-skeleton2[:, 0], skeleton2[:, 1], skeleton2[:, 2], color='b', s=50)

    return angle_between_planes

def calc_pmjpe(joints_3d1, joints_3d2, val_ids):

    joints_3d1_a = np.zeros_like(joints_3d1)
    joints_3d2_a = np.zeros_like(joints_3d2)

    # joints_3d1, joints_3d2, disparity = procrustes(joints_3d1[val_ids], joints_3d2[val_ids])
    disparity = 0
    mean1 = np.mean(joints_3d1, axis = 0)
    mean2 = np.mean(joints_3d1, axis = 0)

    joints_3d1_a[:] = mean1
    joints_3d2_a[:] = mean2

    # joints_3d1_a[val_ids] = joints_3d1[:, :]
    # joints_3d2_a[val_ids] = joints_3d2[:, :]

    joints_3d1_a[:, :] = joints_3d1[:, :]
    joints_3d2_a[:, :] = joints_3d2[:, :]

    disp_sqrt = np.sqrt(disparity)
    disp_sqrt = disp_sqrt / len(val_ids)

    score = np.round(disp_sqrt , 3)

    return joints_3d1_a, joints_3d2_a, score

def vis_pose3d(pairs, pose3d_paths, pose2d_paths):

    num_people = 1
    kpts3d_data1, _ = get_3d_pose(pose3d_paths[0], num_people, name="hmr")
    kpts3d_data2, _ = get_3d_pose(pose3d_paths[1], num_people, name="hmr")

    kpts2d_data1 = get_2d_pose(pose2d_paths[0], num_people)
    kpts2d_data2 = get_2d_pose(pose2d_paths[1], num_people)

    num_frames = kpts3d_data1.shape[1]
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.view_init(elev=10, azim=0)

    avg_pmpjpe = 0
    avg_angles = 0
    angles_lst = []
    print('len pairs ', len(pairs))
    prev_angle = None

    for i, (bid, cid) in enumerate(pairs):
        
        ax.clear()
        gmins, gmaxs = None, None

        # print('bid ', bid, ' cid ', cid)

        joints_3d1  = kpts3d_data1[0, bid]   
        joints_2d1  = kpts2d_data1[0, bid]

        joints_3d2  = kpts3d_data2[0, cid]
        joints_2d2  = kpts2d_data2[0, cid]    

        all_nan1 = np.isnan(joints_3d1).all()
        all_nan2 = np.isnan(joints_3d2).all()
        
        if all_nan1 or all_nan2:continue

        angle = angle_bw_planes(joints_3d1, joints_3d2, ax)
        angles_lst.append(angle)
        print(i, bid, cid, ' angle ', np.round(angle, 3))
        avg_angles += angle
        prev_angle = angle

        # val_ids, non_val_ids = valid_indices(joints_2d1, joints_2d2)
        # joints_3d1, joints_3d2, pmpjpe = calc_pmjpe(joints_3d1, joints_3d2, val_ids)
        # avg_pmpjpe += pmpjpe

        # joints_2d1[non_val_ids, 2] = 0.0
        # joints_2d2[non_val_ids, 2] = 0.0

        plot_3d_joints(ax, joints_3d1, joints_2d=joints_2d1, dcolor='blue', conf_thresh=conf_thresh, shift=[0.0, 0.0], name="hmr")
        plot_3d_joints(ax, joints_3d2, joints_2d=joints_2d2, dcolor='red', conf_thresh=conf_thresh, shift=[0.5, 0.5], name="hmr")

        gmin = np.min(joints_3d1) 
        gmax = np.max(joints_3d1) 

        ax.set_xlim([gmin - 0.1, gmax + 0.1])  # Adjust as necessary
        ax.set_ylim([gmin - 0.1, gmax + 0.1])  # Adjust as necessary
        ax.set_zlim([gmin - 0.05, gmax + 0.05]) 

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.tight_layout()
        plt.pause(0.0000000000001)

    avg_angles = np.round(np.median(angles_lst), 3)
    print('avg_angles ', avg_angles)

    avg_pmpjpe = avg_pmpjpe / len(pairs)
    print('avg_pmpjpe ', avg_pmpjpe)

    # plt.figure(figsize=(10, 6))  # Set the size of the figure (optional)
    # plt.hist(angles_lst, bins=10, color='blue', edgecolor='black')  # You can adjust the number of bins
    # plt.title('Avg angles: ' + str(avg_angles))
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # # plt.show()
    # plot_name = json_path.rsplit('/', 1)[1].rsplit('.', 1)[0].rsplit('-', 1)[0] + '.jpg'
    # plot_path = os.path.join(json_path.rsplit('/', 1)[0], plot_name)
    # plt.savefig(plot_path, format='jpg', dpi=300)

def get_pairs(json_path, thresh = 3.5):

    with open(json_path, 'r') as file:
        data = json.load(file)

    data = np.array(data)
    
    # indices = np.where((data[:, 2] <= thresh) & (data[:, 3] == 1))[0]
    indices = np.where(data[:, 3] == 1)[0]

    return data[indices, :2].astype('int')

if __name__ == '__main__':

    task_name = "Closing_Overhead_Bin"
    json_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/" + task_name + "/"
    json_paths = glob.glob(os.path.join(json_dir, '*.json'), recursive=False)
    json_paths.sort()
    conf_thresh = 0.35
    pose_dir = "/home/tumeke-balaji/Documents/results/human-mesh-recovery/delta_data_videos_poses_res/" + task_name + "/videos/" 
    
    for json_path in json_paths:
        name1, name2 = json_path.rsplit('/', 1)[1].rsplit('-')[0].rsplit('_')
        if name2 not in ["baseline1"]: continue
        if name1 not in ["baseline2"]:continue
        pose3d_path1 = os.path.join(pose_dir, name1 + '_pose3d.pkl')
        pose3d_path2 = os.path.join(pose_dir, name2 + '_pose3d.pkl')
        
        pose3d_paths = [pose3d_path1, pose3d_path2]

        pose2d_path1 = os.path.join(pose_dir, name1 + '_pose2d.pkl')
        pose2d_path2 = os.path.join(pose_dir, name2 + '_pose2d.pkl')
        pose2d_paths = [pose2d_path1, pose2d_path2]
        print('json_path ', json_path)
        pairs = get_pairs(json_path)
        print('pairs ', pairs)
        vis_pose3d(pairs, pose3d_paths, pose2d_paths)