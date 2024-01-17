import mmcv
import pickle
import cv2
import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def plot_3d(ax, joints_3d_org, color, shiftx = 0.0, idxs = None):

    joints_3d = joints_3d_org.copy()
    joints_3d[:, 0] += shiftx

    for connection in connections:
        if idxs and connection[0] not in idxs and connection[1] not in idxs:continue
        xs = [-joints_3d[connection[0], 0], -joints_3d[connection[1], 0]]
        ys = [joints_3d[connection[0], 1], joints_3d[connection[1], 1]]
        zs = [joints_3d[connection[0], 2], joints_3d[connection[1], 2]]
        ax.plot(xs, ys, zs, 'o-', color=color)

def get_min_max(joint_3ds):

    gmin, gmax = 1000, -1000 
    for joints_3d in joint_3ds:
        min_ = np.min(joints_3d)
        max_ = np.max(joints_3d)
        gmin = min(gmin, min_)
        gmax = max(gmax, max_)

    return gmin, gmax

def match_range(bvideo, cvideo, bidx, crange, bpose, cpose, vis=False):

    if len(bpose[bidx]) == 0:
        return [5] * crange[1]
    
    base_joints_3d_org = bpose[bidx][0]['keypoints_3d']
    vals = []

    if vis:
        bframe = bvideo[bidx]
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, width_ratios=[1, 1])

        ax = fig.add_subplot(gs[0]) 
        ax1 = fig.add_subplot(gs[1], projection='3d')

    for cidx in range(crange[0], crange[1]):
        
        if len(cpose[cidx]) == 0:
            val = 5
            continue

        cand_joints_3d_org = cpose[cidx][0]['keypoints_3d']
        base_joints_3d = base_joints_3d_org.copy()
        cand_joints_3d = cand_joints_3d_org.copy()
        R, _ = orthogonal_procrustes(cand_joints_3d, base_joints_3d)
        cand_joints_3d = cand_joints_3d @ R

        val = np.sum((base_joints_3d - cand_joints_3d) ** 2)

        if vis:
            ax.clear()
            ax1.clear()
            plot_3d(ax1, base_joints_3d, 'red', shiftx = 0.0, idxs = None)
            plot_3d(ax1, cand_joints_3d, 'green', shiftx = 0.0, idxs = None)

            pad = 0.5
            gminn, gmaxn = get_min_max(cand_joints_3d)
            ax1.set_xlim([gminn - pad, gmaxn + pad])  # Adjust as necessary
            ax1.set_ylim([gminn - pad, gmaxn + pad])  # Adjust as necessary
            ax1.set_zlim([gminn - pad, gmaxn + pad])
            
            cframe = cvideo[cidx]
        
            concat = np.hstack((bframe, cframe))
            concat = cv2.resize(concat, None, fx = 0.5, fy = 0.5)
            # print(cidx, round(disparity, 3), round(val, 3))
            print(cidx, round(val, 3))
            # cv2.imshow('concat ', concat)
            # cv2.waitKey(-1)

            ax.imshow(cv2.cvtColor(concat, cv2.COLOR_BGR2RGB))
            plt.tight_layout()
            plt.pause(0.0001)
        
        vals.append(val)

    return vals

def dtw_with_precomputed_distances(D):
    N, M = D.shape
    cost = np.zeros((N, M))
    cost[0, 0] = D[0, 0]
    
    for i in range(1, N):
        cost[i, 0] = cost[i-1, 0] + D[i, 0]
    for j in range(1, M):
        cost[0, j] = cost[0, j-1] + D[0, j]
    for i in range(1, N):
        for j in range(1, M):
            cost[i, j] = np.min([cost[i-1, j], cost[i, j-1], cost[i-1, j-1]]) + D[i, j]

    # Extracting the warping path
    i, j = N-1, M-1
    path = [(i, j, np.round(D[i, j], 2))]
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            argmin_cost = np.argmin([cost[i-1, j], cost[i, j-1], cost[i-1, j-1]])
            if argmin_cost == 0:
                i -= 1
            elif argmin_cost == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j, np.round(D[i, j], 2)))
    path.reverse()

    return path

def dist_matrix(bvideo, cvideo, bposes_3d, cposes_3d):

    base_len = len(bvideo)
    mat = []
    
    for i in range(base_len):
        vals = match_range(bvideo, cvideo, i, [0, len(cvideo)], bposes_3d, cposes_3d)
        mat.append(vals)

    return mat

def apply_average_filter(matrix, kernel_size):
    # Create an averaging filter kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    # Apply the filter using convolution
    # 'same' mode returns the convolved signal of the same size as input
    filtered_matrix = convolve2d(matrix, kernel, mode='same', boundary='wrap')
    
    return filtered_matrix

def get_matching(bvideo, cvideo, out_video_path, bposes_3d, cposes_3d):

    dist_mat = dist_matrix(bvideo, cvideo, bposes_3d, cposes_3d)
    dist_mat = np.array(dist_mat)
    print(len(bvideo), len(cvideo))
    print('dist_mat shape ', dist_mat.shape)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path,
                                fourcc, cvideo.fps, (1080, 960))
    
    # dist_mat_avg = apply_average_filter(dist_mat, kernel_size = 9)
    dist_mat_avg = dist_mat
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # cax1 = axs[0].imshow(dist_mat, cmap='viridis', aspect='auto')
    # fig.colorbar(cax1, ax=axs[0])
    # axs[0].set_title('Matrix 1')

    # # Display second matrix
    # cax2 = axs[1].imshow(dist_mat_avg, cmap='viridis', aspect='auto')
    # fig.colorbar(cax2, ax=axs[1])
    # axs[1].set_title('Matrix 2')

    # plt.show()
    
    # plt.matshow(dist_mat)
    # plt.colorbar()
    # plt.pause(2)

    path = dtw_with_precomputed_distances(dist_mat_avg)
    print('path ', path)

    for b, c, dist in path:
        bframe = bvideo[b]
        cframe = cvideo[c]

        print(b, c, dist)
        cv2.putText(cframe, 'PMPJPE: ' + str(dist), (250, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        concat = np.hstack((bframe, cframe))
        concat = cv2.resize(concat, None, fx=0.5, fy=0.5)
        video_writer.write(concat)
        print('concat shape ', concat.shape)
        cv2.imshow('concat ', concat)
        cv2.waitKey(-1)

    video_writer.release()

if __name__ == "__main__":

    act_name =  'Lower_Galley_Carrier' # 'Serving_from_Basket' # 'Pushing_cart' # 'Lower_Galley_Carrier' # Stowing_carrier # 'Removing_Item_from_Bottom_of_Cart'
    root_pose = '/home/tumeke-balaji/Documents/results/delta/joints/' + act_name + '/'    

    bvideo_path = root_pose + '/baseline/baseline_n.mov'
    cvideo_path = root_pose + '/candidate/candidate_n.mov'
    out_video_path = root_pose + act_name + '_align.mov'

    bpose_path_3d = "/home/tumeke-balaji/Documents/results/delta/joints/" + act_name + "/baseline/pose_3d.p"
    cpose_path_3d = "/home/tumeke-balaji/Documents/results/delta/joints/" + act_name + "/candidate/pose_3d.p"

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

    bvideo = mmcv.VideoReader(bvideo_path)
    cvideo = mmcv.VideoReader(cvideo_path)

    with open(bpose_path_3d, 'rb') as f:
        bposes_3d = pickle.load(f)
    
    with open(cpose_path_3d, 'rb') as f:
        cposes_3d = pickle.load(f)

    get_matching(bvideo, cvideo, out_video_path, bposes_3d, cposes_3d)