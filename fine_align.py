import mmcv
import pickle
import cv2
import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

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

    base_joints_3d_org = bpose[bidx][0]['keypoints_3d']
    bframe = bvideo[bidx]
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    ax = fig.add_subplot(gs[0]) 
    ax1 = fig.add_subplot(gs[1], projection='3d')

    for cidx in range(crange[0], crange[1]):

        ax.clear()
        ax1.clear()
        
        cand_joints_3d_org = cpose[cidx][0]['keypoints_3d']
        base_joints_3d = base_joints_3d_org.copy()
        cand_joints_3d = cand_joints_3d_org.copy()
        # base_joints_3d, cand_joints_3d, disparity = procrustes(base_joints_3d, cand_joints_3d)
        R, _ = orthogonal_procrustes(cand_joints_3d, base_joints_3d)
        cand_joints_3d = cand_joints_3d @ R

        plot_3d(ax1, base_joints_3d, 'red', shiftx = 0.0, idxs = None)
        plot_3d(ax1, cand_joints_3d, 'green', shiftx = 0.0, idxs = None)
        # plot_3d(ax1, cand_joints_3d_org, 'green', shiftx = 0.25, idxs = None)

        pad = 0.5
        gminn, gmaxn = get_min_max(cand_joints_3d)
        ax1.set_xlim([gminn - pad, gmaxn + pad])  # Adjust as necessary
        ax1.set_ylim([gminn - pad, gmaxn + pad])  # Adjust as necessary
        ax1.set_zlim([gminn - pad, gmaxn + pad])
        
        # base_joints_3d[[0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = 0.0
        # cand_joints_3d[[0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]] = 0.0
        val = np.sum((base_joints_3d - cand_joints_3d) ** 2)

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
        
if __name__ == "__main__":

    act_name = 'Stowing_carrier' # 'Serving_from_Basket' # 'Pushing_cart' # 'Lower_Galley_Carrier'
    root_pose = '/home/tumeke-balaji/Documents/results/delta/joints/' + act_name + '/'    

    bvideo_path = root_pose + '/baseline/baseline_n.mov'
    cvideo_path = root_pose + '/candidate1/candidate1_n.mov'

    bpose_path_3d = "/home/tumeke-balaji/Documents/results/delta/joints/" + act_name + "/baseline/pose_3d.p"
    cpose_path_3d = "/home/tumeke-balaji/Documents/results/delta/joints/" + act_name + "/candidate1/pose_3ds.p"

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

    match_range(bvideo, cvideo, 100, [0, len(cvideo)], bposes_3d, cposes_3d)