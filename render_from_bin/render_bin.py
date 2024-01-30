import numpy as np
import os
import sys
import cv2
import utils_bin as utils
import coloring
import math
import mmcv

jointMapping = {
    "Neck": 8,
    "Thorax": 7,
    "Head": 9,
    "Right shoulder": 12,
    "Left shoulder": 13,
    "Right elbow": 11,
    "Left elbow": 14,
    "Left wrist": 15,
    "Right hip": 2,
    "Left hip": 3,
    "Right wrist": 10,
    "Left wrist": 15,
    "Right knee": 1,
    "Left knee": 4,
    "Left ankle": 5,
    "Right ankle": 0,
    "Right heel": 23,
    "Right toe": 21,
    "Left toe": 22,
    "Left heel": 24,
    "Left eye": 19,
    "Right eye": 17,
    "Left ear": 20,
    "Right ear": 18,
    "Neck base": 16
}

skeletonMapping = [["Left hip", "Left shoulder"], ["Right hip", "Left hip"], ["Right hip", "Right shoulder"],
                   ["Right shoulder", "Right elbow"], ["Left shoulder", "Left elbow"], ["Right elbow", "Right wrist"],
                   ["Left elbow", "Left wrist"], ["Left knee", "Left ankle"], ["Left shoulder", "Neck"],
                   ["Neck", "Right shoulder"], ["Right hip", "Right knee"], ["Right knee", "Right ankle"],
                   ["Left hip", "Left knee"], ["Right ankle", "Right heel"], ["Right ankle", "Right toe"],
                   ["Left ankle", "Left heel"], ["Left ankle", "Left toe"], ["Neck", "Head"]]

angle_bounds = {
    'trunk': [[0, 10], [10, 20], [20, 30], [30, 40], [40, sys.maxsize]],
    'arm': [[0, 20], [20, 40], [40, 60], [60, 80], [80, sys.maxsize]],  
    'fore_arm': [[0, 25], [25, 50], [50, 75], [75, 100], [100, sys.maxsize]],
    'thigh': [[0, 10], [10, 20], [20, 30], [30, 40], [40, sys.maxsize]],
    'leg': [[0, 15], [15, 25], [25, 35], [35, 45], [45, sys.maxsize]],
}

colors = [[(0, 255, 0)], [(0, 255, 0), (0, 103, 255)], [(0, 103, 255)], [(0, 103, 255), (0, 0, 255)], [(0, 0, 255)]]
# colors = [[solid: green], [gradient: green_to_orange], [solid: orange], [gradient: orange_to_red], [solid: red]]

def get_color_helper(joint_name, angle):
    
    color = coloring.get_color(joint_name, angle_bounds, angle, colors)

    return color

def getColor(joints, deviations):

    if (joints == ["Left hip", "Left shoulder"] or \
        joints == ["Right hip", "Left hip"] or \
        joints == ["Right hip", "Right shoulder"] or \
        joints == ["Left shoulder", "Thorax"] or \
        joints == ["Thorax", "Right shoulder"]):
        angle = math.sqrt(deviations[0]**2 + deviations[1]**2)
        return get_color_helper("trunk", angle)
    
    if (joints == ["Right shoulder", "Right elbow"]):
        angle = deviations[3]
        return get_color_helper("arm", angle)
    if (joints == ["Left shoulder", "Left elbow"]):
        angle = deviations[2]
        return get_color_helper("arm", angle)
    if (joints == ["Right elbow", "Right wrist"]):
        angle = deviations[5]
        return get_color_helper("fore_arm", angle)
    if (joints == ["Left elbow", "Left wrist"]):
        angle = deviations[4]
        return get_color_helper("fore_arm", angle)
    
    if (joints == ["Right hip", "Right knee"]):
        angle = deviations[7]
        return get_color_helper("thigh", angle)
    if (joints == ["Left hip", "Left knee"]):
        angle = deviations[6]
        return get_color_helper("thigh", angle)
    if (joints == ["Right knee", "Right ankle"]):
        return get_color_helper("leg", deviations[9])
    if (joints == ["Left knee", "Left ankle"]):
        return get_color_helper("leg", deviations[8])
    
    return (0, 255, 0)

def get_head_point(joints, thresh = 0.5):

    is_valid = valid_head_point(joints, thresh = thresh)
        
    if not is_valid:
        return np.array([-1, -1])

    neck_p = joints[jointMapping["Neck"], :2]
    lear_p = joints[jointMapping["Left ear"], :2]
    rear_p = joints[jointMapping["Right ear"], :2]

    neck_p = np.array(neck_p)
    lear_p = np.array(lear_p)
    rear_p = np.array(rear_p)
    
    avg_ear = 0.5 * (lear_p + rear_p)
    head_p = utils.interpolate_point(avg_ear, neck_p, t=0.3)

    return head_p

def get_point(joints, joint_name):

    if (joint_name == "Head"):
        return get_head_point(joints)
        
    return joints[jointMapping[joint_name], :2]

def valid_head_point(joints, thresh):

    joint_names = ["Neck base", "Left eye", "Right eye"]

    for j in joint_names:
        p = joints[jointMapping[j], :2]
        score = joints[jointMapping[j], 2]
        
        if (p[0] < 0 or p[1] < 0 or score < thresh):
            return False

    return True

def drawSkeleton(frame, joints_2d, deviations, base, thresh):

    sx, sy = 1080, 1920

    for pair in skeletonMapping:
        startPoint = get_point(joints_2d, pair[0])
        endPoint = get_point(joints_2d, pair[1])
        start_score = joints_2d[jointMapping[pair[0]], 2]
        end_score = joints_2d[jointMapping[pair[1]], 2]

        if (startPoint[0] < 0 or startPoint[1] < 0 or endPoint[0] < 0 or endPoint[1] < 0 or start_score < thresh or end_score < thresh):
            continue
        startPoint = (int(startPoint[0] * sx), int(startPoint[1] * sy))
        endPoint = (int(endPoint[0] * sx), int(endPoint[1] * sy))

        if base:color = (0, 255, 0)
        else: color = getColor(pair, deviations)

        frame = cv2.line(frame, startPoint, endPoint, color, utils.get_skeleton_thickness(frame.shape))
    
    return frame

def render_video(base_info, cand_info, deviations):

    base_video, bjoints_2d = base_info
    cand_video, cjoints_2d = cand_info

    num_bframes = bjoints_2d.shape[0]
    num_cframes = cjoints_2d.shape[0]

    print('num_bframes ', num_bframes)
    print('num_cframes ', num_cframes)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Define video writer
    video_writer = cv2.VideoWriter(out_video_path,
                                fourcc, cvideo.fps, (1080, 960))
    
    min_len = min(num_bframes, num_cframes)

    bindices = list(range(num_bframes))
    cindices = list(range(num_cframes))

    bids = utils.pick_equidistant_elements(bindices, min_len)
    cids = utils.pick_equidistant_elements(cindices, min_len)

    for bidx, cidx in zip(bids, cids):
        
        bframe = base_video[bidx].copy()
        bjoint_2d = bjoints_2d[bidx]
        deviation = deviations[cidx]

        bframe = drawSkeleton(bframe, bjoint_2d, deviation, base=True, thresh=conf_thresh)
                
        cframe = cand_video[cidx].copy()
        cjoint_2d = cjoints_2d[cidx]
        cframe = drawSkeleton(cframe, cjoint_2d, deviation, base=False, thresh=conf_thresh)
        deviation = np.round(deviation, 2)    
        cframe = utils.print_deviations(cframe, deviation)

        bframe = cv2.resize(bframe, None, fx = 0.5, fy = 0.5)
        cframe = cv2.resize(cframe, None, fx = 0.5, fy = 0.5)

        concat = np.hstack((bframe, cframe))
        
        print('concat ', concat.shape)
        video_writer.write(concat)

        cv2.imshow('concat ', concat)
        cv2.waitKey(-1)

    video_writer.release()

if __name__ == "__main__":

    root_pose = "/home/tumeke-balaji/Documents/results/delta/bin_files/"
    base_dir = root_pose + "/baseline/"
    cand_dir = root_pose + "/candidate/"
    dev_path = root_pose + "deviations.bin"
    
    joint_bins = ["joints2d.bin", "scores.bin", "joints3d.bin"]
    base_joint_bins = [os.path.join(base_dir, joint_bin) for joint_bin in joint_bins]
    bjoints_2d, bjoints_3d = utils.get_info(base_joint_bins)
    print('bjoints_2d, bjoints_3d ', bjoints_2d.shape, bjoints_3d.shape)

    cand_joint_bins = [os.path.join(cand_dir, joint_bin) for joint_bin in joint_bins]
    cjoints_2d, cjoints_3d = utils.get_info(cand_joint_bins)
    print('cjoints_2d, cjoints_3d ', cjoints_2d.shape, cjoints_3d.shape)

    deviations = []
    utils.read_dev_file(deviations, dev_path)
    print('deviations ', len(deviations))

    base_video = base_dir + "baseline.mov"
    cand_video = cand_dir + "candidate.mov"
    out_video_path = root_pose + "compare.mov"

    bvideo = mmcv.VideoReader(base_video)
    cvideo = mmcv.VideoReader(cand_video)
    conf_thresh = 0.35

    render_video([bvideo, bjoints_2d], [cvideo, cjoints_2d], deviations)