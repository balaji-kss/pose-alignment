import numpy as np
import cv2
import sys
import coloring
import mmcv
import utils
import pickle

jointMapping = {
    "Neck": 9,
    "Thorax": 8,
    "Head": 10,
    "Right shoulder": 14,
    "Left shoulder": 11,
    "Right elbow": 15,
    "Left elbow": 12,
    "Right wrist": 16,
    "Left wrist": 13,
    "Right hip": 1,
    "Left hip": 4,
    "Right knee": 2,
    "Left knee": 5,
    "Left ankle": 6,
    "Right ankle": 3,
    "Left eye": 18,
    "Right eye": 19,
    "Left ear": 20,
    "Right ear": 21,
    "Neck base": 22
}

skeletonMapping = [["Left hip", "Left shoulder"], ["Right hip", "Left hip"], ["Right hip", "Right shoulder"],
                   ["Right shoulder", "Right elbow"], ["Left shoulder", "Left elbow"], ["Right elbow", "Right wrist"],
                   ["Left elbow", "Left wrist"], ["Left knee", "Left ankle"], ["Left shoulder", "Neck"],
                   ["Neck", "Right shoulder"], ["Right hip", "Right knee"], ["Right knee", "Right ankle"],
                   ["Left hip", "Left knee"], ["Neck", "Head"]]

angle_bounds = {
    'trunk': [[0, 7.5], [7.5, 15], [15, 22.5], [22.5, 30], [30, sys.maxsize]],
    'arm': [[0, 15], [15, 30], [30, 40], [40, 50], [50, sys.maxsize]],  
    'fore_arm': [[0, 15], [15, 30], [30, 40], [40, 50], [50, sys.maxsize]],
    'thigh': [[0, 7.5], [7.5, 15], [15, 22.5], [22.5, 30], [30, sys.maxsize]],
    'leg': [[0, 15], [15, 30], [30, 40], [40, 50], [50, sys.maxsize]],
}

colors = [[(0, 255, 0)], [(0, 255, 0), (0, 103, 255)], [(0, 103, 255)], [(0, 103, 255), (0, 0, 255)], [(0, 0, 255)]]

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
    head_p = interpolate_point(avg_ear, neck_p, t=0.3)

    return head_p

def valid_head_point(joints, thresh):

    joint_names = ["Neck base", "Left eye", "Right eye"]

    for j in joint_names:
        p = joints[jointMapping[j], :2]
        score = joints[jointMapping[j], 2]
        
        if (p[0] < 0 or p[1] < 0 or score < thresh):
            return False

    return True

def interpolate_point(A, B, t):

    x1, y1 = A
    x2, y2 = B
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2

    return np.array([x, y])

def get_point(joints, joint_name):

    if (joint_name == "Head"):
        return get_head_point(joints)
        
    return joints[jointMapping[joint_name], :2]

def getColor(joints, deviations):

    if (joints == ["Left hip", "Left shoulder"] or \
        joints == ["Right hip", "Left hip"] or \
        joints == ["Right hip", "Right shoulder"]):
        return get_color_helper(0, "trunk", deviations)
    if (joints == ["Right shoulder", "Right elbow"]):
        return get_color_helper(2, "arm", deviations)
    if (joints == ["Left shoulder", "Left elbow"]):
        return get_color_helper(1, "arm", deviations)
    if (joints == ["Left elbow", "Left wrist"]):
        return get_color_helper(3, "fore_arm", deviations)
    if (joints == ["Right elbow", "Right wrist"]):
        return get_color_helper(4, "fore_arm", deviations)
    
    if (joints == ["Right hip", "Right knee"]):
        return get_color_helper(6, "thigh", deviations)
    if (joints == ["Left hip", "Left knee"]):
        return get_color_helper(5, "thigh", deviations)
    if (joints == ["Right knee", "Right ankle"]):
        return get_color_helper(8, "leg", deviations)
    if (joints == ["Left knee", "Left ankle"]):
        return get_color_helper(7, "leg", deviations)
    
    return (0, 255, 0)

def get_color_helper(id_, joint_name, deviations):
    
    angle = deviations[id_]
    color = coloring.get_color(joint_name, angle_bounds, angle, colors)

    return color

def get_skeleton_thickness(frame_size):
    smaller_dim = int(min(frame_size[0], frame_size[1]))
    thickness = int(max(1, smaller_dim/120))
    return thickness

def drawSkeleton(frame, joints_2d, deviations, base=False, thresh=0.35):

    for pair in skeletonMapping:
        startPoint = get_point(joints_2d, pair[0])
        endPoint = get_point(joints_2d, pair[1])
        start_score = joints_2d[jointMapping[pair[0]], 2]
        end_score = joints_2d[jointMapping[pair[1]], 2]
        if (startPoint[0] < 0 or startPoint[1] < 0 or endPoint[0] < 0 or endPoint[1] < 0 or start_score < thresh or end_score < thresh):
            continue
        startPoint = (int(startPoint[0]), int(startPoint[1]))
        endPoint = (int(endPoint[0]), int(endPoint[1]))

        if base:color = (0, 255, 0)
        else: color = getColor(pair, deviations)

        print('startPoint ', startPoint)
        print('endPoint ', endPoint)
        print('color ', color)

        frame = cv2.line(frame, startPoint, endPoint, color, get_skeleton_thickness(frame.shape))
    
    return frame

def print_deviations(frame, deviations):

    

def render_results(bvideo_path, cvideo_path, deviations_list):

    bvideo = mmcv.VideoReader(bvideo_path)
    cvideo = mmcv.VideoReader(cvideo_path)

    v1_seg, v2_seg = utils.get_segs()

    # Iterate through both videos
    gidx = 0
    for i in range(len(v1_seg)):
        
        bseg, cseg = utils.normalize_segment(v1_seg[i], v2_seg[i]) # make sure the length of the segments are same
        print(len(bseg), len(cseg))

        for b, c in zip(bseg, cseg):

            bframe = bvideo[b]
            cframe = cvideo[c]  

            bjoints_2d = deviations_list[gidx]['bjoints_2d']
            cjoints_2d = deviations_list[gidx]['cjoints_2d']
            deviations = deviations_list[gidx]['deviations']

            bframe = drawSkeleton(bframe, bjoints_2d, deviations, base=True, thresh=0.35)
            cframe = drawSkeleton(cframe, cjoints_2d, deviations, base=False, thresh=0.35)

            bframe = cv2.resize(bframe, None, fx = 0.5, fy = 0.5)
            cframe = cv2.resize(cframe, None, fx = 0.5, fy = 0.5)

            gidx += 1

            cv2.imshow('Baseline ', bframe)
            cv2.imshow('Candidate ', cframe)
            cv2.waitKey(-1)

if __name__ == "__main__":

    act_name = 'Serving_from_Basket' # 'Pushing_cart' # 'Removing_Item_from_Bottom_of_Cart' # 'Lower_Galley_Carrier'
    root_pose = '/home/tumeke-balaji/Documents/results/delta/joints/' + act_name + '/'    

    bvideo_path = root_pose + '/baseline/baseline.mov'
    cvideo_path = root_pose + '/candidate/candidate.mov'
    deviation_path = root_pose + 'deviations.pkl'
    
    with open(deviation_path, 'rb') as f:
        deviations = pickle.load(f)

    render_results(bvideo_path, cvideo_path, deviations)