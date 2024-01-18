import numpy as np
import cv2
import sys
import coloring
import mmcv
import utils
import pickle
import math

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
                   ["Left elbow", "Left wrist"], ["Left knee", "Left ankle"], ["Left shoulder", "Thorax"],
                   ["Thorax", "Right shoulder"], ["Right hip", "Right knee"], ["Right knee", "Right ankle"],
                   ["Left hip", "Left knee"], ["Thorax", "Head"]]

angle_bounds = {
    'trunk': [[0, 15], [15, 25], [25, 35], [35, 40], [40, sys.maxsize]],
    'arm': [[0, 25], [25, 50], [50, 75], [75, 90], [90, sys.maxsize]],  
    'fore_arm': [[0, 30], [30, 60], [60, 90], [90, 120], [120, sys.maxsize]],
    'thigh': [[0, 10], [10, 20], [20, 30], [30, 40], [40, sys.maxsize]],
    'leg': [[0, 15], [15, 25], [25, 35], [35, 45], [45, sys.maxsize]],
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
        angle = math.sqrt(deviations[0]**2 + deviations[1]**2)
        return get_color_helper("trunk", angle)
    if (joints == ["Right shoulder", "Right elbow"]):
        angle = deviations[3]
        return get_color_helper("trunk", angle)
    if (joints == ["Left shoulder", "Left elbow"]):
        angle = deviations[2]
        return get_color_helper("arm", angle)
    if (joints == ["Left elbow", "Left wrist"]):
        angle = deviations[4]
        return get_color_helper("fore_arm", angle)
    if (joints == ["Right elbow", "Right wrist"]):
        angle = deviations[5]
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

def get_color_helper(joint_name, angle):
    
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

        frame = cv2.line(frame, startPoint, endPoint, color, get_skeleton_thickness(frame.shape))
    
    return frame

def print_deviations(frame, deviations):

    trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev = deviations

    start = 40
    step = 30
    w = 700
    frame = cv2.putText(frame, "Trunk: " + str(trunk_dev), (w, start), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA) 
    y = start + step
    frame = cv2.putText(frame, "Trunk Twist: " + str(trunk_twist_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA) 
    y = start + 2 * step
    frame = cv2.putText(frame, "Left arm: " + str(larm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    y = start + 3 * step
    frame = cv2.putText(frame, "Right arm: " + str(rarm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                  1, (0, 0, 255), 1, cv2.LINE_AA)
    y = start + 4 * step
    frame = cv2.putText(frame, "Left Fore arm: " + str(lfarm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    y = start + 5 * step
    frame = cv2.putText(frame, "Right Fore arm: " + str(rfarm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    y = start + 6 * step
    frame = cv2.putText(frame, "Left thigh: " + str(lthigh_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    y = start + 7 * step
    frame = cv2.putText(frame, "Right thigh: " + str(rthigh_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    y = start + 8 * step
    frame = cv2.putText(frame, "Left Leg: " + str(lleg_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    y = start + 9 * step
    frame = cv2.putText(frame, "Right leg: " + str(rleg_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 1, cv2.LINE_AA)
    
    return frame

def valid_dev(dev, bjoints_2d, cjoints_2d, idxs, thresh = 0.35):

    for idx in idxs:
        if bjoints_2d[idx, 2] < thresh or cjoints_2d[idx, 2] < thresh:
            return 0.0
        
    return dev

def mask_dev(deviations, bjoints_2d, cjoints_2d, thresh):

    # NOTE: check mask for trunk
    # NOTE: get neck angle

    trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev = deviations

    larm_dev = valid_dev(larm_dev, bjoints_2d, cjoints_2d, [11, 12], thresh=thresh)
    rarm_dev = valid_dev(rarm_dev, bjoints_2d, cjoints_2d, [14, 15], thresh=thresh)

    lfarm_dev = valid_dev(lfarm_dev, bjoints_2d, cjoints_2d, [11, 12, 13], thresh=thresh)
    rfarm_dev = valid_dev(rfarm_dev, bjoints_2d, cjoints_2d, [14, 15, 16], thresh=thresh)

    lthigh_dev = valid_dev(lthigh_dev, bjoints_2d, cjoints_2d, [4, 5], thresh=thresh)
    rthigh_dev = valid_dev(rthigh_dev, bjoints_2d, cjoints_2d, [1, 2], thresh=thresh)
    
    lleg_dev = valid_dev(lleg_dev, bjoints_2d, cjoints_2d, [4, 5, 6], thresh=thresh)
    rleg_dev = valid_dev(rleg_dev, bjoints_2d, cjoints_2d, [1, 2, 3], thresh=thresh)

    return [trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev]

def render_results(bvideo_path, cvideo_path, output_video_path, path_ids, deviations_list, thresh):

    bvideo = mmcv.VideoReader(bvideo_path)
    cvideo = mmcv.VideoReader(cvideo_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Define video writer
    video_writer = cv2.VideoWriter(output_video_path,
                                fourcc, cvideo.fps, (1080, 960))


    # Iterate through both videos
    for t, (b, c, isalign) in enumerate(path_ids):

            bframe = bvideo[b].copy()
            cframe = cvideo[c].copy()  

            bjoints_2d = deviations_list[t]['bjoints_2d']
            cjoints_2d = deviations_list[t]['cjoints_2d']
            deviations = deviations_list[t]['deviations']
            
            deviations = mask_dev(deviations, bjoints_2d, cjoints_2d, thresh)

            bframe = drawSkeleton(bframe, bjoints_2d, deviations, base=True, thresh=thresh)

            cframe = print_deviations(cframe, deviations)
            cframe = drawSkeleton(cframe, cjoints_2d, deviations, base=False, thresh=thresh)

            if isalign:text = "   ALIGNED"
            else:text = "  NOT-ALIGNED"

            bframe = cv2.putText(bframe, "Frame: " + str(b), (500, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA) 
            cframe = cv2.putText(cframe, "Frame: " + str(c), (500, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA) 
            
            bframe = cv2.resize(bframe, None, fx = 0.5, fy = 0.5)
            cframe = cv2.resize(cframe, None, fx = 0.5, fy = 0.5)
            concat = np.hstack((bframe, cframe))
            concat = cv2.putText(concat, text, (400, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA)
            
            video_writer.write(concat)
            cv2.imshow('output ', concat)
            cv2.waitKey(-1)

    video_writer.release()

if __name__ == "__main__":

    act_name = 'Serving_from_Basket' # 'Serving_from_Basket' # 'Pushing_cart' # 'Lower_Galley_Carrier'
    root_pose = '/home/tumeke-balaji/Documents/results/delta/joints/' + act_name + '/'    

    bvideo_path = root_pose + '/baseline/baseline.mov'
    cvideo_path = root_pose + '/candidate/candidate.mov'
    out_video_path = root_pose + act_name + '.mov'
    deviation_path = root_pose + 'deviations.pkl'
    
    with open(deviation_path, 'rb') as f:
        deviations = pickle.load(f)

    render_results(bvideo_path, cvideo_path, out_video_path, deviations)