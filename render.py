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

arm = [[0, 10], [10, 20], [20, 65], [65, 85], [85, sys.maxsize]]
leg = [[0, 5], [5, 10], [10, 35], [35, 60], [60, sys.maxsize]]
trunk = [[0, 7.5], [7.5, 15], [15, 30], [30, 45], [45, sys.maxsize]]

angle_bounds = {
    'trunk': trunk,
    'arm': arm,  
    'fore_arm': arm,
    'inter_fore_arm': arm,
    'thigh': leg,
    'leg': leg,
}

# angle_bounds = {
#     # 'trunk': [[0, 15], [15, 25], [25, 35], [35, 45], [45, sys.maxsize]], # Incorrect closing overhead bin, Incorrect lowering crew bag
#     'trunk': [[0, 10], [10, 20], [20, 30], [30, 40], [40, sys.maxsize]], # Incorrect lift galley carrier, 
#     'arm': [[0, 20], [20, 40], [40, 60], [60, 80], [80, sys.maxsize]],  
#     'fore_arm': [[0, 25], [25, 50], [50, 75], [75, 100], [100, sys.maxsize]],
#     'inter_fore_arm': [[0, 15], [15, 30], [30, 45], [45, 60], [60, sys.maxsize]],
#     'thigh': [[0, 10], [10, 20], [20, 30], [30, 40], [40, sys.maxsize]],
#     'leg': [[0, 15], [15, 25], [25, 35], [35, 45], [45, sys.maxsize]],
# }

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
        angle = deviations[9]
        return get_color_helper("leg", angle)
    if (joints == ["Left knee", "Left ankle"]):
        angle = deviations[8]
        return get_color_helper("leg", angle)
    
    return (0, 255, 0)

def get_color_helper(joint_name, angle):
    
    color = coloring.get_color(joint_name, angle_bounds, angle, colors)

    return color

def get_skeleton_thickness(frame_size):
    smaller_dim = int(min(frame_size[0], frame_size[1]))
    thickness = int(max(1, smaller_dim/120))
    return thickness

def drawSkeleton(frame, joints_2d, deviations, base, thresh):

    # if base:
    #     sx, sy = 1080, 1920
    # else:
    #     sx, sy = 1, 1

    sx, sy = 1, 1

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

        frame = cv2.line(frame, startPoint, endPoint, color, get_skeleton_thickness(frame.shape))
    
    return frame

def print_deviations(frame, deviations, w=100):

    trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev, farm_dev, fleg_dev = deviations

    fh, fw = frame.shape[0], frame.shape[1]

    if fh == 1080 and fw == 1920:
        w = 500
        start = 100
    else:
        start = 600
        w = 100

    step = 30
    thickness = 2

    frame = cv2.putText(frame, "Trunk: " + str(trunk_dev), (w, start), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA) 
    y = start + step
    frame = cv2.putText(frame, "Trunk Twist: " + str(trunk_twist_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA) 
    y = start + 2 * step
    frame = cv2.putText(frame, "Left arm: " + str(larm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 3 * step
    frame = cv2.putText(frame, "Right arm: " + str(rarm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                  1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 4 * step
    frame = cv2.putText(frame, "Left Fore arm: " + str(lfarm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 5 * step
    frame = cv2.putText(frame, "Right Fore arm: " + str(rfarm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 6 * step
    frame = cv2.putText(frame, "Left thigh: " + str(lthigh_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 7 * step
    frame = cv2.putText(frame, "Right thigh: " + str(rthigh_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 8 * step
    frame = cv2.putText(frame, "Left Leg: " + str(lleg_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 9 * step
    frame = cv2.putText(frame, "Right leg: " + str(rleg_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 10 * step
    frame = cv2.putText(frame, "Fore arm: " + str(farm_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    y = start + 11 * step
    frame = cv2.putText(frame, "Across leg: " + str(fleg_dev), (w, y), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), thickness, cv2.LINE_AA)
    
    return frame

def crop_images(bframe, cframe, dtwframe=None):

    bh, bw = bframe.shape[:2]
    ch, cw = cframe.shape[:2]
    
    sz = np.min([bh, bw, ch, cw])
    bframe = bframe[bh//2 - sz//2 : bh//2 + sz//2, bw//2 - sz//2 : bw//2 + sz//2]
    cframe = cframe[ch//2 - sz//2 : ch//2 + sz//2, cw//2 - sz//2 : cw//2 + sz//2]

    if dtwframe is not None:
        dh, dw = dtwframe.shape[:2]
        dtwframe = dtwframe[dh//2 - sz//2 : dh//2 + sz//2, dw//2 - sz//2 : dw//2 + sz//2]

    return bframe, cframe, dtwframe 

def render_results(bvideo_path, cvideo_path, dtw_video_path, output_video_path, path_ids, deviations_list, thresh):

    bvideo = mmcv.VideoReader(bvideo_path)
    cvideo = mmcv.VideoReader(cvideo_path)
    dtwvideo = mmcv.VideoReader(dtw_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Define video writer
    video_writer = cv2.VideoWriter(output_video_path,
                                fourcc, cvideo.fps, (1620, 540))
    
    # Iterate through both videos

    for t, (b, c, isalign) in enumerate(path_ids):

            # if not isalign:continue

            bframe = bvideo[b].copy()
            cframe = cvideo[c].copy()  
            dtwframe = dtwvideo[t].copy()  

            bjoints_2d = deviations_list[t]['bjoints_2d']
            cjoints_2d = deviations_list[t]['cjoints_2d']
            deviations = deviations_list[t]['deviations']

            bframe = drawSkeleton(bframe, bjoints_2d, deviations, base=True, thresh=thresh)

            cframe = drawSkeleton(cframe, cjoints_2d, deviations, base=False, thresh=thresh)
            cframe = print_deviations(cframe, deviations)

            if isalign:text = "   ALIGNED"
            else:text = "  NOT-ALIGNED"

            bframe = cv2.putText(bframe, "Frame: " + str(b), (500, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA) 
            cframe = cv2.putText(cframe, "Frame: " + str(c), (500, 500), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA) 
            
            bframe = cv2.resize(bframe, None, fx = 0.5, fy = 0.5)
            cframe = cv2.resize(cframe, None, fx = 0.5, fy = 0.5)
            dtwframe = dtwframe[:, 1080:]
            bframe, cframe, dtwframe = crop_images(bframe, cframe, dtwframe)
            
            concat = np.hstack((bframe, cframe))
            concat = np.hstack((concat, dtwframe))

            concat = cv2.putText(concat, text, (400, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA)
            
            video_writer.write(concat)
            cv2.imshow('output ', concat)
            cv2.waitKey(-1)

    video_writer.release()

def blur_face(frame, joints2d):

    face_pts = joints2d[17:]
    w, h, _ = np.max(face_pts, axis = 0) - np.min(face_pts, axis = 0)
    cx, cy, _ = 0.5 * (np.max(face_pts, axis = 0) + np.min(face_pts, axis = 0))
    sz = 0.5 * (w + h) * 2.5

    xmin, ymin = cx - 0.5 * sz, cy - 0.5 * sz
    xmax, ymax = cx + 0.5 * sz, cy + 0.5 * sz

    xmin, ymin = int(xmin), int(ymin)
    xmax, ymax = int(xmax), int(ymax)

    frame[ymin:ymax, xmin:xmax] = cv2.GaussianBlur(frame[ymin:ymax, xmin:xmax], (51, 51), 0)

    return frame

def render_compare_video(bvideo_path, cvideo_path, output_video_path, base_dict, deviations_dict, smooth_pair_ids, thresh, isblur = False, isauto=True):

    bvideo = mmcv.VideoReader(bvideo_path)
    cvideo = mmcv.VideoReader(cvideo_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    video_writer = cv2.VideoWriter(output_video_path,
                                fourcc, cvideo.fps, (1080, 540))

    min_len = min(len(base_dict), len(deviations_dict))

    if isauto:
        bids = utils.pick_equidistant_elements(list(base_dict.keys()), min_len)
        cids = utils.pick_equidistant_elements(list(deviations_dict.keys()), min_len)

        bids = np.expand_dims(bids, axis=1)
        cids = np.expand_dims(cids, axis=1)
        
        pair_lst = np.hstack((bids, cids))
    else:
        pair_lst = smooth_pair_ids

    for bidx, cidx, isalign in pair_lst:
        
        if bidx > max(list(base_dict.keys())) or cidx > max(list(deviations_dict.keys())):continue

        bframe = bvideo[bidx].copy()
        deviations = base_dict[bidx][0]
        bjoints_2d = base_dict[bidx][1]

        cframe = cvideo[cidx].copy()
        deviations = deviations_dict[cidx][0]
        cjoints_2d = deviations_dict[cidx][1]

        if isblur:
            bframe = blur_face(bframe, bjoints_2d)
            cframe = blur_face(cframe, cjoints_2d)

        bframe = drawSkeleton(bframe, bjoints_2d, deviations, base=True, thresh=thresh)
            
        if isalign:
            cframe = drawSkeleton(cframe, cjoints_2d, deviations, base=False, thresh=thresh)    

        bframe, cframe, _ = crop_images(bframe, cframe)
        ch, cw = cframe.shape[:2]

        if not isalign:
            cv2.rectangle(cframe, (0, 0), (cw - 1, ch - 1), (0, 0, 255), 20)

        concat = np.hstack((bframe, cframe))
        concat = cv2.resize(concat, None, fx = 0.5, fy = 0.5)
        video_writer.write(concat)

        cv2.imshow('output ', concat)
        cv2.waitKey(-1)
        
    video_writer.release()

def render_cand_video(cvideo_path, output_video_path, deviations_dict, thresh):

    cvideo = mmcv.VideoReader(cvideo_path)
    cw, ch = cvideo.resolution
    cw, ch = int(cw//2), int(ch//2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Define video writer
    video_writer = cv2.VideoWriter(output_video_path,
                                fourcc, cvideo.fps, (cw, ch))
    
    # Iterate through both videos
    num_frames = len(deviations_dict)

    for i in range(num_frames):

            # if not isalign:continue
            cframe = cvideo[i].copy()  
            deviations = deviations_dict[i][0]
            cjoints_2d = deviations_dict[i][1]

            cframe = print_deviations(cframe, deviations, w=100)
            cframe = drawSkeleton(cframe, cjoints_2d, deviations, base=False, thresh=thresh)

            cframe = cv2.putText(cframe, "Frame: " + str(i), (500, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                   1, (0, 0, 255), 2, cv2.LINE_AA)
            cframe = cv2.resize(cframe, None, fx = 0.5, fy = 0.5)

            video_writer.write(cframe)
            cv2.imshow('output ', cframe)
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