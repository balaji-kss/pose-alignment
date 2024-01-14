import numpy as np
import cv2

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


def get_head_point(data_2d, joint_table, frame_no, thresh = 0.5):

    is_valid = valid_head_point(data_2d, joint_table, frame_no, thresh = thresh)
        
    if not is_valid:
        return np.array([-1, -1])

    neck_p = data_2d[frame_no][jointMapping["Neck"] * 2:jointMapping["Neck"] * 2 + 2]
    lear_p = data_2d[frame_no][jointMapping["Left ear"] * 2:jointMapping["Left ear"] * 2 + 2]
    rear_p = data_2d[frame_no][jointMapping["Right ear"] * 2:jointMapping["Right ear"] * 2 + 2]

    neck_p = np.array(neck_p)
    lear_p = np.array(lear_p)
    rear_p = np.array(rear_p)
    
    avg_ear = 0.5 * (lear_p + rear_p)
    head_p = interpolate_point(avg_ear, neck_p, t=0.3)

    return head_p

def valid_head_point(data_2d, joint_table, frame_no, thresh):

    joint_names = ["Neck base", "Left eye", "Right eye"]

    for j in joint_names:
        p = data_2d[frame_no][jointMapping[j] * 2:jointMapping[j] * 2 + 2]
        score = joint_table.jointScores[frame_no][jointMapping[j]]
        
        if (p[0] < 0 or p[1] < 0 or score < thresh):
            return False

    return True

def interpolate_point(A, B, t):

    x1, y1 = A
    x2, y2 = B
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2

    return np.array([x, y])

def get_skeleton_thickness(frame_size):
    smaller_dim = int(min(frame_size[0], frame_size[1]))
    thickness = int(max(1, smaller_dim/120))
    return thickness


def get_point(joint_table, frame_no, joint_name):
    data_2d = joint_table.joints2D

    if (joint_name == "Head"):
        return get_head_point(data_2d, joint_table, frame_no)
        
    return data_2d[frame_no][jointMapping[joint_name] * 2:jointMapping[joint_name] * 2 + 2]

def drawSkeleton(frame, frameNo, joint_table, prevColor, is_portrait):

    data_2d = joint_table.joints2D
    data_3d = joint_table.processed_3D_joints
    
    if (len(data_2d[frameNo]) == 0):
        return frame
            
    for pair in skeletonMapping:
        startPoint = get_point(joint_table, frameNo, pair[0])
        endPoint = get_point(joint_table, frameNo, pair[1])
        start_score = joint_table.jointScores[frameNo][jointMapping[pair[0]]]
        end_score = joint_table.jointScores[frameNo][jointMapping[pair[1]]]
        if (startPoint[0] < 0 or startPoint[1] < 0 or endPoint[0] < 0 or endPoint[1] < 0 or start_score < .35 or end_score < .35):
            continue
        startPoint = (int(startPoint[0] * frame.shape[1]), int(startPoint[1] * frame.shape[0]))
        endPoint = (int(endPoint[0] * frame.shape[1]), int(endPoint[1] * frame.shape[0]))
        color = getColor(pair, data_3d, frameNo, prevColor)

        frame = cv2.line(frame, startPoint, endPoint, color, get_skeleton_thickness(frame.shape))
    
    return frame

def getColor(joints, data_3d, frame_no, prevColor):
    if (joints == ["Left hip", "Left shoulder"] or \
        joints == ["Right hip", "Left hip"] or \
        joints == ["Right hip", "Right shoulder"]):
        return get_color_helper("Hip", data_3d, frame_no, prevColor)
    if (joints == ["Right shoulder", "Right elbow"]):
        return get_color_helper("Right shoulder", data_3d, frame_no, prevColor)
    if (joints == ["Left shoulder", "Left elbow"]):
        return get_color_helper("Left shoulder", data_3d, frame_no, prevColor)
    if (joints == ["Left elbow", "Left wrist"]):
        return get_color_helper("Left elbow", data_3d, frame_no, prevColor)
    if (joints == ["Right elbow", "Right wrist"]):
        return get_color_helper("Right elbow", data_3d, frame_no, prevColor)
    if (joints == ["Neck", "Head"]):
        return get_color_helper("Neck", data_3d, frame_no, prevColor)
    return (0, 255, 0)

def get_color_helper(jointClass, data_3d, frame_no, prevColor):
    angle = data_3d[jointClass][frame_no]

    if (angle == -1):
        if ("elbow" in jointClass):
            return prevColor["elbow"]
        if "Neck" in jointClass:
            return (0, 255, 0)

        return prevColor[jointClass]

    new_color = skeleton_coloring.get_color(
        jointClass, angle
    )

    if ("elbow" in jointClass):
        side = jointClass.split(" ")[0]
        shoulder_angle_key = f"{side} shoulder"
        shoulder_angle = data_3d[shoulder_angle_key][frame_no]
        cutoff = skeleton_coloring.SKELETON_COLORING["constants"]["lower_arm_cutoff"]
        if (shoulder_angle <= cutoff):
            new_color = skeleton_coloring.SKELETON_COLORING[jointClass]["default"]["Color"]

    
    prevColor[jointClass] = new_color

    return prevColor[jointClass]