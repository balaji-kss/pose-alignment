import math
import sys
import numpy as np

angle_bounds = {
    'trunk': [[0, 10], [10, 20], [20, 30], [30, 40], [40, sys.maxsize]],
    'arm': [[0, 20], [20, 40], [40, 60], [60, 80], [80, sys.maxsize]],  
    'fore_arm': [[0, 25], [25, 50], [50, 75], [75, 100], [100, sys.maxsize]],
    'thigh': [[0, 10], [10, 20], [20, 30], [30, 40], [40, sys.maxsize]],
    'leg': [[0, 15], [15, 25], [25, 35], [35, 45], [45, sys.maxsize]],
}

colors = [[(0, 255, 0)], [(0, 255, 0), (0, 103, 255)], [(0, 103, 255)], [(0, 103, 255), (0, 0, 255)], [(0, 0, 255)]]
# colors = [[solid: green], [gradient: green_to_orange], [solid: orange], [gradient: orange_to_red], [solid: red]]

# get_color_helper is same as defined here: https://github.com/tumeke-tech/cpu-compute/blob/f1e98662db2bb4a4624b5feda82ff975e82781ee/helpers/videos/render.py#L114

skeletonMapping = [["Left hip", "Left shoulder"], ["Right hip", "Left hip"], ["Right hip", "Right shoulder"],
                   ["Right shoulder", "Right elbow"], ["Left shoulder", "Left elbow"], ["Right elbow", "Right wrist"],
                   ["Left elbow", "Left wrist"], ["Left knee", "Left ankle"], ["Left shoulder", "Neck"],
                   ["Neck", "Right shoulder"], ["Right hip", "Right knee"], ["Right knee", "Right ankle"],
                   ["Left hip", "Left knee"], ["Right ankle", "Right heel"], ["Right ankle", "Right toe"],
                   ["Left ankle", "Left heel"], ["Left ankle", "Left toe"], ["Neck", "Head"]]

def getColor(joints, deviations):

    if (joints == ["Left hip", "Left shoulder"] or \
        joints == ["Right hip", "Left hip"] or \
        joints == ["Right hip", "Right shoulder"]):
        angle = math.sqrt(deviations[0]**2 + deviations[1]**2)
        angle = deviations[0]
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

# This function is used to read the deviations.bin for the delta training
def read_joint_files(deviations, bin_path):

    f = open(bin_path, "rb")
    num_frames = 0
    header_len = 4 * 4
    byte_arr = f.read(header_len)
    header = np.frombuffer(byte_arr, dtype=np.float32).tolist()
    num_people = int(header[0])
    num_angles_per_person = int(header[1]) * 4

    while True:
        for j in range(num_people):
            byte_arr = f.read(num_angles_per_person)
            if (byte_arr == b''):
                f.close()
                return num_frames
            deviations.append(
                np.frombuffer(byte_arr, dtype=np.float32).tolist()
            )
        num_frames += 1
    f.close()
    return num_frames

# deviations.bin is in the following format
# [num_people, num_items_per_pair, json_colors_version, reserved] + [baseline_frame_id, candidate_frame_id, is_align, angle_deviations] * num_pairs

# In our case:
# num_people = 1 
# num_items_per_pair = 13 - 3(baseline_frame_id, candidate_frame_id, is_align) + 10(angle deviations)
# baseline_frame_id = frame id of the baseline video - id starts from 0
# candidate_frame_id = corresponding frame id of the candidate video - id starts from 0
# is_align = 0 / 1 - whether the pair is aligned(1) or not(0)
# angle_deviations (following are the indices and the joint angle)
###  0 - trunk angle deviation
###  1 - trunk twist angle deviation
###  2 - left arm angle deviation 
###  3 - right arm angle deviation 
###  4 - left fore-arm angle deviation
###  5 - right fore-arm angle deviation
###  6 - left thigh angle deviation
###  7 - right thigh angle deviation
###  8 - left leg angle deviation
###  9 - right leg angle deviation

# num_people = 1
# num_items_per_frame = 10
# [num_people, num_items_per_frame, json_colors_version, reserved] + [angle_deviations] * num_candidate_frames