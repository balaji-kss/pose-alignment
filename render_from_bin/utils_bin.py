import numpy as np
import cv2

def interpolate_point(A, B, t):

    x1, y1 = A
    x2, y2 = B
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2

    return np.array([x, y])

def read_joint_files(joints_2d, local_file, num_bytes_per_person, num_people):
    f = open(local_file, "rb")
    num_frames = 0
    while True:
        for j in range(num_people):
            byte_arr = f.read(num_bytes_per_person)
            if (byte_arr == b''):
                f.close()
                return num_frames
            joints_2d.append(
                np.frombuffer(byte_arr, dtype=np.float32).tolist()
            )
        num_frames += 1
    f.close()
    return num_frames

def get_info(paths):

    joints_2d_path, joints_2d_score_path, joints_3d_path = paths

    num_people = 1
    wrnch_joint_len = 25
    joints_2d = []
    joints_3d = []
    joints_2d_score = []

    num_bytes_per_person = wrnch_joint_len * 2 * 4
    read_joint_files(joints_2d, joints_2d_path, num_bytes_per_person, num_people) # N, 50
    joints_2d_np = np.array(joints_2d).reshape((-1, wrnch_joint_len, 2))

    num_bytes_per_person = wrnch_joint_len * 1 * 4
    read_joint_files(joints_2d_score, joints_2d_score_path, num_bytes_per_person, num_people) # N, 25   
    joints_2d_score_np = np.array(joints_2d_score).reshape((-1, wrnch_joint_len, 1))
    
    joints_2d_np = np.concatenate((joints_2d_np, joints_2d_score_np), axis=-1)

    num_bytes_per_person = wrnch_joint_len * 3 * 4
    read_joint_files(joints_3d, joints_3d_path, num_bytes_per_person, num_people) # N, 75   
    joints_3d_np = np.array(joints_3d).reshape((-1, wrnch_joint_len, 3))

    return joints_2d_np, joints_3d_np

def get_skeleton_thickness(frame_size):
    smaller_dim = int(min(frame_size[0], frame_size[1]))
    thickness = int(max(1, smaller_dim/120))
    return thickness

def read_dev_file(deviations, bin_path):

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

def pick_equidistant_elements(A, m):
    
    idx = np.round(np.linspace(0, len(A) - 1, m)).astype(int)

    return np.array(A)[idx]

def print_deviations(frame, deviations, w=500):

    trunk_dev, trunk_twist_dev, larm_dev, rarm_dev, lfarm_dev, rfarm_dev, lthigh_dev, rthigh_dev, lleg_dev, rleg_dev = deviations

    start = 100
    step = 30
    
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