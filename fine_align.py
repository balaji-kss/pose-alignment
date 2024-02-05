import mmcv
import pickle
import cv2
import numpy as np
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import imageio
from tqdm import tqdm

def draw_2d_skeletons(frame, kpts_2d, color=(255, 0, 0), scale = (1, 1)):
    radius = 5
    connections_2d = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [18, 19], [19, 20], [20, 21], [22, 23], [23, 24], [24, 25], [26, 27], [27, 28], [28, 29], [30, 31], [31, 32], [32, 33], [34, 35], [35, 36], [36, 37], [38, 39], [39, 40], [40, 41]]
    
    for sk_id, sk in enumerate(connections_2d):
        if sk[0] > 16 or sk[1] > 16:continue
        pos1 = (int(kpts_2d[sk[0], 0]*scale[0]), int(kpts_2d[sk[0], 1]*scale[1]))
        pos2 = (int(kpts_2d[sk[1], 0]*scale[0]), int(kpts_2d[sk[1], 1]*scale[1]))
        cv2.line(frame, pos1, pos2, color, thickness=radius)
        
    return frame

def fetch_segment_frames(vid, poses, new_w, new_h, index, pose_win_sec=0.3):
    """ Since we move the sliding window forward in stride=pose_win/2 frames, for simplification, unless the last index,
    we only fetch pose_win//2 frames for each index
    """
    def get_saliency(person_dict):
        # Assuming the bounding box format is [x1, y1, x2, y2, score]
        bbox = person_dict['bbox']
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        area = w * h
        return area      
    stride = 1 # max(1, int(vid.fps * pose_win_sec / 2))
    clips = []
    st = index * stride
    ed = min(vid.frame_cnt, (index+1) * stride)
    for i in range(st, ed):
        frame1 = vid[i]
        scale = (new_w / frame1.shape[1], new_h / frame1.shape[0])
        frame1 = mmcv.imresize(frame1, (new_w, new_h))         
        if poses[i]:
            most_salient_person_1 = max(poses[i], key=get_saliency)
            most_salient_person_1['keypoints'] = most_salient_person_1['keypoints_2d'][:17]
            # draw_pose_frame_1 = vis_pose_result(pose_model, frame1, [most_salient_person_1])
            draw_pose_frame_1 = draw_2d_skeletons(frame1, most_salient_person_1['keypoints'], scale=scale)
            frame1 = draw_pose_frame_1[:,:,::-1]
        frame1 = mmcv.imresize(frame1, (new_w//2, new_h//2))
        clips.append(frame1)
    # clips = np.stack(clips)
    return clips

def generate_dtw_alignment_video(template_name, candidate_name, distance_np, dtw_path, task_path, pose_folders, normal_folders, pose_win_sec=0.3):

    template_pose_file = os.path.join(pose_folders[0], f"{template_name}.pkl")
    template_video_file = os.path.join(normal_folders[0], f"{template_name}_normal.mp4")

    candidate_pose_file = os.path.join(pose_folders[1], f"{candidate_name}.pkl")
    candidate_video_file = os.path.join(normal_folders[1], f"{candidate_name}_normal.mp4")

    # get the meta data of video file
    template_vid = mmcv.VideoReader(template_video_file)
    candidate_vid = mmcv.VideoReader(candidate_video_file)

    SHORT_SIDE = 480
    with open(template_pose_file, 'rb') as kp:
        template_poses = pickle.load(kp)    
    with open(candidate_pose_file, 'rb') as kp:
        candidate_poses = pickle.load(kp)            
    new_w, new_h = mmcv.rescale_size((template_vid.width, template_vid.height), (SHORT_SIDE, np.Inf))
    # new_w, new_h = template_vid.width, template_vid.height

    # Calculate the aspect ratio of the DTW plot
    plot_aspect_ratio = 1
    plot_new_height = int(new_w//2 * plot_aspect_ratio)

    # Set up the DTW path visualization
    fig, ax = plt.subplots()
    cax = ax.imshow(distance_np, origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')
    ax.plot(dtw_path[:, 1], dtw_path[:, 0], color='cyan', linestyle='-', linewidth=4, alpha=0.5)
    circle = Circle((0, 0), radius=4, color='red', fill=True)
    ax.add_patch(circle)
    # Initialize a text box for distance value
    distance_text = ax.text(0, 0, '', color='white', fontsize=20, ha='left', va='top', backgroundcolor='black')    
    canvas = FigureCanvas(fig)

    writer = imageio.get_writer(f"{task_path}/{template_name}_{candidate_name}_alignment-fa.mp4", fps=int(min(template_vid.fps, candidate_vid.fps)))
    pre_frames_1, pre_frames_2 = [], []

    for i, (index1, index2, _) in tqdm(enumerate(dtw_path)):
        # fetch the index1 / index2 segment frames from both videos
        frames_1 = fetch_segment_frames(template_vid, template_poses, new_w, new_h, index1, pose_win_sec)
        frames_2 = fetch_segment_frames(candidate_vid, candidate_poses, new_w, new_h, index2, pose_win_sec)
        num_f = min(len(frames_1), len(frames_2))

        # check constant index
        if i > 0 and index1 == dtw_path[i-1][0]:
            frames_1 = [ pre_frames_1[-1] for _ in range(num_f)]
        if i > 0 and index2 == dtw_path[i-1][1]:
            frames_2 = [ pre_frames_2[-1] for _ in range(num_f)]

        # Update the DTW path visualization
        circle.center = (index2, index1)
        # Update the distance text
        d = distance_np[index1, index2]
        distance_text.set_position((index2, index1))
        distance_text.set_text(f'{d:.2f}')   

        canvas.draw()  # Render the canvas as an image
        dtw_plot_image = np.fromstring(canvas.tostring_rgb(), dtype='uint8', sep='')
        dtw_plot_image = dtw_plot_image.reshape(canvas.get_width_height()[::-1] + (3,))

        dtw_plot_image = cv2.resize(dtw_plot_image, (new_w//2, plot_new_height))

        # Pad the DTW plot image to match the height of the video frames
        pad_height = (new_h//2 - plot_new_height) // 2
        dtw_plot_image_padded = np.pad(dtw_plot_image, ((pad_height, new_h//2 - plot_new_height - pad_height), (0, 0), (0, 0)), mode='constant', constant_values=0)
        # Combine the video frames and padded DTW plot image
        # dtw_plot_image = cv2.resize(dtw_plot_image, (new_w//2, plot_new_height))
        for f in range(num_f):
            combined_frame = np.concatenate((frames_1[f], frames_2[f], dtw_plot_image_padded), axis=1)
            writer.append_data(combined_frame)
        pre_frames_1 = frames_1
        pre_frames_2 = frames_2
    writer.close()

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

def find_indices_of_duplicates(array):

    seen = {}
    duplicate_indices = []
    for idx, element in enumerate(array):
        if element in seen:
            # Add the index of the duplicate
            duplicate_indices.append(idx)
        else:
            # Record the index of the first occurrence
            seen[element] = idx

    return duplicate_indices

def get_pmpjpe_align(bvideo, cvideo, bposes_3d, cposes_3d):

    dist_mat = dist_matrix(bvideo, cvideo, bposes_3d, cposes_3d)
    dist_mat_np = np.array(dist_mat)
    path = dtw_with_precomputed_distances(dist_mat_np)
    path_np = np.array(path).astype('int')
    
    indicesb = find_indices_of_duplicates(path_np[:, 0])
    indicesc = find_indices_of_duplicates(path_np[:, 1])

    path_np[:, 2] = 1
    
    path_np[indicesb + indicesc, 2] = 0

    return dist_mat_np, path_np.tolist()

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

    path = dtw_with_precomputed_distances(dist_mat_avg)
    print('path ', path)

    for b, c, dist in path:
        bframe = bvideo[b]
        cframe = cvideo[c]

        print(b, c, dist)
        cv2.putText(cframe, 'PMPJPE: ' + str(dist), (250, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        bframe = cv2.putText(bframe, "Frame: " + str(b), (500, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                1, (0, 0, 255), 2, cv2.LINE_AA) 
        cframe = cv2.putText(cframe, "Frame: " + str(c), (500, 40), cv2.FONT_HERSHEY_SIMPLEX,  
                1, (0, 0, 255), 2, cv2.LINE_AA) 
        # concat = np.hstack((bframe, cframe))
        # concat = cv2.resize(concat, None, fx=0.5, fy=0.5)
        # video_writer.write(concat)

        cv2.imshow('concat ', concat)
        cv2.waitKey(-1)

    video_writer.release()

if __name__ == "__main__":

    bact_name =  'Serving_from_Basket' # 'Lower_Galley_Carrier' # 'Serving_from_Basket' # 'Pushing_cart' # 'Lower_Galley_Carrier' # Stowing_carrier # 'Removing_Item_from_Bottom_of_Cart'
    cact_name =  'Serving_from_Basket'
    # ["baseline.mov", "candidate.mov"]

    root_path = '/home/tumeke-balaji/Documents/results/delta/input_videos/delta_all_data/delta_data/'
    base_path = root_path + bact_name
    cand_path = root_path + cact_name
    
    bvideo_path = base_path + "/videos/baseline14.mov"
    cvideo_path = cand_path + "/videos/candidate1.mov"

    out_video_path = '/home/tumeke-balaji/Documents/results/delta/input_videos/' + bact_name + '_' + cact_name + '_align.mov'

    bpose_path = root_path + bact_name + '/poses/joints/'
    cpose_path = root_path + cact_name + '/poses/joints/'

    bpose_path_3d = bpose_path + "baseline14_pose_3d.p"
    cpose_path_3d = cpose_path + "candidate1_pose_3d.p"

    bnormal_folder = base_path + '/videos/normalized_video/'
    cnormal_folder = cand_path + '/videos/normalized_video/'

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

    # get_matching(bvideo, cvideo, out_video_path, bposes_3d, cposes_3d)

    print('len bposes_3d ', len(bposes_3d))
    print('len cposes_3d ', len(cposes_3d), len(cvideo))

    distance_np, dtw_path = get_pmpjpe_align(bvideo, cvideo, bposes_3d, cposes_3d)

    template_name  = "baseline14" 
    candidate_name = "candidate1" 
    task_path = base_path
    pose_win_sec = 0.3

    generate_dtw_alignment_video(template_name, candidate_name, distance_np, np.array(dtw_path), task_path, [bpose_path, cpose_path], [bnormal_folder, cnormal_folder], pose_win_sec=pose_win_sec)