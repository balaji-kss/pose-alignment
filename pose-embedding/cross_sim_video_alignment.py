import torch
import matplotlib.pyplot as plt
import time
import os
import json
import glob
import mmcv
import pickle
import argparse
import numpy as np
import shutil

from scipy.signal import find_peaks
import cv2
import functools
import imageio
import io

from matplotlib.patches import Circle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from pose_embedding.common import models, keypoint_utils, loss_utils, data_utils, constants, keypoint_profiles, visualization_utils
from pose_embedding.api.temporal_embedding_feature import extract_embedding_temporal_feature
from pose_embedding.model.repnet_embedding import CrossSimilarityProbDistance
from pose_embedding.common.utils.dtw import dtw_with_precomputed_distances as dtw
from pose_embedding.common.utils.dtw import temporal_averaging
from tqdm import tqdm
from mmdet.apis import init_detector
from mmpose.apis import init_pose_model, vis_pose_result   

def parse_args():
    parser = argparse.ArgumentParser(description='Posture matching using pose embedding.')
    parser.add_argument(
        '--input',
        default="/home/jun/data/delta-clips/stowing-carrier" , #stow-full-cart", #removing-item-from-bottom-of-cart", #pushing-cart", #serving-from-basket", # lift-galley-carrier", #closing-overhead-bin", #lower-galley-carrier" , #  , , 
        help='video folder is $input/videos, pose folder is $input/poses/joints')  
    parser.add_argument(
        '--output',
        default= "data/output/vis", #
        help='repetition short videos')       
    parser.add_argument(
        '--resume-ckpt',
        default= 'checkpoints/fullbody-occlusion-1018-0.5-0.2/training_model_10_0.pth', 
        help='resume training from previous checkpoint')   
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='CUDA device id')
    
    args = parser.parse_args()   
    args.video_folder = os.path.join(args.input, 'videos') 
    args.pose_folder = os.path.join(args.input, 'poses/joints') 
    args.pose_embedding_folder = os.path.join(args.input, 'poses/embedding') 
    args.normal_folder = os.path.join(args.input, 'videos/normalized_video') 

    args.model_name = args.resume_ckpt.split('/')[-1]

    mmcv.mkdir_or_exist(args.pose_embedding_folder)
    mmcv.mkdir_or_exist(args.output)
    return args 

def visualize_dtw_path(distance_matrix, dtw_path_indices, baseline, candidate, path):
    # Creating the visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(distance_matrix, origin='lower', cmap='viridis', interpolation='nearest', aspect='auto')
    fig.colorbar(cax, ax=ax, label='Distance')
    ax.plot(dtw_path_indices[:, 1], dtw_path_indices[:, 0], color='r', linewidth=4, linestyle='-')
    ax.set_title('DTW Alignment Path')
    ax.set_xlabel(f'Segments in {candidate}')
    ax.set_ylabel(f'Segments in {baseline}')
    ax.grid(False)    # axes.show()
    pathname = os.path.join(path, f"{baseline}_{candidate}_dtw_path.png")
    plt.savefig(pathname, format='png')
    plt.close(fig)

def draw_2d_skeletons(frame, kpts_2d, color=(255, 0, 0)):
    radius = 5
    scale = 1
    connections_2d = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [18, 19], [19, 20], [20, 21], [22, 23], [23, 24], [24, 25], [26, 27], [27, 28], [28, 29], [30, 31], [31, 32], [32, 33], [34, 35], [35, 36], [36, 37], [38, 39], [39, 40], [40, 41]]
    
    for sk_id, sk in enumerate(connections_2d):
        if sk[0] > 16 or sk[1] > 16:continue
        pos1 = (int(kpts_2d[sk[0], 0]*scale), int(kpts_2d[sk[0], 1]*scale))
        pos2 = (int(kpts_2d[sk[1], 0]*scale), int(kpts_2d[sk[1], 1]*scale))
        cv2.line(frame, pos1, pos2, color, thickness=radius)
        
    return frame

def cross_video_alignment(template_name, candidate_name, task_path, scalar_parameters, pose_embedding_folder, pose_folder, normal_folder, device, pose_win_sec=0.3):
    template_embedding_file = os.path.join(pose_embedding_folder, f"{template_name}_embeddings.pkl")
    template_pose_file = os.path.join(pose_folder, f"{template_name}.pkl")
    template_video_file = os.path.join(normal_folder, f"{template_name}_normal.mp4")

    candidate_embedding_file = os.path.join(pose_embedding_folder, f"{candidate_name}_embeddings.pkl")
    candidate_pose_file = os.path.join(pose_folder, f"{candidate_name}.pkl")
    candidate_video_file = os.path.join(normal_folder, f"{candidate_name}_normal.mp4")

    # get the meta data of video file
    template_vid = mmcv.VideoReader(template_video_file)
    candidate_vid = mmcv.VideoReader(candidate_video_file)

    SHORT_SIDE = 480
    with open(template_pose_file, 'rb') as kp:
        pose_results = pickle.load(kp)    
    new_w, new_h = mmcv.rescale_size((template_vid.width, template_vid.height), (SHORT_SIDE, np.Inf))

    with open(template_embedding_file,'rb') as f:
        embeddings_dict = pickle.load(f)
    seq_embedding_x = extract_embedding_temporal_feature(embeddings_dict, track_id=0, win_size_sec=pose_win_sec, stride_sec=pose_win_sec/2)
    seq_embedding_x = torch.Tensor(seq_embedding_x).to(device)

    with open(candidate_embedding_file,'rb') as f:
        embeddings_dict = pickle.load(f)
    seq_embedding_y = extract_embedding_temporal_feature(embeddings_dict, track_id=0, win_size_sec=pose_win_sec, stride_sec=pose_win_sec/2)
    seq_embedding_y = torch.Tensor(seq_embedding_y).to(device)

    CSP = CrossSimilarityProbDistance(device=device, raw_a=scalar_parameters["raw_a"], raw_b=scalar_parameters["raw_b"],
                                      smoothing=1e-8, path=task_path)
    distances_mat = CSP(seq_embedding_x, seq_embedding_y)
    distances_np = distances_mat.detach().cpu().numpy()

    # # Apply temporal averaging
    # kernel = np.ones(7) / 7  # Uniform kernel as an example
    # rate = 3
    # distances_np = temporal_averaging(distances_np, kernel, rate)    

    d, dtw_path = dtw(distances_np)
    visualize_dtw_path(distances_np, np.array(dtw_path), template_name, candidate_name, task_path)
    return distances_np, dtw_path

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
    stride = max(1, int(vid.fps * pose_win_sec / 2))
    clips = []
    st = index * stride
    ed = min(vid.frame_cnt, (index+1) * stride)
    for i in range(st, ed):
        frame1 = vid[i]
        frame1 = mmcv.imresize(frame1, (new_w, new_h))         
        if poses[i]:
            most_salient_person_1 = max(poses[i], key=get_saliency)
            most_salient_person_1['keypoints'] = most_salient_person_1['keypoints_2d'][:17]
            # draw_pose_frame_1 = vis_pose_result(pose_model, frame1, [most_salient_person_1])
            draw_pose_frame_1 = draw_2d_skeletons(frame1, most_salient_person_1['keypoints'])
            frame1 = draw_pose_frame_1[:,:,::-1]
        frame1 = mmcv.imresize(frame1, (new_w//2, new_h//2))
        clips.append(frame1)
    # clips = np.stack(clips)
    return clips

def generate_dtw_alignment_video(template_name, candidate_name, distance_np, dtw_path, task_path, pose_folder, normal_folder, pose_win_sec=0.3):
    template_pose_file = os.path.join(pose_folder, f"{template_name}.pkl")
    template_video_file = os.path.join(normal_folder, f"{template_name}_normal.mp4")

    candidate_pose_file = os.path.join(pose_folder, f"{candidate_name}.pkl")
    candidate_video_file = os.path.join(normal_folder, f"{candidate_name}_normal.mp4")

    # get the meta data of video file
    template_vid = mmcv.VideoReader(template_video_file)
    candidate_vid = mmcv.VideoReader(candidate_video_file)

    SHORT_SIDE = 480
    with open(template_pose_file, 'rb') as kp:
        template_poses = pickle.load(kp)    
    with open(candidate_pose_file, 'rb') as kp:
        candidate_poses = pickle.load(kp)            
    # new_w, new_h = mmcv.rescale_size((template_vid.width, template_vid.height), (SHORT_SIDE, np.Inf))
    new_w, new_h = template_vid.width, template_vid.height

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

    writer = imageio.get_writer(f"{task_path}/{template_name}_{candidate_name}_alignment.mp4", fps=int(min(template_vid.fps, candidate_vid.fps)))
    pre_frames_1, pre_frames_2 = [], []

    for i, (index1, index2) in tqdm(enumerate(dtw_path)):
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

def generate_aligned_unaligned_path(template_name, candidate_name, distance_matrix, dtw_path, pose_embedding_folder, pose_win_sec=0.3):
    """
    Categorize the DTW path into aligned and misaligned segments.

    Args:
    dtw_path (list of tuples): The DTW path as a list of index pairs.

    Returns:
    dict: A dictionary with two keys 'aligned' and 'misaligned', each containing a list of segments as well as a distance value.
    """
    def fetch_segment_frames_id(fps, frame_cnt, index):
        stride = max(1, int(fps * pose_win_sec / 2))
        st = index * stride
        ed = min(frame_cnt, (index+1) * stride)
        return np.array(list(range(st, ed))).astype(int)
    
    template_embedding_file = os.path.join(pose_embedding_folder, f"{template_name}_embeddings.pkl")
    candidate_embedding_file = os.path.join(pose_embedding_folder, f"{candidate_name}_embeddings.pkl")
    with open(template_embedding_file,'rb') as f:
        embeddings_dict = pickle.load(f)
    fps1 = embeddings_dict['meta_info']['fps']
    emb_length1 = len(embeddings_dict['embeddings'])
    with open(candidate_embedding_file,'rb') as f:
        embeddings_dict = pickle.load(f)
    fps2 = embeddings_dict['meta_info']['fps']
    emb_length2 = len(embeddings_dict['embeddings'])

    result_path = []
    pre_frames_1, pre_frames_2 = None, None
    for i, (index1, index2) in tqdm(enumerate(dtw_path)):
        # fetch the index1 / index2 segment frames from both videos
        frames_1 = fetch_segment_frames_id(fps1, emb_length1, index1)
        frames_2 = fetch_segment_frames_id(fps2, emb_length2, index2)
        num_f = min(len(frames_1), len(frames_2))
        # adjust the different frame rate
        sampling_idx = np.linspace(0, max(len(frames_1), len(frames_2))-1, num_f).astype(int)
        if len(frames_1) > num_f:
            frames_1 = frames_1[sampling_idx]
        if len(frames_2) > num_f:
            frames_2 = frames_2[sampling_idx]
        d = float(distance_matrix[index1, index2])

        # check constant index
        unaligned = False
        if i+1 < len(dtw_path) and index1 == dtw_path[i+1][0]:
            unaligned = True
            if pre_frames_1 is not None:
                frames_1 = [ pre_frames_1[-1] for _ in range(num_f)]
        if i+1 < len(dtw_path) and index2 == dtw_path[i+1][1]:
            unaligned = True
            if pre_frames_2 is not None:
                frames_2 = [ pre_frames_2[-1] for _ in range(num_f)]
        
        if unaligned:
            frames_pair = [ (int(frames_1[i]), int(frames_2[i]), d, False) for i in range(num_f)]
        else:
            frames_pair = [ (int(frames_1[i]), int(frames_2[i]), d, True) for i in range(num_f)]

        pre_frames_1 = frames_1
        pre_frames_2 = frames_2

        result_path.extend(frames_pair)

    return result_path

if __name__ == '__main__':
    args = parse_args()

    if args.resume_ckpt is None:
        raise ValueError('Should provide a valid checkpoint!') 
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # We only need sigmoid parameters when a related distance kernel is used.
    raw_a = torch.nn.Parameter(torch.tensor(constants.sigmoid_raw_a_initial, device=device))
    raw_b = torch.nn.Parameter(torch.tensor(constants.sigmoid_b_initial, device=device))
    scalar_parameters = dict(raw_a=raw_a, raw_b=raw_b, epoch=0, epoch_step=0)
    keypoint_profile_2d = keypoint_profiles.create_keypoint_profile("LEGACY_2DCOCO13")

    """ Prepare pose embedder 
    """
    embedder_fn = models.get_embedder(
        base_model_type=constants.BASE_MODEL_TYPE_SIMPLE,
        embedding_type=constants.EMBEDDING_TYPE_GAUSSIAN,
        num_embedding_components=constants.num_embedding_components,
        embedding_size=constants.embedding_size,
        num_embedding_samples=constants.num_embedding_samples,
        weight_max_norm=0.0,
        feature_dim=3*keypoint_profile_2d.keypoint_num,
        seed=0)
    
    embedder_fn.keywords['base_model_fn'].to(device)
    embedder_fn.keywords['base_model_fn'].eval()

    models.load_training_state(embedder_fn=embedder_fn, optimizer=None, scalar_parameters=scalar_parameters, ckpt_resume_path=args.resume_ckpt)        
    raw_a = scalar_parameters['raw_a']
    raw_b = scalar_parameters['raw_b']
    print(raw_a, raw_b)      

    """ Prepare pose estimator 
    """
    pose_model = init_pose_model('pose_embedding/tools/demo_config/hrnet_w32_coco_256x192.py', 
                                 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
                                device)   
    det_model = init_detector('pose_embedding/tools/demo_config/faster_rcnn_r50_fpn_2x_coco.py', 
                              'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                                'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                                'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth', 
                                device)
    assert det_model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert det_model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'

    task_name = args.input.split('/')[-1]
    task_path = os.path.join('data/output/vis', task_name)
    mmcv.mkdir_or_exist(args.pose_folder)
    mmcv.mkdir_or_exist(task_path)

    template_name  = "baseline" # "baseline-IMG_0032" # "baseline"
    candidate_name = "candidate-bad" # "candidate-IMG_0033" # "candidate-good"
    pose_win_sec = 0.3  #pose_win_sec=0.3
    distance_np, dtw_path = cross_video_alignment(template_name, candidate_name, task_path, pose_win_sec=pose_win_sec)
    categorized_path = generate_aligned_unaligned_path(template_name, candidate_name, distance_np, dtw_path, pose_win_sec=pose_win_sec)
    # export the categorized dtw path
    with open(os.path.join(task_path, f"{template_name}_{candidate_name}-dtw_path.json"),"w") as f:
        json.dump(categorized_path, f)
    generate_dtw_alignment_video(template_name, candidate_name, distance_np, np.array(dtw_path), task_path, pose_win_sec=pose_win_sec)