""" This file is depreciated because we use video-alignment-demo.py instead 
"""

import torch
from pose_embedding.common import visualization_utils
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

from sklearn.neighbors import BallTree
from fastdtw import fastdtw
from scipy.signal import find_peaks
import cv2
import functools
import imageio
import io

from pose_embedding.common import models, keypoint_utils, loss_utils, data_utils, constants, keypoint_profiles
from pose_embedding.common.utils.peaks import detect_local_valleys
from tqdm import tqdm
from mmdet.apis import init_detector
from mmpose.apis import init_pose_model, vis_pose_result   

def parse_args():
    parser = argparse.ArgumentParser(description='Video alignment using pose embedding.')
    parser.add_argument(
        '--input',
        default= '/data2/junhe/repcount-videos', #'/data2/junhe/hand_strain/tumeke_custom_hand_rep', # "/data2/junhe/ActionRecognition/data/lift-demo-export" ,# '/data2/junhe/hand_strain/tumeke_hand_rep' , # '/data2/junhe/hand_strain/tumeke_custom_hand_rep', '/data2/junhe/har_new_vids_clip' ,#
        help='video folder is $input/videos, pose folder is $input/poses/joints')  
    parser.add_argument(
        '--output',
        default= "output/upperbody", #
        help='repetition short videos')       
    parser.add_argument(
        '--resume-ckpt',
        default='checkpoints/upperbody-model-occlusion-occlusion0.20/training_model_8_0.pth',
        # default= './checkpoints/fullbody-model-occlusion/training_model_8_0.pth', # 8 None, # 'checkpoints/model_0905_no_dropout_epoch_3/training_model_2.pth', #
        help='resume training from previous checkpoint')   
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='CUDA device id')
    
    args = parser.parse_args()   
    args.video_folder = os.path.join(args.input, 'videos') 
    args.pose_folder = os.path.join(args.input, 'poses/joints') 
    args.pose_embedding_folder = os.path.join(args.input, 'poses/embedding-upperbody') 
    args.normal_folder = os.path.join(args.input, 'videos/normalized_video') 

    args.model_name = args.resume_ckpt.split('/')[-1]

    mmcv.mkdir_or_exist(args.pose_embedding_folder)
    mmcv.mkdir_or_exist(args.output)
    return args 
    
def show_pose(kp2d, keypoint_profile_2d):
    # visualization  
    kp2d = kp2d.unsqueeze(0).unsqueeze(0)
    fig = visualization_utils.draw_poses_2d(data_utils.flatten_first_dims(kp2d, 
                                                            num_last_dims_to_keep=2),
                            keypoint_profile_2d=keypoint_profile_2d,num_cols=1)
    fig.savefig(f"output/test_coco.png")
    plt.close(fig)


def get_saliency(person_dict):
    # Assuming the bounding box format is [x1, y1, x2, y2, score]
    bbox = person_dict['bbox']
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    area = w * h
    return area  

def save_embeddings(pose_folder, embedder):        
    pose_files = glob.glob(os.path.join(pose_folder, '*.pkl'))    
    for pose_f in pose_files:
        pose_name = pose_f.split('/')[-1].split('.')[0]
        embedding_file = os.path.join(args.pose_embedding_folder, f'{pose_name}_embeddings.pkl')
        if os.path.exists(embedding_file):
            print(f"Embedding vectors of {pose_f} exists")
            continue
        else:
            print(f"Extracting the embedding vectors of {pose_f}")
        embeddings = []
        with open(pose_f, "rb") as f:
            poses = pickle.load(f)
            for i, pose in tqdm(enumerate(poses)):
                # pose_key = i # f"{i:06d}.jpg"
                if len(pose) == 0:
                    coco_dict = keypoint_profiles.coco17_to_upperbody9(torch.zeros(17, 3))
                else:
                    most_salient_person = max(pose, key=get_saliency)
                    most_salient_person['keypoints'] = most_salient_person['keypoints'][:17]
                    coco_dict = keypoint_profiles.coco17_to_upperbody9(torch.tensor(most_salient_person['keypoints']))

                # visualization  
                # show_pose(coco_dict[constants.KEY_PREPROCESSED_KEYPOINTS_2D])

                coco_dict['model_inputs'] = coco_dict['model_inputs'].to(device)
                embedding_output, activations = embedder(coco_dict['model_inputs'])
                for k in embedding_output:
                    embedding_output[k] = embedding_output[k].detach().cpu().numpy()
                embeddings.append(embedding_output)
        print(f"Embeddings saved in {embedding_file}: {len(embeddings)}")
        mmcv.dump(embeddings, embedding_file)

#############################
""" Code for temporal similarity search
"""
def temporal_averaging(dist_matrix, kernel, rate):
    # Modify the kernel to account for the dilation rate
    if rate > 1:
        dilated_kernel = np.zeros(1 + (len(kernel)-1) * rate)
        dilated_kernel[::rate] = kernel
    else:
        dilated_kernel = kernel
    
    # Apply convolution
    return np.apply_along_axis(lambda m: np.convolve(m, dilated_kernel, mode='same'), axis=0, arr=dist_matrix)


def compute_pairwise_distances(sequence_a, sequence_b, distance_fn):
    distances = np.zeros((len(sequence_a), len(sequence_b)))
    for i in range(len(sequence_a)):
        for j in range(len(sequence_b)):
            distances[i, j] = distance_fn(sequence_a[i], sequence_b[j])
    return distances

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
    path = [(i, j)]
    
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
        path.append((i, j))
    path.reverse()
    return cost[N-1, M-1], path

def resize_with_aspect_ratio(image, target_width, target_height):
    img_height, img_width = image.shape[0], image.shape[1]
    
    # Calculate the aspect ratio
    aspect = img_width / img_height

    # Determine scaling based on the shorter side
    if target_height/img_height < target_width/img_width: # Height is the restricting dimension
        new_height = target_height
        new_width = int(aspect * new_height)
    else: # Width is the restricting dimension
        new_width = target_width
        new_height = int(new_width / aspect)
        
    # Resize with the new dimensions
    resized_image = cv2.resize(image, (new_width, new_height))
    
    return resized_image

def resize_img_with_padding(image, desired_width, desired_height):
    # Original dimensions
    original_width = image.shape[1]
    original_height = image.shape[0]

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate the new dimensions
    if (desired_width / aspect_ratio) <= desired_height:
        new_width = desired_width
        new_height = int(desired_width / aspect_ratio)
    else:
        new_height = desired_height
        new_width = int(desired_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Calculate padding
    pad_vert = desired_height - new_height
    pad_top = pad_vert // 2
    pad_bot = pad_vert - pad_top

    pad_horz = desired_width - new_width
    pad_left = pad_horz // 2
    pad_right = pad_horz - pad_left

    # Apply padding
    final_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return final_image

def adjust_template_length(template, candidate_length):
    output_frames = []

    if len(template) > candidate_length:
        section_size = len(template) / candidate_length

        # Generate linearly spaced indices to sample frames from template
        indices = np.linspace(0, len(template)-1, candidate_length).astype(int)
        output_frames = [template[idx] for idx in indices]
    else:
        repeat_factor = candidate_length // len(template)

        for frame in template:
            output_frames.extend([frame] * repeat_factor)

        # if there's any remainder, we just add additional frames from the beginning of the template
        remainder = candidate_length - len(output_frames)
        output_frames.extend(template[:remainder])

    return output_frames

# Rendering one frame of the demo video
def prepare_plot_figure(template_poses_clip, cur_frame, frame_scores, peak_indices, plot_info_dict):
    # frame_idx,  win_idx,  max_score, fps, new_h ):
    frame_idx       = plot_info_dict['frame_idx']
    max_score       = plot_info_dict['max_score'] 
    fps             = plot_info_dict['fps'] 
    new_h           = plot_info_dict['new_h'] 
    template_idx    = plot_info_dict['template_idx']
    rep_count       = plot_info_dict['rep_count']

    FONTFACE = cv2.FONT_HERSHEY_DUPLEX
    FONTSCALE = 2 # 0.75
    FONTCOLOR = (255, 255, 0)  # BGR, white
    FONTCOLOR2 = (255, 255, 255)  # BGR, white
    THICKNESS = 2
    LINETYPE = 1

    plt.figure(figsize=(4, 3))
    x_values_seconds = [i / fps for i in range(frame_idx+1)]
    plt.plot(x_values_seconds, frame_scores[:frame_idx+1])
    plt.scatter(x_values_seconds[frame_idx], frame_scores[frame_idx], color='red')  # Highlight current frame's score
    plt.xlabel('Frame Index')
    plt.ylabel('Score')
    plt.title('Similarity Scores (Lower is better)')
    plt.ylim(0, 1.5*max_score)  # Set y-axis limits
    # Convert the Matplotlib plot to an OpenCV image
    buf = io.BytesIO()  # Create an in-memory buffer
    plt.savefig(buf, format='png')  # Save the figure to the buffer
    buf.seek(0)  # Move the buffer position to the start
    plot_img = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    plot_img = cv2.imdecode(plot_img, cv2.IMREAD_COLOR)
    plt.close()

    side_by_side_frames = np.hstack([template_poses_clip[template_idx], cur_frame])     

    cv2.putText(side_by_side_frames, f'{rep_count}', \
                    (10, 80), FONTFACE, FONTSCALE,
                    FONTCOLOR2, THICKNESS, LINETYPE)
        
    # 1. Compute the required padding size
    padding_height = new_h - side_by_side_frames.shape[0]
    padding_width = side_by_side_frames.shape[1]  # since the width remains constant

    # 2. Create the white padding
    white_padding = np.ones((padding_height, padding_width, 3), dtype=np.uint8) * 255

    # 3. Overlay or replace the plot on the white padding
    resized_plot_img = resize_with_aspect_ratio(plot_img, padding_width, padding_height)
    start_y = (white_padding.shape[0] - resized_plot_img.shape[0]) // 2
    start_x = (white_padding.shape[1] - resized_plot_img.shape[1]) // 2
    end_y = start_y + resized_plot_img.shape[0]
    end_x = start_x + resized_plot_img.shape[1]
    
    white_padding[start_y:end_y, start_x:end_x] = resized_plot_img

    # 4. Vertically stack
    frame_assembled = np.vstack([side_by_side_frames, white_padding])
    return frame_assembled

""" template_clip: frames []
"""
def generate_matching_video_demo(demo_name, pose_file, video_file, 
                                 template_poses_clip, scores, alignment_stride, 
                                 slow_down_rate=3, demo_sec=-1):
    SHORT_SIDE = 480

    with open(pose_file, 'rb') as kp:
        pose_results = pickle.load(kp)    
    vid = mmcv.VideoReader(video_file)
    fps = vid.fps
    new_w, new_h = mmcv.rescale_size((vid.width, vid.height), (SHORT_SIDE, np.Inf))

    num_stride_in_a_clip = len(template_poses_clip)//alignment_stride

    # adjust the scale of template frames 
    template_poses_clip= [ resize_img_with_padding(f, new_w//2, new_h//2) for f in template_poses_clip]

    posture_scores = np.array([s[-1] for s in scores])

    # posture_score_mean = np.mean(posture_scores)
    # posture_score_stddev = np.std(posture_scores)
    # score_threshold = max(0 , posture_score_mean - 1*posture_score_stddev)

    # Since the sliding window is 0.5 sec with 50% overlaps, distance=3 means valley is at least 3 samples (0.75 sec) apart.
    # prominence=1 means we will suppress the surrounding neighbors height difference less than 1
    # peak_indices, _ = find_peaks(-posture_scores, distance=5, prominence=0.5, height=-score_threshold)
    peak_indices = detect_local_valleys(posture_scores, window_size=50, overlap=40, alpha=1)
    max_score = np.max(posture_scores)

    print(f"Exporting {demo_name}......\n Matching postures at {peak_indices}: {posture_scores[peak_indices]}")
    writer = imageio.get_writer(demo_name, fps=int(fps))
    frame_scores = []
    rep_count = 0
    template_idx = 0
    for frame_idx in tqdm(range(vid.frame_cnt)):
        if demo_sec > 0 and frame_idx // vid.fps > demo_sec:
            break
        cur_frame = vid[frame_idx]
        cur_frame = mmcv.imresize(cur_frame, (new_w, new_h))        
        if pose_results[frame_idx]:
            most_salient_person_1 = max(pose_results[frame_idx], key=get_saliency)
            most_salient_person_1['keypoints'] = most_salient_person_1['keypoints'][:17]
            draw_pose_frame_1 = vis_pose_result(pose_model, cur_frame, [most_salient_person_1])
            cur_frame = draw_pose_frame_1[:,:,::-1]
        cur_frame = mmcv.imresize(cur_frame, (new_w//2, new_h//2))

        # Draw the dynamic scores curve up to the current frame
        win_idx = frame_idx // alignment_stride  # Using stride to compute the alignment window index
        if win_idx >= len(scores):
            break
        if 0 < win_idx < len(scores) -1:
            frame_scores.append( (scores[win_idx][-1] + scores[win_idx-1][-1])/2 )
        else:
            frame_scores.append(scores[win_idx][-1])

        if any( (win_idx-n_stride) in peak_indices for n_stride in range(num_stride_in_a_clip)):
        # if win_idx in peak_indices or (win_idx-1) in peak_indices: # win_idx -1 is not suitable if we use smaller stride
            # show count digit at template_poses_clip
            if win_idx in peak_indices:
                rep_count = int(np.where(peak_indices==win_idx)[0])+1
            template_idx = min(template_idx+1, len(template_poses_clip)-1)
        else:
            template_idx = 0

        frame_assembled = prepare_plot_figure(template_poses_clip, cur_frame, frame_scores, peak_indices,
                                              dict(frame_idx=frame_idx, max_score=max_score, 
                                                   fps=fps, new_h=new_h,
                                                   template_idx=template_idx, rep_count=rep_count))
        # Slow motion for the detection 
        # if win_idx in peak_indices or (win_idx-1) in peak_indices:
        if any( (win_idx-n_stride) in peak_indices for n_stride in range(num_stride_in_a_clip)):
            for i in range(slow_down_rate):
                writer.append_data(frame_assembled)
        else:
            writer.append_data(frame_assembled)
    # plot the remaining part
    while frame_idx < vid.frame_cnt:
        cur_frame = vid[frame_idx]
        cur_frame = mmcv.imresize(cur_frame, (new_w, new_h))        
        if pose_results[frame_idx]:
            most_salient_person_1 = max(pose_results[frame_idx], key=get_saliency)
            most_salient_person_1['keypoints'] = most_salient_person_1['keypoints'][:17]
            draw_pose_frame_1 = vis_pose_result(pose_model, cur_frame, [most_salient_person_1])
            cur_frame = draw_pose_frame_1[:,:,::-1]
        cur_frame = mmcv.imresize(cur_frame, (new_w//2, new_h//2))
        frame_scores.append(scores[-1][-1])

        # if win_idx in peak_indices or (win_idx-1) in peak_indices:
        if any( (win_idx-n_stride) in peak_indices for n_stride in range(num_stride_in_a_clip)):
            # show count digit at template_poses_clip
            if win_idx in peak_indices:
                rep_count = int(np.where(peak_indices==win_idx)[0])+1
            template_idx = min(template_idx+1, len(template_poses_clip)-1)
        else:
            template_idx = 0

        frame_assembled = prepare_plot_figure(template_poses_clip, cur_frame, frame_scores, peak_indices,
                                              dict(frame_idx=frame_idx, win_idx=win_idx, 
                                                   max_score=max_score, fps=fps, new_h=new_h,
                                                   template_idx=template_idx, rep_count=rep_count))
        # Slow motion for the detection 
        # if win_idx in peak_indices or (win_idx-1) in peak_indices:
        if any( (win_idx-n_stride) in peak_indices for n_stride in range(num_stride_in_a_clip)):
            for i in range(slow_down_rate):
                writer.append_data(frame_assembled)
        else:
            writer.append_data(frame_assembled)

        frame_idx += 1

    writer.close()


def video_alignment_demo(
    # template_name="zach_bookshelf_loading_short-6s-7s", start_sec=6.0, end_sec=7.0, 
    # candidate_name="zach_bookshelf_loading_short-6s-7s", max_alignment_win=64, 
    # alignment_stride_sec=0.2, use_fastdtw = False
    template_name="Felled-2", start_sec=2*60+8, end_sec=2*60+9, # Felled-2.08-2.09 
    candidate_name="Felled-2", max_alignment_win=64, 
    alignment_stride_sec=0.5, use_fastdtw = False    
):
    # template_name="c983f7be-731c-4cb7-9687-5c98afe65e34", start_sec=24.2,
    #                        end_sec=0.5 , candidate_name="kontoor1", max_alignment_win=64, 
    #                        alignment_stride_sec=0.5, use_fastdtw = False

    if end_sec <= start_sec:
        print(f"{template_name} end_sec {end_sec} should larger than start_sec {start_sec}!")
        return 
    if candidate_name is None:
        candidate_name = template_name

    template_embedding_file = os.path.join(args.pose_embedding_folder, f"{template_name}_embeddings.pkl")
    template_pose_file = os.path.join(args.pose_folder, f"{template_name}.pkl")
    template_video_file = os.path.join(args.normal_folder, f"{template_name}_normal.mp4")

    candidate_embedding_file = os.path.join(args.pose_embedding_folder, f"{candidate_name}_embeddings.pkl")
    candidate_pose_file = os.path.join(args.pose_folder, f"{candidate_name}.pkl")
    candidate_video_file = os.path.join(args.normal_folder, f"{candidate_name}_normal.mp4")

    sigmoid_a, sigmoid_b = loss_utils.get_sigmoid_parameters( 
        raw_a=scalar_parameters["raw_a"],
        raw_b=scalar_parameters["raw_b"],
        a_range=(None, constants.sigmoid_a_max))  
    
    probabilistic_distance = functools.partial(
            loss_utils.probabilistic_distance,
            sigmoid_a=float(sigmoid_a),
            sigmoid_b=float(sigmoid_b)
            )
    # get the meta data of video file
    template_vid = mmcv.VideoReader(template_video_file)
    candidate_vid = mmcv.VideoReader(candidate_video_file)

    SHORT_SIDE = 480
    with open(template_pose_file, 'rb') as kp:
        pose_results = pickle.load(kp)    
    new_w, new_h = mmcv.rescale_size((template_vid.width, template_vid.height), (SHORT_SIDE, np.Inf))

    with open(template_embedding_file,'rb') as f:
        embeddings = pickle.load(f)
    # sequence_a is our template, sequence_b is the candidate
    pose_win_sec = end_sec - start_sec
    pose_win_template = int(pose_win_sec * template_vid.fps) # template posture window size
    pose_win_candidate = int(pose_win_sec * candidate_vid.fps) # candidate posture window size
    alignment_stride = int(alignment_stride_sec * candidate_vid.fps) # alignment stride frames

    alignment_win = min(pose_win_candidate, pose_win_template, max_alignment_win)   # used for down-sampling a sliding windows 

    template_s = int(template_vid.fps*start_sec)
    template_e = int(template_vid.fps*end_sec)

    if template_e > template_vid.frame_cnt:
        print(f"end_sec {end_sec} should less than the length of {template_name} {template_vid.frame_cnt/template_vid.fps: .2f}!")
        return

    sequence_a = embeddings[template_s:template_e]
    sequence_a = np.stack([em[constants.KEY_EMBEDDING_SAMPLES] for em in sequence_a], axis=0)
    sequence_a = sequence_a.reshape(sequence_a.shape[0], -1)

    # down-sampling the template if pose_win_template > alignment_win
    if pose_win_template > alignment_win:
        sampling_idx = np.linspace(0, pose_win_template-1, alignment_win).astype(int)
        sequence_a = sequence_a[sampling_idx]

    clips = []
    for i in range(template_s, template_e):
        frame1 = template_vid[i]
        frame1 = mmcv.imresize(frame1, (new_w, new_h))         
        if pose_results[i]:
            most_salient_person_1 = max(pose_results[i], key=get_saliency)
            most_salient_person_1['keypoints'] = most_salient_person_1['keypoints'][:17]
            draw_pose_frame_1 = vis_pose_result(pose_model, frame1, [most_salient_person_1])
            frame1 = draw_pose_frame_1[:,:,::-1]
        frame1 = mmcv.imresize(frame1, (new_w//2, new_h//2))
        clips.append(frame1)
    template_poses_clip = []
    template_poses_clip.extend(clips)
    template_poses_clip = adjust_template_length(template_poses_clip, pose_win_candidate)

    with open(candidate_embedding_file,'rb') as f:
        embeddings = pickle.load(f)

    cand_s, cand_e = 0, pose_win_candidate
    scores = []
    t0 = time.time()
    while cand_e < candidate_vid.frame_cnt:
        sequence_b = embeddings[cand_s:cand_e]
        sequence_b = np.stack([em[constants.KEY_EMBEDDING_SAMPLES] for em in sequence_b], axis=0)
        sequence_b = sequence_b.reshape(sequence_b.shape[0], -1)
        # down-sampling the template if pose_win_candidate > alignment_win
        if pose_win_candidate > alignment_win:
            sampling_idx = np.linspace(0, pose_win_candidate-1, alignment_win).astype(int)
            sequence_b = sequence_b[sampling_idx]        

        if use_fastdtw:
            distance, _ = fastdtw(sequence_a, sequence_b, dist=probabilistic_distance)
        else:
            distance_matrix = compute_pairwise_distances(sequence_a, sequence_b, probabilistic_distance)
            # Apply temporal averaging
            kernel = np.ones(7) / 7  # Uniform kernel as an example
            rate = 3
            averaged_distances = temporal_averaging(distance_matrix, kernel, rate)
            # Compute DTW distance using the averaged distances
            distance, path = dtw_with_precomputed_distances(averaged_distances) #  distance_matrix
        print(f"[{start_sec: .1f}, {end_sec: .1f}]-- "
              f"[{cand_s/candidate_vid.fps: .1f}, {cand_e/candidate_vid.fps: .1f}] DTW distance: {distance:.2f}. "
              f"fast_dtw ({use_fastdtw})")    
        scores.append([cand_s, cand_e, distance])
        cand_s += alignment_stride
        cand_e += alignment_stride
    print(f"Matched {candidate_vid.frame_cnt/candidate_vid.fps:.2f} seconds video cost {time.time() - t0: .4f} seconds")
    demo_name = f"{args.output}/{template_name}-{template_s/template_vid.fps:.1f}-{template_e/template_vid.fps:.1f}-{candidate_name}.mp4"

    score_name = f"{args.output}/{template_name}-{template_s/template_vid.fps:.1f}-" \
                    f"{template_e/template_vid.fps:.1f}-{alignment_stride_sec:.1f}-{candidate_name}.txt"
    posture_scores = np.array([s[-1] for s in scores])
    np.savetxt(score_name, posture_scores, '%.4f')

    generate_matching_video_demo(demo_name,candidate_pose_file, candidate_video_file, template_poses_clip, scores, alignment_stride)
        

if __name__ == '__main__':
    args = parse_args()

    if args.resume_ckpt is None:
        raise ValueError('Should provide a valid checkpoint!') 
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # We only need sigmoid parameters when a related distance kernel is used.
    raw_a = torch.nn.Parameter(torch.tensor(constants.sigmoid_raw_a_initial, device=device))
    raw_b = torch.nn.Parameter(torch.tensor(constants.sigmoid_b_initial, device=device))
    scalar_parameters = dict(raw_a=raw_a, raw_b=raw_b, epoch=0, epoch_step=0)
    keypoint_profile_2d = keypoint_profiles.create_keypoint_profile("UPPERBODY_9_2D")

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
    pose_model = init_pose_model('tools/demo_config/hrnet_w32_coco_256x192.py', 
                                 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
                                device)   
    det_model = init_detector('tools/demo_config/faster_rcnn_r50_fpn_2x_coco.py', 
                              'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                                'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                                'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth', 
                                device)
    assert det_model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert det_model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'

    mmcv.mkdir_or_exist(args.pose_folder)

    with torch.no_grad():
        save_embeddings(pose_folder=args.pose_folder, embedder=embedder_fn)
    
    video_alignment_demo()