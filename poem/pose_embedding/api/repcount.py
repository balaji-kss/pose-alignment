import torch
import numpy as np
import time
import pickle
import functools

from pose_embedding.common import loss_utils, constants
from pose_embedding.common.utils.peaks import valleys_dynamic_weighting_thresholding_nan, \
                                                height_based_detection_without_prominence_nan, \
                                                    plot_and_save_valleys
from pose_embedding.common.utils.dtw import temporal_averaging, compute_pairwise_distances, dtw_with_precomputed_distances


""" posture_repetition_count is used for selected posture repetition counting. 
    Input:
        embedding_file: Embedding pkl file downloaded from S3 "embedding.pkl"
        posture_start: Selected posture start frame
        posture_end: Selected posture end frame
        track_id: Which person we are counting, default 0
        stride_sec: Sliding window stride, unit in second, default 0.5 seconds.
        max_win: Max sliding window frame size, default 16 frames, if window size is larger than max_win, we will do subsample
        dtw: Wheather or not use DTW for better temporal comparison, default True
        save_plot: Save the repetition detection plot in the same folder of the embedding file, default False.
    Return:
        If an error occurred, return None.
        Otherwise, return a list of detected repetitions (in frames).
        " [[132, 157], [252, 277], [312, 337], [408, 433], [456, 481], [504, 529], [552, 577], [636, 661]] "
"""
def posture_repetition_count(embedding_file, posture_start, posture_end, track_id=0,
                             stride_sec=0.5, max_win=16, dtw=True, save_plot=False):
    def check_all_sublists_empty(lst):
        return all([not sublist for sublist in lst])    
    
    def padding_sequence_to_np(emb_list, track_id, max_win):
        sequence = []
        for em in emb_list:
            for person_em in em:
                if person_em['track_id'] == track_id:
                    sequence.append(person_em[constants.KEY_EMBEDDING_SAMPLES])
        # It is possible sequence is empty due to camera view change
        if len(sequence) == 0:
            return None
        
        while len(sequence) < len(emb_list):
            sequence.append(np.copy(sequence[-1]))

        sequence = np.stack(sequence, axis=0)
        sequence = sequence.reshape(sequence.shape[0], -1)
        if sequence.shape[0] > max_win:
            sampling_idx = np.linspace(0, sequence.shape[0]-1, max_win).astype(int)
            sequence = sequence[sampling_idx]

        return sequence

    try:
        with open(embedding_file, "rb") as f:
            embeddings = pickle.load(f)
    except Exception as e:
        print(f"Try to open the embedding file -- an error occurred: {e}")
        return None
    meta_info = embeddings['meta_info']
    embedding_list = embeddings['embeddings']

    if posture_end <= posture_start or posture_start < 0 or posture_end < 0 or \
        posture_start > len(embedding_list) or posture_end > len(embedding_list):
        print(f"Posture range: [{posture_start}, {posture_end}] should be a valid range. (Total frames: {len(embedding_list)})")
        return None

    posture_start_sec = posture_start / meta_info['fps']
    posture_end_sec = posture_end / meta_info['fps']

    pose_win_sec = posture_end_sec - posture_start_sec
    template_s = posture_start # int(meta_info['fps'] * posture_start_sec)
    template_e = posture_end   # int(meta_info['fps'] * posture_end_sec)
    pose_win_template = template_e - template_s # template posture window size
    max_win = min(pose_win_template, max_win)
    stride_sec = min(stride_sec, pose_win_sec/2)
    stride = int(stride_sec * meta_info['fps']) # stride frames


    # For template posture, if there is non-person frame, we will 
    if check_all_sublists_empty(embedding_list[template_s:template_e]):
        print(f"From {posture_start_sec} to {posture_end_sec}, nobody was detected!")
        return None

    # print(f"embeddings: {len(embedding_list)}, fps: {meta_info['fps']:.2f}, raw_a: {meta_info['raw_a']:.2f}, raw_b: {meta_info['raw_b']:.2f}")

    sigmoid_a, sigmoid_b = loss_utils.get_sigmoid_parameters( 
        raw_a=torch.tensor(meta_info["raw_a"]),
        raw_b=torch.tensor(meta_info["raw_b"]),
        a_range=(None, constants.sigmoid_a_max))  

    probabilistic_distance = functools.partial(
            loss_utils.probabilistic_distance,
            sigmoid_a=float(sigmoid_a),
            sigmoid_b=float(sigmoid_b)
            )
    sequence_a = padding_sequence_to_np(embedding_list[template_s:template_e], track_id=track_id, max_win=max_win)
    if sequence_a is None:
        print(f"Posture of track_id={track_id} not exists between [{posture_start_sec}--{posture_end_sec}].")
        return None

    # # test non-person detection empty edge cases
    # if False:
    #     empty_idx = np.random.randint(0, len(embedding_list))
    #     empty_list = [[] for i in range(120)]
    #     new_list = []
    #     new_list.extend(empty_list)
    #     new_list.extend(embedding_list[:empty_idx])
    #     new_list.extend(empty_list)
    #     empty_idx2 = np.random.randint(empty_idx+1, len(embedding_list))
    #     new_list.extend(embedding_list[empty_idx:empty_idx2])
    #     new_list.extend(empty_list)
    #     new_list.extend(embedding_list[empty_idx2:])    
    #     new_list.extend(empty_list)
    #     embedding_list = new_list

    # Using sliding window to compare
    cand_s, cand_e = 0, pose_win_template
    scores = []
    t_dtw = 0
    t_total_dist_mat = 0
    while cand_e < len(embedding_list):
        if not check_all_sublists_empty(embedding_list[cand_s:cand_e]):
            sequence_b = padding_sequence_to_np(embedding_list[cand_s:cand_e], track_id=track_id, max_win=max_win)
            if sequence_b is None:
                scores.append([cand_s, cand_e, np.nan])
            else:
                t1 = time.time()
                distance_matrix = compute_pairwise_distances(sequence_a, sequence_b, probabilistic_distance)
                # Apply temporal averaging
                kernel = np.ones(7) / 7  # Uniform kernel as an example
                rate = 3
                averaged_distances = temporal_averaging(distance_matrix, kernel, rate)
                t_dist_mat = time.time() - t1
                if dtw:
                    # Compute DTW distance using the averaged distances
                    distance, path = dtw_with_precomputed_distances(averaged_distances) #  distance_matrix
                else:
                    distance = np.mean(distance_matrix)
                t_dtw += time.time() - t1 - t_dist_mat
                t_total_dist_mat += t_dist_mat

                # print(f"[{posture_start_sec: .1f}, {posture_end_sec: .1f}]-- "
                #     f"[{cand_s/meta_info['fps']: .1f}, {cand_e/meta_info['fps']: .1f}] DTW distance: {distance:.2f}. "
                #     f"dist_matrix time {t_dist_mat:.3e}, dtw time {t_dtw:.3e}")    
                scores.append([cand_s, cand_e, distance])
        else:
            scores.append([cand_s, cand_e, np.nan])
        cand_s += stride
        cand_e += stride

    t0 = time.time()
    # Detect valleys using prominence-based approach
    posture_scores = np.array([s[-1] for s in scores])
    posture_start_array_sec = np.array([s[0]/meta_info['fps'] for s in scores])

    valleys_prominence = valleys_dynamic_weighting_thresholding_nan(posture_scores, window_size=20)
    
    # Detect valleys using height-based approach without prominence with further adjusted height factor
    valleys_height = height_based_detection_without_prominence_nan(posture_scores, window_size=20, height_factor=0.7)
    
    # Intersect the results
    intersected_valleys = list(set(valleys_prominence) & set(valleys_height))
    intersected_valleys = sorted(intersected_valleys)

    peak_indices = np.array(intersected_valleys)
    t_findpeaks = time.time() - t0

    if save_plot:
        valley_plot_name = '.'.join(embedding_file.split('.')[:-1]) + f'_{posture_start_sec:.1f}-{posture_end_sec:.1f}_dtw_{dtw}.png'
        title_prefix = f"{posture_start_sec:.1f}-{posture_end_sec:.1f}: "
        plot_and_save_valleys(posture_scores, posture_start_array_sec, peak_indices, stride, meta_info['fps'], title_prefix, filename=valley_plot_name)
        # print(f"DTW {t_dtw:.4f} sec, fps={len(embedding_list)/t_dtw: .2f} \n"
        #     f"Pair-wise distance matrix {t_total_dist_mat:.4f} sec, fps={len(embedding_list)/t_total_dist_mat: .2f}\n"
        #     f"Find peaks {t_findpeaks:.4f} sec, fps={len(embedding_list)/t_findpeaks:.2f}")
    repetition_postures = [scores[idx][:] for idx in intersected_valleys]
    repetition_postures.sort(key=lambda x: x[2], reverse=False)
    return repetition_postures
