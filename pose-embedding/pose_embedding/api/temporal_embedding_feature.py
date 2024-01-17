import pickle
import numpy as np
from pose_embedding.common import constants

embedding_shape = [1, constants.num_embedding_components, constants.num_embedding_samples, constants.embedding_size]

def pad_array(X, N):
    M = X.shape[0]
    if N < M:
        raise ValueError("N should be greater than or equal to M")
    padding = np.tile(X[-1,:][np.newaxis, :], (N-M, 1))
    return np.concatenate([X, padding], axis=0)

def max_inclusive_empty(em_list):
    return max(em_list) if em_list else -1

def fill_missing_data(arrays):
    n = len(arrays)
    for i in range(n):
        if np.all(arrays[i] == 0):
            # Find nearest non-zero array
            j = 1
            while True:
                prev_idx = i - j
                next_idx = i + j

                prev_exists = prev_idx >= 0
                next_exists = next_idx < n

                if not prev_exists and not next_exists:
                    break

                if prev_exists and not np.all(arrays[prev_idx] == 0):
                    arrays[i] = arrays[prev_idx]
                    break
                elif next_exists and not np.all(arrays[next_idx] == 0):
                    arrays[i] = arrays[next_idx]
                    break

                j += 1
    return arrays

def extract_embedding_temporal_feature(embeddings, track_id=0, 
                                       win_size_sec=0.5, stride_sec=0.25, win_size_frames=8):
    fps = embeddings['meta_info']['fps']
    stride = max(1, int(fps * stride_sec))  # if stride_sec==0, simply use frame-wise
    cand_s, cand_e = 0, max(1, int(fps * win_size_sec)) # if win_size_sec==0, simply use frame-wise

    if cand_e - cand_s == 1:
        win_size_frames = 1

    seq_embedding_list = []
    embedding_list = embeddings['embeddings']
    # t0 = time.time()
    while cand_e < len(embedding_list):
        sequence_b = embedding_list[cand_s:cand_e]
        sequence_b_list = []
        for frame_ems in sequence_b:
            added = False
            for em in frame_ems:
                if em.get('track_id', -1)==track_id and constants.KEY_EMBEDDING_SAMPLES in em:
                    sequence_b_list.append(em[constants.KEY_EMBEDDING_SAMPLES])
                    added = True
                    break
            if not added:
                sequence_b_list.append(np.zeros(embedding_shape))
        # padding the missing pose (np.zeros) by using 
        # sequence_b_list = fill_missing_data(sequence_b_list)
        sequence_b = np.stack(sequence_b_list, axis=0)
        sequence_b = sequence_b.reshape(sequence_b.shape[0], -1)
        # down-sampling the template if pose_win_template > alignment_win
        if cand_e-cand_s > win_size_frames: # subsampling
            sampling_idx = np.linspace(0, cand_e-cand_s-1, win_size_frames).astype(int)
            sequence_b = sequence_b[sampling_idx]
        elif cand_e-cand_s < win_size_frames: # padding the last frame
            sequence_b = pad_array(sequence_b, win_size_frames)

        seq_embedding_list.append(sequence_b)

        cand_s += stride
        cand_e += stride
    seq_embedding = np.stack(seq_embedding_list)
    return seq_embedding
