import pickle
import torch
import torch.nn.functional as F
import numpy as np
import time
import glob
import os
import mmcv
from pose_embedding.common.utils.dtw import SelfSimilarityProbDistance
from pose_embedding.common import constants

def pad_array(X, N):
    M = X.shape[0]
    if N < M:
        raise ValueError("N should be greater than or equal to M")
    padding = np.tile(X[-1,:][np.newaxis, :], (N-M, 1))
    return np.concatenate([X, padding], axis=0)

dataset_path = '/home/jun/tumeke/data/gpu-compute-test-data' #"/home/jun/data/repcount-video-collection" # "output/fullbody-occlusion-1018-0.5-0.2/embedding-ep10-custom-bg" # "-ep10-custom-upper/"
embedding_path = os.path.join(dataset_path, "poses/embedding")
embedding_file_list = glob.glob(os.path.join(embedding_path, "*.pkl"))
feature_store_path = os.path.join(dataset_path, "feature-store")
mmcv.mkdir_or_exist(feature_store_path)
win_size_sec = 0.5
stride_sec   = 0.25
win_size_frames = 8  # For high fps, do subsampling. For low fps, padding the last frame

for candidate_embedding_file in embedding_file_list:
    with open(candidate_embedding_file,'rb') as f:
        embeddings = pickle.load(f)

        fps = embeddings['meta_info']['fps']
        stride = int(fps * stride_sec)
        # win_size_frames = 8  # For high fps, we use 8 frames per window
        cand_s, cand_e = 0, int(fps * win_size_sec)
        seq_embedding_list = []
        embedding_list = embeddings['embeddings']
        # t0 = time.time()
        while cand_e < len(embedding_list):
            sequence_b = embedding_list[cand_s:cand_e]
            sequence_b = np.stack([em[constants.KEY_EMBEDDING_SAMPLES] for em in sequence_b], axis=0)
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
        # print(f"Matched {candidate_vid.frame_cnt/candidate_vid.fps:.2f} seconds video cost {time.time() - t0: .4f} seconds")
        seq_embedding = np.stack(seq_embedding_list)
        name = candidate_embedding_file.split('/')[-1].split('.')[0]
        temporal_emb_file = os.path.join(feature_store_path, f"{name}.npy")
        # if seq_embedding.shape[0] > 255:
        print(f"{name}: temporal embedding length {seq_embedding.shape[0]}")
        np.save(temporal_emb_file, seq_embedding)
