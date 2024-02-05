import time
import pickle
import numpy as np
import torch

from pose_embedding.api.repcount import posture_repetition_count
from pose_embedding.api.temporal_embedding_feature import extract_embedding_temporal_feature_rewrite, extract_embedding_temporal_feature
from pose_embedding.common.utils.convert_embedding import convert_embedding_fbf_pbp, convert_embedding_pbp_fbf
from pose_embedding.common.loss_utils_numpy import probabilistic_distance
from pose_embedding.common.loss_utils import probabilistic_distance_torch

# embedding_file = "data/gpu_compute_output/sura_twist/embedding_sura_twist_2.pkl"
# embedding_file = "data/gpu_compute_output/kontoor1/embedding_kontoor1.pkl"
# embedding_file = "/data/junhe/data/gpu-compute-output/box-lifting/full_embedding.pkl"
# embedding_file = "/data/junhe/data/gpu-compute-output/kontoor2-6/full_embedding.pkl"
# embedding_file = "/data/junhe/data/gpu_compute_output/box-lifting/full_embedding.pkl"
# embedding_file = "data/gpu_compute_output/Line24_2_persons/embedding_line24_two_persons.pkl"
# embedding_file = "data/gpu_compute_output/dw_hand/embedding_meta.pkl"
embedding_file1 = "./data/embedding-fps.pkl"
embedding_file2 = "/home/jun/data/delta-clips/closing-overhead-bin/poses/embedding/baseline_embeddings.pkl"



dtw=True
start_sec = 136
end_sec = 137
track_id = 0

t0 = time.time()

try:
    with open(embedding_file1, "rb") as f:
        embeddings_rw = pickle.load(f)
except FileNotFoundError:
    print(f"{embedding_file1} doest not exist ... ")

feat_rew = extract_embedding_temporal_feature_rewrite(embeddings_rw, track_id=0)

A = feat_rew[:20]
B = feat_rew[5:20+5]
sigmoid_a = embeddings_rw[0]["metadata"]["sigmoid_a"]
sigmoid_b = embeddings_rw[0]["metadata"]["sigmoid_b"]
dist1 = probabilistic_distance(A, B, sigmoid_a, sigmoid_b)
dist2 = probabilistic_distance_torch(torch.tensor(A), torch.tensor(B), embeddings_rw[0]["metadata"]["sigmoid_a"], embeddings_rw[0]["metadata"]["sigmoid_b"])

convert_embedding_pbp_fbf(embeddings_rw)

try:
    with open(embedding_file2, "rb") as f:
        embeddings = pickle.load(f)
except FileNotFoundError:
    print(f"{embedding_file2} doest not exist ... ")

embeddings_pbp = convert_embedding_fbf_pbp(embeddingData=embeddings)
feat = extract_embedding_temporal_feature(embeddings, track_id=0)
feat_2 = extract_embedding_temporal_feature_rewrite(embeddings_pbp, track_id=0)

print(f"|feat - feat2|={np.linalg.norm(feat - feat_2):.2e}")
# fps = embeddings['meta_info']['fps']
# postures = posture_repetition_count(embedding_file, int(start_sec * fps), int(end_sec * fps), track_id=track_id, dtw=dtw, save_plot=True)

# count_time=time.time()-t0
# print(f"DTW={dtw}: {count_time:.3f} seconds, fps={len(embeddings['embeddings'])/count_time:.2f}")
# print([int(start_sec * fps), int(end_sec * fps)], ": ", postures)