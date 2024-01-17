import time
import pickle

from pose_embedding.api.repcount import posture_repetition_count

# embedding_file = "data/gpu_compute_output/sura_twist/embedding_sura_twist_2.pkl"
# embedding_file = "data/gpu_compute_output/kontoor1/embedding_kontoor1.pkl"
# embedding_file = "/data/junhe/data/gpu-compute-output/box-lifting/full_embedding.pkl"
embedding_file = "/data/junhe/data/gpu-compute-output/kontoor2-6/full_embedding.pkl"
# embedding_file = "/data/junhe/data/gpu_compute_output/box-lifting/full_embedding.pkl"
# embedding_file = "data/gpu_compute_output/Line24_2_persons/embedding_line24_two_persons.pkl"
# embedding_file = "data/gpu_compute_output/dw_hand/embedding_meta.pkl"
# embedding_file = "data/gpu_compute_output/Line28_vertical/embedding_line28.pkl"


dtw=True
start_sec = 136
end_sec = 137
track_id = 0

t0 = time.time()

try:
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
except FileNotFoundError:
    print(f"{embedding_file} doest not exist ... ")

fps = embeddings['meta_info']['fps']
postures = posture_repetition_count(embedding_file, int(start_sec * fps), int(end_sec * fps), track_id=track_id, dtw=dtw, save_plot=True)

count_time=time.time()-t0
print(f"DTW={dtw}: {count_time:.3f} seconds, fps={len(embeddings['embeddings'])/count_time:.2f}")
print([int(start_sec * fps), int(end_sec * fps)], ": ", postures)