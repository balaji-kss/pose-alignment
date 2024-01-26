import pickle
import torch
import torch.nn.functional as F
import numpy as np
import time
import glob
import os
from pose_embedding.common.utils.dtw import SelfSimilarityProbDistance


embedding_path = "output/fullbody-occlusion-1018-0.5-0.2/embedding-ep10-repcount" # "-ep10-custom-upper/" repcount
embedding_file_list = glob.glob(os.path.join(embedding_path, "*.pkl"))
device = "cuda:0" # 

start, end = 0, 512 # 256 grids equal 256*0.25 ~ 1 min 
# device = "cpu" # 
for candidate_embedding_file in embedding_file_list:
    with open(candidate_embedding_file,'rb') as f:
        embeddings = pickle.load(f)

    raw_a = torch.Tensor(embeddings['meta_info']['raw_a']).to(device)
    raw_b = torch.Tensor(embeddings['meta_info']['raw_b']).to(device)
    sims_prob = SelfSimilarityProbDistance(device=device,raw_a=raw_a, raw_b=raw_b,
                                           path="/".join(candidate_embedding_file.split('/')[:-1]))
    t0=time.time()
    tsm_matrix = sims_prob.test_by_second(embeddings, win_size=0.5, stride=0.25, start=start, end=end, 
                                name=candidate_embedding_file.split('/')[-1],
                                device=device)
    print(f"{end-start}*{end-start} TSM : {time.time()-t0:.2f} seconds (using {device}).")
