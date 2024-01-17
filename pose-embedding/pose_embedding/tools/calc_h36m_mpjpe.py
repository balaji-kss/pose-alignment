import torch
import numpy as np
import time
import json

from pose_embedding.common import models, keypoint_utils, loss_utils, data_utils, constants, keypoint_profiles, distance_utils
from human36m.Human36M import Human36M

import matplotlib.pyplot as plt

def select_points(matrix, threshold):
    n = len(matrix)
    
    # Calculate minimum distances for each point and sort by them
    min_dists = [(i, min(matrix[i])) for i in range(n)]
    sorted_points = [i for i, dist in sorted(min_dists, key=lambda x: -x[1])]
    
    S = [sorted_points[0]]  # Start with the point with the maximum minimum distance
    
    for i in sorted_points[1:]:
        add_point = True
        for s in S:
            if matrix[i][s] <= threshold:
                add_point = False
                break
        if add_point:
            S.append(i)
    S.sort()
    return S



h36m = Human36M('test')
h36m.load_keypoint3d()

# keypoint_profile_3d = keypoint_profiles.create_keypoint_profile("EXTRACTED_3DH36M17")
distance_fn = keypoint_utils.compute_procrustes_aligned_mpjpes
min_negative_dist = 0.02
select_3d_dict = {}

i = 0
for key, value in h36m.joints3d.items():
    keypoints_3d = torch.stack(value)
    t0 = time.time()
    dist_mat = distance_utils.compute_distance_matrix(keypoints_3d, keypoints_3d, distance_fn)
    mask_mat = dist_mat >= min_negative_dist
    print(time.time() - t0)

    subset = select_points(dist_mat, min_negative_dist)
    select_3d_dict[key] = subset

    i += 1
    # # Adjust figure size
    # plt.figure(figsize=(10,10))
    # # Plot the tensor
    # plt.imshow(mask_mat.numpy(), cmap='gray', aspect='auto')
    # plt.colorbar()
    # plt.title('Boolean Tensor Visualization')
    # plt.savefig(f"data/{key}.png", dpi=300)  # adjust dpi as needed
    # plt.close()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

with open("data/h36m_selected_3d_frames.json", 'w') as out_file:
    json_string = json.dumps(select_3d_dict, indent=2)
    out_file.write(json_string)