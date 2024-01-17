# def pick_equidistant_elements(A, m):
#     n = len(A)
#     if m > n:
#         raise ValueError("m cannot be greater than the number of elements in the list")

#     # Calculate the step size
#     step = n // m

#     # Select elements
#     selected_elements = [A[i] for i in range(0, n, step)]

#     # Adjust the length of the result if it's longer than m
#     return selected_elements[:m]

# # Example usage
# A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# m = 6
# print(pick_equidistant_elements(A, m))

# import numpy as np
# idx = np.round(np.linspace(0, len(A) - 1, m)).astype(int)
# print(np.array(A)[idx])

# import pickle
# # inp_pkl = "/home/tumeke-balaji/Downloads/baseline.pkl"
# inp_pkl = "/home/tumeke-balaji/Documents/results/delta/input_videos/Removing_Item_from_Bottom_of_Cart/poses/joints/baseline.pkl"
# with open(inp_pkl, "rb") as f:
#     poses = pickle.load(f)

# print('poses ', poses)

import json
json_path = '/home/tumeke-balaji/Documents/results/pose-estimation-3d/input_videos/results_vid1.json'
json_data = json.load(open(json_path))
skeleton = json_data['meta_info']['skeleton_links']
skeleton_link_colors = json_data['meta_info']["skeleton_link_colors"]["__ndarray__"]
kpt_colors = json_data['meta_info']["keypoint_colors"]["__ndarray__"]
print('skeleton ', skeleton)