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

# import json
# json_path = '/home/tumeke-balaji/Documents/results/pose-estimation-3d/input_videos/results_vid1.json'
# json_data = json.load(open(json_path))
# skeleton = json_data['meta_info']['skeleton_links']
# skeleton_link_colors = json_data['meta_info']["skeleton_link_colors"]["__ndarray__"]
# kpt_colors = json_data['meta_info']["keypoint_colors"]["__ndarray__"]
# print('skeleton ', skeleton)

# import numpy as np

# def find_segments(arr):
#     # Start with an empty list to store the segments
#     segments = []
    
#     # Initialize start index and the initial value
#     start = 0
#     current_value = arr[0]
    
#     # Iterate over the array
#     for i in range(1, len(arr)):
#         # Check if the value has changed
#         if arr[i] != current_value:
#             # If it has, record the segment
#             segments.append((current_value, start, i - 1))
#             # Update the start index and current value
#             start = i
#             current_value = arr[i]
    
#     # Add the last segment
#     segments.append((current_value, start, len(arr) - 1))
    
#     return segments

# # Example usage
# arr = np.array([1,1,1,1,1,1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
# segments = find_segments(arr)
# print(segments)

import numpy as np

# Assuming 'matrix' is your NumPy matrix of shape (N, 22, 3) with non-negative values
# Example initialization for demonstration
N = 5  # Example size for N
matrix = np.random.randint(1, 10, (N, 22, 3))  # Generates a sample matrix with random values in the range [1, 9]

# Create a mask to ignore zeros
mask = matrix != 0

# Apply the mask, replace non-matching elements (zeros) with np.inf for min calculation and -np.inf for max calculation
masked_min_matrix = np.where(mask, matrix, np.inf)
masked_max_matrix = np.where(mask, matrix, -np.inf)

# Calculate the min and max across the first and second dimensions
min_values = masked_min_matrix.min(axis=(0, 1))
max_values = masked_max_matrix.max(axis=(0, 1))

print("Minimum values, excluding zeros:", min_values)
print("Maximum values, excluding zeros:", max_values)
