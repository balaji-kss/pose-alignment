import json
import utils
import numpy as np

def find_segments(arr):
    # Start with an empty list to store the segments
    segments = []
    
    # Initialize start index and the initial value
    start = 0
    current_value = arr[0]
    
    # Iterate over the array
    for i in range(1, len(arr)):
        # Check if the value has changed
        if arr[i] != current_value:
            # If it has, record the segment
            segments.append((current_value, start, i - 1))
            # Update the start index and current value
            start = i
            current_value = arr[i]
    
    # Add the last segment
    segments.append((current_value, start, len(arr) - 1))
    
    return np.array(segments)
    
def check_valid(part):

    assert part.shape[0] == 3

    

def interpolate_aligns(pairs):

    print('pairs a/na ', pairs[:, 2])
    segments = find_segments(pairs[:, 2])
    print('segments ', segments)

    non_align_se = np.where(segments[:, 0] == 0)[0]
    print('non_align_se ', non_align_se)
    
    for i in range(1, len(non_align_se)):

        part = segments[non_align_se[i-1]:non_align_se[i] + 1]
        check_valid(part)

if __name__ == "__main__":

    input_json_path = '/home/tumeke-balaji/Documents/results/delta/input_videos/delta_data/Closing_Overhead_Bin/baseline_candidate-dtw_path.json'

    pairs = utils.pose_embed_video_align(input_json_path)
    pairs_np = np.array(pairs) 
    interpolate_aligns(pairs_np)
    