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
    
def is_missed_step(path_ids, sidx):

    if path_ids[sidx][1] == path_ids[sidx + 1][1]:
        return False

    return True

def find_missing_segments(segments, start, end):
    # Sort segments by their starting values
    segments = sorted(segments, key=lambda x: x[0])
    
    # Initialize the list of missing segments
    missing_segments = []
    
    # Check for a gap before the first segment
    if segments[0][0] > start:
        missing_segments.append([start, segments[0][0] - 1])
    
    # Iterate through the segments to find gaps
    for i in range(1, len(segments)):
        prev_end = segments[i-1][1]
        curr_start = segments[i][0]
        
        # If there is a gap between the current and previous segment, add it to the list
        if prev_end + 1 < curr_start:
            missing_segments.append([prev_end + 1, curr_start - 1])
    
    # Check for a gap after the last segment
    if segments[-1][1] < end:
        missing_segments.append([segments[-1][1] + 1, end])
    
    return missing_segments

def gen_ids(base_segment_lst, cand_segment_lst, isalign):

    gsbase_inds = []
    gscand_inds = []

    for bsegment, csegment in zip(base_segment_lst, cand_segment_lst):
        base_len = bsegment[1] - bsegment[0] + 1
        cand_len = csegment[1] - csegment[0] + 1
        
        if base_len == 1:min_len = cand_len
        elif cand_len == 1:min_len = base_len
        else:min_len = min(base_len, cand_len)

        base_inds = list(range(bsegment[0], bsegment[1] + 1))
        cand_inds = list(range(csegment[0], csegment[1] + 1))

        sbase_inds = utils.pick_equidistant_elements(base_inds, min_len).tolist()
        scand_inds = utils.pick_equidistant_elements(cand_inds, min_len).tolist()

        gsbase_inds += sbase_inds
        gscand_inds += scand_inds

    if isalign:
        vals = np.ones((len(gscand_inds), 1))
    else:
        vals = np.zeros((len(gscand_inds), 1))

    gsbase_inds = np.expand_dims(gsbase_inds, axis=1)
    gscand_inds = np.expand_dims(gscand_inds, axis=1)
    
    pair_lst = np.hstack((gsbase_inds, gscand_inds))
    pair_lst = np.hstack((pair_lst, vals)).astype('int')

    return pair_lst

def get_smooth_paths(path_ids, num_base_frames, num_cand_frames, thresh = 0.4):

    path_ids_np = np.array(path_ids)

    # get align and non align segments 
    segments = find_segments(path_ids_np[:, 2]) # start and end included

    # get indices of non align segments
    non_align_se = np.where(segments[:, 0] == 0)[0]
    base_non_align_seg = []
    cand_non_align_seg = []

    for na_idx in non_align_se:
        # ignore non-aligns at the start and end of video 
        # because the activity might not have started
        if na_idx == 0 or na_idx == len(segments) - 1:continue 
        sidx = segments[na_idx][1]
        eidx = segments[na_idx][2]
        len_seg = eidx - sidx
        if len_seg == 0: continue

        is_misstep = is_missed_step(path_ids, sidx)

        if is_misstep:
            if len_seg > thresh * num_base_frames:
                
                base_non_align_seg.append([path_ids[sidx][0], path_ids[eidx][0]])
                cand_non_align_seg.append([path_ids[sidx][1], path_ids[eidx][1]])
        else:
            if len_seg > thresh * num_cand_frames:
                
                base_non_align_seg.append([path_ids[sidx][0], path_ids[eidx][0]])
                cand_non_align_seg.append([path_ids[sidx][1], path_ids[eidx][1]])

    if len(base_non_align_seg) == 0:
        base_align_seg = [[0, num_base_frames - 1]]
        cand_align_seg = [[0, num_cand_frames - 1]]
    else:
        base_align_seg = find_missing_segments(base_non_align_seg, 0, num_base_frames - 1)
        cand_align_seg = find_missing_segments(cand_non_align_seg, 0, num_cand_frames - 1)
    
    align_pairs = gen_ids(base_align_seg, cand_align_seg, isalign=True)
    non_align_pairs = gen_ids(base_non_align_seg, cand_non_align_seg, isalign=False)

    print('align_pairs shape ', align_pairs.shape)
    print('non_align_pairs shape ', non_align_pairs.shape)

    pairs = np.concatenate((align_pairs, non_align_pairs), axis = 0)

    sorted_indices = np.lexsort((pairs[:, 1], pairs[:, 0]))
    sorted_pairs = pairs[sorted_indices]
    
    return sorted_pairs

    