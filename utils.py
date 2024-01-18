import mmcv
import numpy as np
import cv2
import pickle
import render 
import json

def number_frames():

    vname = 'baseline'
    act_name = 'Lower_Galley_Carrier'
    root_dir = '/home/tumeke-balaji/Documents/results/delta/joints/' + act_name + '/'
    input_video_path =  root_dir + vname + '/' + vname + '.mov'
    ext = input_video_path.rsplit('.', 1)[1]
    output_video_path = input_video_path.rsplit('.', 1)[0] + '_n.' + ext
    print('input_video_path ', input_video_path)
    print('output_video_path ', output_video_path)

    # Load video using mmcv
    video = mmcv.VideoReader(input_video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Define video writer
    video_writer = cv2.VideoWriter(output_video_path,
                                fourcc, video.fps, (video.width, video.height))
    
    for i, frame in enumerate(video):
        # Write frame number on top-left corner
        cv2.putText(frame, f'Frame: {i}', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write the frame to output video
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()

def draw_2d_skeletons_3d(frame, kpts_2d, color=(0, 0, 255), conf_thresh=0.35):
    radius = 10
    scale = 1

    for sk_id, sk in enumerate(connections):

        if kpts_2d[sk[0], 2] <= conf_thresh or kpts_2d[sk[1], 2] <= conf_thresh:continue

        pos1 = (int(kpts_2d[sk[0], 0]*scale), int(kpts_2d[sk[0], 1]*scale))
        pos2 = (int(kpts_2d[sk[1], 0]*scale), int(kpts_2d[sk[1], 1]*scale))
        cv2.line(frame, pos1, pos2, color, thickness=radius)

    return frame

def pick_equidistant_elements(A, m):
    
    idx = np.round(np.linspace(0, len(A) - 1, m)).astype(int)

    return np.array(A)[idx]
    
def normalize_segment(bsegment, csegment):

    sb, eb = bsegment
    sc, ec = csegment

    bids = list(range(sb, eb + 1))
    cids = list(range(sc, ec + 1))

    num_bids = len(bids)
    num_cids = len(cids)

    req_len = min(num_bids, num_cids) 

    if num_bids > num_cids:
        new_bids = pick_equidistant_elements(bids, req_len)
        new_cids = cids
    else:
        new_cids = pick_equidistant_elements(cids, req_len)
        new_bids = bids

    new_bids = np.expand_dims(new_bids, axis=1)
    new_cids = np.expand_dims(new_cids, axis=1)

    pair_lst = np.hstack((new_bids, new_cids))

    return pair_lst.tolist()

def get_segs(vname):

    if vname == "Stowing_carrier":
        v1_seg = [[60, 137], [140, 172], [175, 260], [280, 385]] # baseline
        v2_seg = [[100, 180], [225, 260], [260, 320], [360, 510]] # candidate
        return v1_seg, v2_seg
    
    if vname == "Lower_Galley_Carrier":
        v1_seg = [[31, 200], [240, 300]] # baseline
        v2_seg = [[31, 120], [135, 185]] # candidate
        return v1_seg, v2_seg
    
    if vname == "Pushing_cart":
        v1_seg = [[60, 120]] # baseline
        v2_seg = [[60, 100]] # candidate
        return v1_seg, v2_seg
    
    if vname == "Removing_Item_from_Bottom_of_Cart":
        v1_seg = [[5, 55], [55, 100], [125, 270]] # baseline
        v2_seg = [[57, 95], [110, 150], [155, 330]] # candidate
        return v1_seg, v2_seg
    
    if vname == "Serving_from_Basket":
        v1_seg = [[90, 155], [165, 185], [185, 240], [245, 300]] # baseline
        v2_seg = [[90, 120], [120, 145], [145, 193],[ 195, 220]] # candidate
        return v1_seg, v2_seg 

def manual_video_align(vname):

    v1_seg, v2_seg = get_segs(vname)

    fpairs = []
    for i in range(len(v1_seg)):

        pair_lst = normalize_segment(v1_seg[i], v2_seg[i])
        fpairs += pair_lst
    
    return fpairs

def pose_embed_video_align(align_path):

    with open(align_path, 'r') as file:
        pairs = json.load(file)

    pairs_np = np.array(pairs)

    # align_ids = pairs_np[:, -1]
    # align_indices = np.where(align_ids == 1)[0]

    # align_pairs = pairs_np[align_indices][:, :2].astype('int')

    align_pairs = pairs_np[:, [0, 1, 3]].astype('int')

    return align_pairs.tolist()

def video_align():

    name = 'Removing_Item_from_Bottom_of_Cart'
    video_path1 = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/baseline/baseline_n.mov"
    video_path2 = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/candidate/candidate_n.mov"
    align_path = "/home/tumeke-balaji/Documents/results/delta/input_videos/" + name + "/baseline_candidate-dtw_path.json"
    # Load videos
    video1 = mmcv.VideoReader(video_path1)
    video2 = mmcv.VideoReader(video_path2)

    path_pairs = pose_embed_video_align(align_path)

    # Iterate through both videos
    for t, (b, c) in enumerate(path_pairs):

        frame1 = video1[b]
        frame2 = video2[c]

        # joints_2d1 = pose_3d1[s1 + i][0]['keypoints']
        # joints_2d2 = pose_3d2[s2 + i][0]['keypoints']
        
        # frame1 = draw_2d_skeletons_3d(frame1, joints_2d1, conf_thresh = 0.35)
        # frame2 = draw_2d_skeletons_3d(frame2, joints_2d2, (255, 0, 0), 0.35)
        
        frame1 = cv2.resize(frame1, None, fx = 0.5, fy = 0.5)
        frame2 = cv2.resize(frame2, None, fx = 0.5, fy = 0.5)
        cv2.imshow('Baseline ', frame1)
        cv2.imshow('Candidate ', frame2)
        cv2.waitKey(-1)
        
def render_entire_video():
    
    name = 'Stowing_carrier'
    filename = 'baseline'
    video_path = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/" + filename + "/" + filename + ".mov"
    output_video_path = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/" + filename + "/render_" + filename + ".mov"
    pose_path_3d = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/" + filename + "/pose_3d.p"
    pose_path_2d = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/" + filename + "/pose_2d.p"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    with open(pose_path_2d, 'rb') as f:
        poses_2d = pickle.load(f)
            
    with open(pose_path_3d, 'rb') as f:
        poses_3d = pickle.load(f)

    video = mmcv.VideoReader(video_path)
    num_frames = len(video)

    video_writer = cv2.VideoWriter(output_video_path,
                            fourcc, video.fps, (video.width, video.height))
    
    print(len(poses_3d))
    for i in range(num_frames):

        frame = video[i]
        if len(poses_3d[i]):
            joints_3d = poses_3d[i][0]['keypoints_3d']
            joints_2d = poses_3d[i][0]['keypoints']

            face_2d = poses_2d[i][0]['keypoints_2d'][[0, 1, 2, 3, 4, 17]]                   
            joints_2d = np.concatenate((joints_2d, face_2d), axis = 0)

            frame = render.drawSkeleton(frame, joints_2d, [0]*9, base=True, thresh=0.35)

        video_writer.write(frame)
        # cv2.imshow('frame ', frame)
        # cv2.waitKey(-1)

    video_writer.release()

def merge_videos():

    name = 'Stowing_carrier'
    bvideo_path = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/baseline/render_baseline.mov"
    cvideo_path = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/candidate/render_candidate.mov"
    dev_video = '/home/tumeke-balaji/Documents/results/delta/joints/' + name + '/' + name + '.mov'
    output_video_path = '/home/tumeke-balaji/Documents/results/delta/joints/' + name + '/' + name + '_all.mov'

    bvideo = mmcv.VideoReader(bvideo_path)
    cvideo = mmcv.VideoReader(cvideo_path)
    dvideo = mmcv.VideoReader(dev_video)

    v1_seg = [[60, 137], [137, 140], [140, 172], [172, 175], [175, 260], [260, 280], [280, 385]] # baseline
    v2_seg = [[100, 180], [180, 225], [225, 260], [260, 260], [260, 320], [320, 360], [360, 510]] # candidate

    gidx = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path,
                                fourcc, cvideo.fps, (1080, 960))
    
    for i in range(len(v1_seg)):

        bseg, cseg = normalize_segment(v1_seg[i], v2_seg[i])

        for b, c in zip(bseg, cseg):

            if i % 2 == 0:
                out = dvideo[gidx]
                gidx += 1
                cv2.imshow('out ', out)
            else:
                bframe = bvideo[b]
                cframe = cvideo[c]
                out = np.hstack((bframe, cframe))
                out = cv2.resize(out, None, fx = 0.5, fy = 0.5)
                cv2.imshow('out ', out)
            
            cv2.waitKey(-1)
            video_writer.write(out)
        
    video_writer.release()

if __name__ == "__main__":  

    connections = [
        (0, 1),
        (1, 4),
        (1, 2),
        (2, 3),
        (4, 5),
        (5, 6),
        (1, 14),
        (4, 11),
        (11, 12),
        (12, 13),
        (14, 15),
        (15, 16),
        (8, 9),
        (9, 10),
        (14, 7),
        (7, 11),
        (14, 8),
        (8, 11),
    ]


    # Index frames in a video and save it
    # number_frames()

    # check start and end of action for video alignment
    video_align()

    # render_entire_video()

    # merge_videos()