import mmcv
import numpy as np
import cv2
import pickle

def number_frames(input_video_path, output_video_path):
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

    return new_bids, new_cids

def get_segs():

    # Stowing_carrier
    # v1_seg = [[60, 137], [140, 172], [175, 260], [280, 385]] # baseline
    # v2_seg = [[100, 180], [225, 260], [260, 320], [360, 510]] # candidate
    
    # Lower_Galley_Carrier
    # v1_seg = [[31, 200], [240, 300]] # baseline
    # v2_seg = [[31, 120], [135, 185]] # candidate

    # Pushing_cart
    # v1_seg = [[60, 120]] # baseline
    # v2_seg = [[60, 100]] # candidate

    # Removing_Item_from_Bottom_of_Cart
    # v1_seg = [[5, 55], [55, 100], [125, 270]] # baseline
    # v2_seg = [[57, 95], [110, 150], [155, 330]] # candidate

    # Serving_from_Basket
    v1_seg = [[90, 155], [165, 185], [185, 240], [245, 300]] # baseline
    v2_seg = [[90, 120], [120, 145], [145, 193],[ 195, 220]] # candidate

    return v1_seg, v2_seg 

def video_align():

    name = 'Lower_Galley_Carrier'
    video_path1 = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/baseline/baseline_n.mov"
    pose_path1 = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/baseline/pose_3d.p"

    video_path2 = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/candidate/candidate_n.mov"
    pose_path2 = "/home/tumeke-balaji/Documents/results/delta/joints/" + name + "/candidate/pose_3d.p"

    # with open(pose_path1, 'rb') as f:
    #     pose_3d1 = pickle.load(f)

    # with open(pose_path2, 'rb') as f:
    #     pose_3d2 = pickle.load(f)

    # Load videos
    video1 = mmcv.VideoReader(video_path1)
    video2 = mmcv.VideoReader(video_path2)

    v1_seg, v2_seg = get_segs()

    # Iterate through both videos
    for i in range(0, len(v1_seg)):
        
        bseg, cseg = normalize_segment(v1_seg[i], v2_seg[i]) # make sure the length of the segments are same
        print(len(bseg), len(cseg))

        for b, c in zip(bseg, cseg):

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
    vname = 'candidate'
    act_name = 'Lower_Galley_Carrier'
    root_dir = '/home/tumeke-balaji/Documents/results/delta/joints/' + act_name + '/'
    input_video_path =  root_dir + vname + '/' + vname + '.mov'
    ext = input_video_path.rsplit('.', 1)[1]
    output_video_path = input_video_path.rsplit('.', 1)[0] + '_n.' + ext
    print('input_video_path ', input_video_path)
    print('output_video_path ', output_video_path)
    number_frames(input_video_path, output_video_path)

    # check start and end of action for video alignment
    video_align()