import argparse
import os
from tqdm import tqdm
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description='extract poses from videos.')
    parser.add_argument(
        '--input',
        default= '/home/jun/data/delta-clips/stowing-carrier', 
        help='video folder is $input/videos, pose folder is $input/poses/joints')
    args = parser.parse_args()   

    args.video_folder = os.path.join(args.input, 'videos') 
    args.pose_folder = os.path.join(args.input, 'poses/joints') 
    args.normal_folder = os.path.join(args.input, 'videos/normalized_video') 
    return args 

def video_h264_normalize():
    args = parse_args()
    mmcv.mkdir_or_exist(args.pose_folder)
    mmcv.mkdir_or_exist(args.normal_folder)    
    vid_files = os.listdir(args.video_folder)

    for vid_basename in tqdm(vid_files):
        basename = vid_basename.split('.')[0]
        
        vid_file = os.path.join(args.video_folder, vid_basename)
        if not os.path.isfile(vid_file):
            continue
        print(f"Processing {vid_file} ... ")
        vid = mmcv.VideoReader(vid_file)
        if vid.fps == 0:
            print(f"Warning: {vid_basename} fps == 0, skip this file ......")
            continue        
        total_sec = vid.frame_cnt / vid.fps
        cur_path = os.path.join(args.normal_folder, basename+'_normal.mp4')
        if not os.path.exists(cur_path):
            mmcv.cut_video(vid_file, cur_path, start=0, end=total_sec, vcodec='h264')
    return


if __name__ == '__main__':
    video_h264_normalize()