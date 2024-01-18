import glob
import os
import mmcv
import torch
import time
import pickle
from pose_embedding.common import models, keypoint_utils, loss_utils, data_utils, constants, keypoint_profiles
import extract_embedding
import cross_sim_video_alignment as csva
import tqdm 
import json
import numpy as np

def save_embeddings(pose_folder, video_folder, embedder, pose_embedding_folder, scalar_parameters, device, post_fix=".mp4"):

    pose_files = glob.glob(os.path.join(pose_folder, '*.pkl'))    
    coco_profile = "LEGACY_2DCOCO13"

    for pose_f in pose_files:
        pose_name = pose_f.split('/')[-1].split('.')[0]
        video_f = os.path.join(video_folder, f"{pose_name}{post_fix}")
        embedding_file = os.path.join(pose_embedding_folder, f'{pose_name}_embeddings.pkl')
        embedding_file_batch = os.path.join(pose_embedding_folder, f'{pose_name}_embeddings_batch.pkl')
    
        embeddings = []
        if not os.path.exists(video_f):
            print(f"Video file: {video_f} not exists, use default fps=30 ... ")
            meta_info = dict(fps=30, raw_a=scalar_parameters['raw_a'].detach().cpu().numpy(), 
                            raw_b=scalar_parameters['raw_b'].detach().cpu().numpy())
        else:            
            vid = mmcv.VideoReader(video_f)
            meta_info = dict(fps=vid.fps, raw_a=scalar_parameters['raw_a'].detach().cpu().numpy(), 
                            raw_b=scalar_parameters['raw_b'].detach().cpu().numpy())
              
        with open(pose_f, "rb") as f:
            poses = pickle.load(f)

            t0 = time.time()
            embeddings = extract_embedding.process_pose_embedding(embedder, poses, 0, len(poses), coco_profile, mask_type="NONE", device=device)
            print(f"Sequential processing {len(embeddings)}, fps={len(embeddings)/(time.time()-t0):.2f}")

            t1 = time.time()
            embeddings_batch = extract_embedding.process_embedding_batch(embedder, poses, 0, len(poses), coco_profile, mask_type="NONE", device=device)
            print(f"Batch processing {embedding_file}: {len(embeddings)}, fps={len(embeddings)/(time.time()-t1):.2f}")

        print(f"Embeddings saved in {embedding_file}: {len(embeddings)}, fps={len(embeddings)/(time.time()-t0):.2f}")
        embedding_dict=dict(embeddings=embeddings, meta_info=meta_info)
        embedding_dict_batch=dict(embeddings=embeddings_batch, meta_info=meta_info)
        mmcv.dump(embedding_dict, embedding_file)
        mmcv.dump(embedding_dict_batch, embedding_file_batch)

def video_h264_normalize(video_folder, normal_folder):
    
    vid_files = os.listdir(video_folder)
    
    for vid_basename in vid_files:
        basename = vid_basename.split('.')[0]
        
        vid_file = os.path.join(video_folder, vid_basename)
        if not os.path.isfile(vid_file):
            continue
        print(f"Processing {vid_file} ... ")
        vid = mmcv.VideoReader(vid_file)
        if vid.fps == 0:
            print(f"Warning: {vid_basename} fps == 0, skip this file ......")
            continue        
        total_sec = vid.frame_cnt / vid.fps
        cur_path = os.path.join(normal_folder, basename+'_normal.mp4')
        if not os.path.exists(cur_path):
            mmcv.cut_video(vid_file, cur_path, start=0, end=total_sec, vcodec='h264')

    return

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def extract_save_embedding(input_dir, embedder_fn, scalar_parameters):

    pose_folder = os.path.join(input_dir, 'poses/joints') 
    video_folder = os.path.join(input_dir, 'videos') 
    normal_folder = os.path.join(input_dir, 'videos/normalized_video') 
    pose_embedding_folder = os.path.join(input_dir, 'poses/embedding') 
    
    create_folder_if_not_exists(normal_folder)    
    create_folder_if_not_exists(pose_folder)
    create_folder_if_not_exists(pose_embedding_folder)

    # Normalize video
    print('Normalizing videos ....')
    video_h264_normalize(video_folder, normal_folder)

    print('Save embeddings ....')
    with torch.no_grad():
        save_embeddings(pose_folder=pose_folder, embedder=embedder_fn, 
                        video_folder=normal_folder, pose_embedding_folder=pose_embedding_folder, scalar_parameters=scalar_parameters, device=device, post_fix="_normal.mp4") # "_normal.mp4"
                        # video_folder=args.video_folder, post_fix=".mp4") # "_normal.mp4"

def get_embed_fn(profile_name = "LEGACY_2DCOCO13"):

    resume_ckpt = "./checkpoints/fullbody-occlusion-1018-0.5-0.2/training_model_10_0.pth"

    # We only need sigmoid parameters when a related distance kernel is used.
    raw_a = torch.nn.Parameter(torch.tensor(constants.sigmoid_raw_a_initial, device=device))
    raw_b = torch.nn.Parameter(torch.tensor(constants.sigmoid_b_initial, device=device))
    scalar_parameters = dict(raw_a=raw_a, raw_b=raw_b, epoch=0, epoch_step=0)
    keypoint_profile_2d = keypoint_profiles.create_keypoint_profile(profile_name)

    """ Prepare pose embedder 
    """
    embedder_fn = models.get_embedder(
        base_model_type=constants.BASE_MODEL_TYPE_SIMPLE,
        embedding_type=constants.EMBEDDING_TYPE_GAUSSIAN,
        num_embedding_components=constants.num_embedding_components,
        embedding_size=constants.embedding_size,
        num_embedding_samples=constants.num_embedding_samples,
        weight_max_norm=0.0,
        feature_dim=3*keypoint_profile_2d.keypoint_num,
        seed=0)
    
    embedder_fn.keywords['base_model_fn'].to(device)
    embedder_fn.keywords['base_model_fn'].eval()

    models.load_training_state(embedder_fn=embedder_fn, optimizer=None, scalar_parameters=scalar_parameters, ckpt_resume_path=resume_ckpt)        
    raw_a = scalar_parameters['raw_a']
    raw_b = scalar_parameters['raw_b']
    print(raw_a, raw_b)

    return embedder_fn, scalar_parameters

def cross_sim_video_align():      

    task_name = input_dir.split('/')[-1]
    pose_folder = os.path.join(input_dir, 'poses/joints') 
    normal_folder = os.path.join(input_dir, 'videos/normalized_video') 
    pose_embedding_folder = os.path.join(input_dir, 'poses/embedding') 

    mmcv.mkdir_or_exist(pose_folder)
    task_path = input_dir

    template_name  = "baseline" # "baseline-IMG_0032" # "baseline"
    candidate_name = "candidate" # "candidate-IMG_0033" # "candidate-good"
    pose_win_sec = 0.3  #pose_win_sec=0.3
    distance_np, dtw_path = csva.cross_video_alignment(template_name, candidate_name, task_path, scalar_parameters, pose_embedding_folder, pose_folder, normal_folder, device, pose_win_sec=pose_win_sec)
    categorized_path = csva.generate_aligned_unaligned_path(template_name, candidate_name, distance_np, dtw_path, pose_embedding_folder, pose_win_sec=pose_win_sec)
    with open(os.path.join(task_path, f"{template_name}_{candidate_name}-dtw_path.json"),"w") as f:
        json.dump(categorized_path, f)
    csva.generate_dtw_alignment_video(template_name, candidate_name, distance_np, np.array(dtw_path), task_path, pose_folder, normal_folder, pose_win_sec=pose_win_sec)

    return categorized_path

def list_subdirs(directory_path):
    subdirs = [os.path.join(directory_path, d) for d in os.listdir(directory_path) 
               if os.path.isdir(os.path.join(directory_path, d))]
    return subdirs

if __name__ == "__main__":

    root_dir = "/home/tumeke-balaji/Documents/results/delta/input_videos/Customer_Facing_Demos/"
    input_dirs = list_subdirs(root_dir) # "Serving_from_Basket" #  # 'Removing_Item_from_Bottom_of_Cart' # #'Serving_from_Basket' # 'Pushing_cart' # 'Lower_Galley_Carrier'
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    for input_dir in input_dirs:
        print('input_dir ', input_dir)
        embedder_fn, scalar_parameters = get_embed_fn()
        extract_save_embedding(input_dir, embedder_fn, scalar_parameters)
        cross_sim_video_align()
