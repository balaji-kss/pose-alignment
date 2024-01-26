import torch
# from pose_embedding.common import visualization_utils
# import matplotlib.pyplot as plt
import time
import os
import json
import glob
import mmcv
import pickle
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# from sklearn.neighbors import BallTree
# from fastdtw import fastdtw
# from pose_embedding.common.utils.peaks import valleys_dynamic_weighting_thresholding, \
#                                                 height_based_detection_without_prominence, \
#                                                 plot_and_save_valleys
    
# from pose_embedding.common.utils.dtw import temporal_averaging, compute_pairwise_distances, dtw_with_precomputed_distances
import cv2
import functools
import imageio
import io

from pose_embedding.common import models, keypoint_utils, loss_utils, data_utils, constants, keypoint_profiles
from tqdm import tqdm
# from mmdet.apis import init_detector
# from mmpose.apis import init_pose_model, vis_pose_result   

def parse_args():
    parser = argparse.ArgumentParser(description='Extract pose embedding.')
    parser.add_argument(
        '--input',
        default='/home/jun/data/delta-clips/stowing-carrier', #, serving-from-basket, closing-overhead-bin, removing-item-from-bottom-of-cart, stow-full-cart
        # default='/data/junhe/ActionRecognition/data/custom-bg-poses',
        help='video folder is $input/videos, pose folder is $input/poses/joints')  
    parser.add_argument(
        '--bodypart',
        default="fullbody-occlusion-1018-0.5-0.2",
        help='fullbody or upperbody pose embedding model')      
    parser.add_argument('--mask-type', default= "LEFT_RIGHT_ANKLE_KNEE")   # RIGHT_ELBOW_WRIST LEFT_RIGHT_ANKLE_KNEE
    parser.add_argument(
        '--output',
        default= "output/", #
        help='folder for generated results')
    parser.add_argument(
        '--resume-ckpt',
        default='training_model_10_0.pth', 
        help='resume training from previous checkpoint')   
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--tensorboard', type=bool, default=False, help='log to Tensorboard or not')
        
    args = parser.parse_args()
    args.pose_folder = os.path.join(args.input, 'poses/joints') 
    args.video_folder = os.path.join(args.input, 'videos') 
    args.normal_folder = os.path.join(args.input, 'videos/normalized_video') 

    args.output = os.path.join(args.output, args.bodypart)

    # if args.bodypart == "fullbody":
    #     args.pose_embedding_folder = os.path.join(args.input, 'poses/embedding') 
    #     args.coco_profile = "LEGACY_2DCOCO13"
    #     args.resume_ckpt = f"./checkpoints/{args.bodypart}-model-occlusion/" + args.resume_ckpt
    # elif args.bodypart == "upperbody":
    #     args.pose_embedding_folder = os.path.join(args.input, 'poses/embedding-upperbody') 
    #     args.coco_profile = "UPPERBODY_9_2D"
    #     args.resume_ckpt = f"./checkpoints/{args.bodypart}-model-occlusion/" + args.resume_ckpt
    # else:
    #     args.pose_embedding_folder = os.path.join(args.output, 'embedding') 
    args.pose_embedding_folder = os.path.join(args.input, 'poses/embedding') 
    args.coco_profile = "LEGACY_2DCOCO13"
    args.resume_ckpt = f"./checkpoints/{args.bodypart}/" + args.resume_ckpt

    args.model_name = args.resume_ckpt.split('/')[-1]

    mmcv.mkdir_or_exist(args.output)
    mmcv.mkdir_or_exist(args.pose_embedding_folder)
    return args 

def get_saliency(person_dict):
    # Assuming the bounding box format is [x1, y1, x2, y2, score]
    bbox = person_dict['bbox']
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    area = w * h
    return area  

def process_pose_embedding(embedder, pose_results, start, end, embedding_profile_name_2d, device, mask_type="NONE"):
    embeddings = []
    for i, poses in enumerate(pose_results[start:end]):
        embs = []
        if len(poses) == 0:
            # no person detected
            coco13_dict = keypoint_profiles.coco17_to_coco13(torch.zeros(17, 3), coco_profile=embedding_profile_name_2d, mask_type=mask_type)
            embeddings.append(embs)
        else:
            for idx, person_pose in enumerate(poses):
                person_pose['keypoints'] = person_pose['keypoints_2d'][:17]
                coco13_dict = keypoint_profiles.coco17_to_coco13(torch.tensor(person_pose['keypoints_2d']), 
                                                                    coco_profile=embedding_profile_name_2d,
                                                                    mask_type=mask_type)
                # visualization  
                # show_pose(coco13_dict[constants.KEY_PREPROCESSED_KEYPOINTS_2D])
                coco13_dict['model_inputs'] = coco13_dict['model_inputs'].to(device)
                embedding_output, activations = embedder(coco13_dict['model_inputs'])
                for k in embedding_output:
                    embedding_output[k] = embedding_output[k].detach().cpu().numpy()
                emb_dict = {"track_id": person_pose.get('track_id',idx), "bbox": person_pose["bbox"]}
                emb_dict.update(embedding_output)
                embs.append(emb_dict)
            embeddings.append(embs)
    return embeddings

def process_embedding_batch(embedder, samples, start, end, embedding_profile_name_2d, device, batch_size=256, mask_type="NONE"):
    def process_batch(batch):
        batch_inputs_tensor = torch.stack(batch).to(device)
        batch_output, _ = embedder(batch_inputs_tensor)
        return batch_output

    def batch_postprocessing(batch_output):
        for key in batch_output:
            # Split the batched tensor for each key
            split_tensors = batch_output[key].detach().cpu().numpy()
            for idx_tuple, tensor in zip(sample_indices, split_tensors):
                # Ensure there's a dictionary for the current sample
                idx = idx_tuple[0]
                track_id = idx_tuple[1]['track_id']
                track_id_idx = -1
                for j, emb_dict in enumerate(embeddings[idx]):
                    if track_id == emb_dict['track_id']:
                        track_id_idx = j
                        break
                if track_id_idx == -1:  # new track_id found, add it to current frame list
                    embeddings[idx].append(idx_tuple[1])

                embeddings[idx][track_id_idx][key] = tensor
        
    embeddings = [[] for _ in range(end - start)]
    batch_inputs = []
    sample_indices = []
    for i, sample in enumerate(samples[start:end]):
        if len(sample) == 0:
            continue
        for idx, person in enumerate(sample):
            coco13_dict = keypoint_profiles.coco17_to_coco13(torch.tensor(person['keypoints']), 
                                                    coco_profile=embedding_profile_name_2d,
                                                    mask_type=mask_type)
            batch_inputs.append(coco13_dict['model_inputs'])
            sample_indices.append((i, {"track_id": person.get('track_id',idx), "bbox": person["bbox"]}))

            # Process the batch when the specified batch size is reached
            if len(batch_inputs) == batch_size:
                batch_output = process_batch(batch_inputs)
                batch_postprocessing(batch_output)
                batch_inputs = []
                sample_indices = []

    # Process any remaining items in the batch
    if batch_inputs:
        batch_output = process_batch(batch_inputs)
        batch_postprocessing(batch_output)

    return embeddings


def save_embeddings(pose_folder, video_folder, embedder, post_fix=".mp4"):
    pose_files = glob.glob(os.path.join(pose_folder, '*.pkl'))    
    i_emb = 0
    for pose_f in pose_files:
        pose_name = pose_f.split('/')[-1].split('.')[0]
        video_f = os.path.join(video_folder, f"{pose_name}{post_fix}")
        embedding_file = os.path.join(args.pose_embedding_folder, f'{pose_name}_embeddings.pkl')
        embedding_file_batch = os.path.join(args.pose_embedding_folder, f'{pose_name}_embeddings_batch.pkl')
        # if os.path.exists(embedding_file):
        #     print(f"Embedding vectors of {pose_f} exists")
        #     continue
        # else:
        #     print(f"Extracting the embedding vectors of {pose_f}")

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
            embeddings = process_pose_embedding(embedder, poses, 0, len(poses), args.coco_profile, mask_type="NONE")
            print(f"Sequential processing {len(embeddings)}, fps={len(embeddings)/(time.time()-t0):.2f}")

            t1 = time.time()
            embeddings_batch = process_embedding_batch(embedder, poses, 0, len(poses), args.coco_profile, mask_type="NONE")
            print(f"Batch processing {embedding_file}: {len(embeddings)}, fps={len(embeddings)/(time.time()-t1):.2f}")

            # for i, pose in tqdm(enumerate(poses)):
            #     # pose_key = i # f"{i:06d}.jpg"
            #     if len(pose) == 0:
            #         coco13_dict = keypoint_profiles.coco17_to_coco13(torch.zeros(17, 3), coco_profile=args.coco_profile, mask_type=args.mask_type)
            #     else:
            #         most_salient_person = max(pose, key=get_saliency)
            #         most_salient_person['keypoints'] = most_salient_person['keypoints'][:17]
            #         coco13_dict = keypoint_profiles.coco17_to_coco13(torch.tensor(most_salient_person['keypoints']), \
            #                                                          coco_profile=args.coco_profile, mask_type=args.mask_type)
            #     coco13_dict['model_inputs'] = coco13_dict['model_inputs'].to(device)
            #     embedding_output, activations = embedder(coco13_dict['model_inputs'])
            #     i_emb += 1
            #     for k in activations:
            #         if args.tensorboard:
            #             writer.add_histogram(k, activations[k], i_emb)                       
            #     for k in embedding_output:
            #         if args.tensorboard:
            #             writer.add_histogram(k, embedding_output[k], i_emb)    
            #         embedding_output[k] = embedding_output[k].detach().cpu().numpy()
            #     embeddings.append(embedding_output)
        print(f"Embeddings saved in {embedding_file}: {len(embeddings)}, fps={len(embeddings)/(time.time()-t0):.2f}")
        embedding_dict=dict(embeddings=embeddings, meta_info=meta_info)
        embedding_dict_batch=dict(embeddings=embeddings_batch, meta_info=meta_info)
        mmcv.dump(embedding_dict, embedding_file)
        mmcv.dump(embedding_dict_batch, embedding_file_batch)

if __name__ == '__main__':
    args = parse_args()

    if args.resume_ckpt is None:
        raise ValueError('Should provide a valid checkpoint!') 
    if "fullbody" in args.bodypart:
        profile_name = "LEGACY_2DCOCO13"
    elif "upperbody" in args.bodypart:
        profile_name = "UPPERBODY_9_2D"
    else:
        raise ValueError(f'Bodypart --{args.bodypart}-- should be either fullbody or upperbody!') 
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    if args.tensorboard:
        writer = SummaryWriter(f"runs/embedding-{args.resume_ckpt.split('/')[-1]}/" + args.bodypart + "-" + args.mask_type)

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

    models.load_training_state(embedder_fn=embedder_fn, optimizer=None, scalar_parameters=scalar_parameters, ckpt_resume_path=args.resume_ckpt)        
    raw_a = scalar_parameters['raw_a']
    raw_b = scalar_parameters['raw_b']
    print(raw_a, raw_b)      

    # """ Prepare pose estimator 
    # """
    # pose_model = init_pose_model('pose_embedding/tools/demo_config/hrnet_w32_coco_256x192.py', 
    #                              'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
    #                             device)   
    # det_model = init_detector('pose_embedding/tools/demo_config/faster_rcnn_r50_fpn_2x_coco.py', 
    #                           'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
    #                             'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
    #                             'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth', 
    #                             device)
    # assert det_model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
    #                            'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    # assert det_model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'

    mmcv.mkdir_or_exist(args.pose_folder)

    with torch.no_grad():
        save_embeddings(pose_folder=args.pose_folder, embedder=embedder_fn, 
                        video_folder=args.normal_folder, post_fix="_normal.mp4") # "_normal.mp4"
                        # video_folder=args.video_folder, post_fix=".mp4") # "_normal.mp4"