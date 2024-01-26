import torch
from torch.utils.data import DataLoader
from pose_embedding.common import visualization_utils
import matplotlib.pyplot as plt
import time
import os
import json
import glob
import mmcv
import pickle
import argparse
import numpy as np
import shutil
from sklearn.neighbors import BallTree
import functools

from torch.utils.tensorboard import SummaryWriter
from pose_embedding.common import models, keypoint_utils, loss_utils, data_utils, constants, keypoint_profiles
from pose_embedding.dataset.pose_pairs_dataset import PosePairsDataset
from pose_embedding.human36m.Human36M import Human36M
from tqdm import tqdm
from mmdet.apis import init_detector
from mmpose.apis import init_pose_model, vis_pose_result   
from tumeke_action.api.inference import detection_inference, pose_inference

def parse_args():
    parser = argparse.ArgumentParser(description='Test pose embedding model whole pipeline.')
    parser.add_argument(
        '--writer',
        default='runs/model-test', 
        help='Tensorboard writer logs')
    parser.add_argument(
        '--dataset',
        default= "./data/test-chunk/*.h5", # "./data/test-chunks/*.h5",#
        help='ground truth dataset path of hdf5 files')    
    parser.add_argument(
        '--pose-folder',
        default= "data/poses/h36m", # will be s_09_act_02_subact_01_ca_01.pkl, etc.
        help='estimated folder of h36m dataset, will be used for saving dumped PKL files')    
    parser.add_argument(
        '--embedding-folder',
        default= "data/embedding/h36m", # will be embeddings.pkl {image_path: embedding}
        help='the dumped embeddings of h36m validation set')       
    parser.add_argument(
        '--image-folders',
        default= "/data/h36m/images/", # s_09_act_02_subact_01_ca_01/s_09_act_02_subact_01_ca_01_000091.jpg # "data/tmp", # 
        help='image folder of h36m dataset, will be used for pose estimation')
    parser.add_argument(
        '--resume-ckpt',
        default= './checkpoints/training_model_8_0.pth', # None, # 'checkpoints/model_0905_no_dropout_epoch_3/training_model_2.pth', #
        help='resume training from previous checkpoint')   
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='batch size') 
    parser.add_argument(
        '--chunks', type=int, default=4, help='split annotation into chunks')      
    
    args = parser.parse_args()   
    model_name = args.resume_ckpt.split('/')[-1]
    args.writer = f"{args.writer}-{model_name}"

    return args 

def get_actkey_frameid(dict_key):
    str_key = dict_key.split('/')[-1].split('.')[0]
    act_key, str_key = str_key.split('_ca_')
    frame_id = int(str_key.split('_')[-1])
    return act_key, frame_id
    
def show_pose(kp2d):
    # visualization  
    kp2d = kp2d.unsqueeze(0).unsqueeze(0)
    fig = visualization_utils.draw_poses_2d(data_utils.flatten_first_dims(kp2d, 
                                                            num_last_dims_to_keep=2),
                            keypoint_profile_2d=dataset.keypoint_profile_2d,num_cols=1)
    fig.savefig(f"output/test_coco13.png")
    plt.close(fig)


def save_embeddings(pose_folder, embedder):
    pose_files = glob.glob(os.path.join(pose_folder, '*.pkl'))
    embedding_file = os.path.join(args.embedding_folder, 'embeddings.pkl')
    if os.path.exists(embedding_file):
        with open(embedding_file, 'rb') as f:
            embeddings =  pickle.load(f)
    else:
        embeddings = []
    embedding_key_dict = set()
    for em in embeddings:
        embedding_key_dict.add(list(em.keys())[0])

    for pose_f in pose_files:
        print(f"Extracting the embedding vectors of {pose_f}: \n")
        with open(pose_f, "rb") as f:
            poses = pickle.load(f)
            pose_folder_name = pose_f.split('/')[-1].split('.')[0]
            for i, pose in tqdm(enumerate(poses)):
                pose_key = f"{pose_folder_name}/{pose_folder_name}_{i+1:06d}.jpg"
                if len(pose) == 0:
                    continue
                if pose_key in embedding_key_dict:
                    continue
                coco13_dict = keypoint_profiles.coco17_to_coco13(torch.tensor(pose[0]['keypoints']))

                # visualization  
                # show_pose(coco13_dict[constants.KEY_PREPROCESSED_KEYPOINTS_2D])

                coco13_dict['model_inputs'] = coco13_dict['model_inputs'].to(device)
                embedding_output, activations = embedder(coco13_dict['model_inputs'])
                for k in embedding_output:
                    embedding_output[k] = embedding_output[k].detach().cpu().numpy()
                embeddings.append( {pose_key: embedding_output})
                embedding_key_dict.add(pose_key)
        print(f"Current vectors: {len(embeddings)}")
    mmcv.dump(embeddings, os.path.join(args.embedding_folder, 'embeddings.pkl'))


""" Pose retrieval by ball-tree supported NN search
"""
def pose_retrieval_tree(embedding_index, embedding_array,  embedding_keys, h36m, q_index, nearest_k=10):
    q_key = embedding_keys[q_index] 
    q_em = embedding_array[q_index]
    act_key, frame_id = get_actkey_frameid(q_key)
    q_3d = h36m.joints3d[act_key][frame_id]

    test_folder = f"data/tmp/{act_key}-{frame_id:06d}-treenn"
    mmcv.mkdir_or_exist(test_folder)
    
    Y_flattened = q_em.reshape(-1, np.prod(q_em.shape))
    distances, indices = embedding_index.query(Y_flattened, k=nearest_k)  # For finding the nearest neighbor
    distances = distances.reshape(np.prod(distances.shape))
    indices = indices.reshape(np.prod(indices.shape))

    for d, i in zip(distances, indices):
        if d < constants.threshold_matching_prob_dist:
            act_key, frame_id = get_actkey_frameid(embedding_keys[i])
            candidate_3d = h36m.joints3d[act_key][frame_id]
            mpjpe_dist = keypoint_utils.compute_procrustes_aligned_mpjpes(q_3d, candidate_3d)
            shutil.copy(os.path.join(args.image_folders, embedding_keys[i]), test_folder)
            print(f"score: {d:.3f}, NP-MPJPE: {mpjpe_dist:.3f} -- {embedding_keys[i].split('/')[-1]}")


""" Pose retrieval by brutal force NN search
"""
def pose_retrieval_bf_nn(embedding_array, embedding_keys, h36m, embedding_sample_distance_fn, q_index, nearest_k=10):
    q_key = embedding_keys[q_index] 
    q_em = embedding_array[q_index] #[q_key][constants.KEY_EMBEDDING_SAMPLES] 
    act_key, frame_id = get_actkey_frameid(q_key)
    q_3d = h36m.joints3d[act_key][frame_id]

    test_folder = f"data/tmp/{act_key}-{frame_id:06d}-bfnn"
    mmcv.mkdir_or_exist(test_folder)

    print(q_key)
    q_dist_arr = []
    for i in tqdm(range(embedding_array.shape[0])):
        candidate_em = embedding_array[i]
        d = embedding_sample_distance_fn(q_em, candidate_em)
        q_dist_arr.append(d)

    q_dist_tensor = torch.tensor(q_dist_arr)
    result = torch.topk(q_dist_tensor, nearest_k, largest=False)
    for score, idx in zip(result[0], result[1]):
        score = float(score.detach().cpu().numpy())
        idx = int(idx.detach().cpu().numpy())
        if score < constants.threshold_matching_prob_dist:
            act_key, frame_id = get_actkey_frameid(embedding_keys[idx])
            candidate_3d = h36m.joints3d[act_key][frame_id]
            mpjpe_dist = keypoint_utils.compute_procrustes_aligned_mpjpes(q_3d, candidate_3d)
            shutil.copy(os.path.join(args.image_folders, embedding_keys[idx]), test_folder)
            print(f"score: {score:.3f}, NP-MPJPE: {mpjpe_dist:.3f} -- {embedding_keys[idx]}")

""" Pose retrieval validation on HUMAN 3.6M validation set (Subject 9 and 11)
"""
def pose_retrieve_test(embedding_file, selected_3d, n_test=5):
    # Load embeddings 
    print(f"Loading embedding vectors: {embedding_file} ...")
    with open(embedding_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    sigmoid_a, sigmoid_b = loss_utils.get_sigmoid_parameters( 
        raw_a=scalar_parameters["raw_a"],
        raw_b=scalar_parameters["raw_b"],
        a_range=(None, constants.sigmoid_a_max))      
        
    embedding_keys = []
    embedding_list = []
    for em in tqdm(embeddings):
        candidate_key = list(em.keys())[0]
        act_key, frame_id = get_actkey_frameid(candidate_key)
        if act_key in selected_3d and frame_id in selected_3d[act_key]:
            embedding_list.append(em[candidate_key][constants.KEY_EMBEDDING_SAMPLES])      
            embedding_keys.append(candidate_key)
    embedding_array = np.stack(embedding_list, axis=0)
    # flat [20, 16] into [320] for each data       
    embedding_array = embedding_array.reshape(embedding_array.shape[0], -1)

    probabilistic_distance = functools.partial(
            loss_utils.probabilistic_distance,
            sigmoid_a=float(sigmoid_a),
            sigmoid_b=float(sigmoid_b)
            )
    t0 = time.time()
    embedding_index = BallTree(embedding_array, metric=probabilistic_distance)    
    print(f"Built BallTree cost {time.time() - t0:.2f} seconds")

    # Load groundtruth of 3D keypoints for MPJPE evaluation
    print(f"Loading Human3.6M ground-truth 3D keypoints: ...")
    h36m = Human36M('test')
    h36m.load_keypoint3d()

    embedding_sample_distance_fn_kwargs ={
        'L2_SIGMOID_MATCHING_PROB_a': sigmoid_a,
        'L2_SIGMOID_MATCHING_PROB_b': sigmoid_b,
        'SQUARED_L2_SIGMOID_MATCHING_PROB_a': sigmoid_a,
        'SQUARED_L2_SIGMOID_MATCHING_PROB_b': sigmoid_b,
    }    
    embedding_sample_distance_fn = loss_utils.create_sample_distance_fn(
                        pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
                        distance_kernel=constants.positive_pairwise_distance_kernel,
                        pairwise_reduction=constants.DISTANCE_REDUCTION_MEAN,
                        componentwise_reduction=(
                            constants.DISTANCE_REDUCTION_MEAN),
                        **embedding_sample_distance_fn_kwargs)

    for i_test in range(n_test):
        q_index = np.random.randint(0, embedding_array.shape[0]) 
        t0 = time.time()
        pose_retrieval_tree(embedding_index=embedding_index, 
                            embedding_array=embedding_array, embedding_keys=embedding_keys, h36m=h36m, 
                            q_index=q_index)
        print(f"NN tree-search used {time.time() - t0:.2f} seconds")
                
        t0 = time.time()
        pose_retrieval_bf_nn(embedding_array, embedding_keys, h36m, probabilistic_distance, q_index)
        print(f"Exactly NN search used {time.time() - t0:.2f} seconds")


        

if __name__ == '__main__':
    args = parse_args()

    if args.resume_ckpt is None:
        raise ValueError('Should provide a valid checkpoint!') 
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    writer = None
    # Initialize TensorBoard writer
    # writer = SummaryWriter(args.writer)
    # To view TensorBoard, run `tensorboard --logdir=runs` in the terminal

    t0 = time.time()
    dataset = PosePairsDataset(args.dataset, is_train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Dataset preparation cost {time.time() - t0 : .2f} seconds")

    # We only need sigmoid parameters when a related distance kernel is used.
    raw_a = torch.nn.Parameter(torch.tensor(constants.sigmoid_raw_a_initial, device=device))
    raw_b = torch.nn.Parameter(torch.tensor(constants.sigmoid_b_initial, device=device))
    scalar_parameters = dict(raw_a=raw_a, raw_b=raw_b, epoch=0, epoch_step=0)

    """ Prepare pose embedder 
    """
    embedder_fn = models.get_embedder(
        base_model_type=constants.BASE_MODEL_TYPE_SIMPLE,
        embedding_type=constants.EMBEDDING_TYPE_GAUSSIAN,
        num_embedding_components=constants.num_embedding_components,
        embedding_size=constants.embedding_size,
        num_embedding_samples=constants.num_embedding_samples,
        weight_max_norm=0.0,
        feature_dim=3*dataset.keypoint_profile_2d.keypoint_num,
        seed=0)
    
    embedder_fn.keywords['base_model_fn'].to(device)
    embedder_fn.keywords['base_model_fn'].eval()

    models.load_training_state(embedder_fn=embedder_fn, optimizer=None, scalar_parameters=scalar_parameters, ckpt_resume_path=args.resume_ckpt)        
    raw_a = scalar_parameters['raw_a']
    raw_b = scalar_parameters['raw_b']
    print(raw_a, raw_b)      

    """ Prepare pose estimator 
    """
    pose_model = init_pose_model('tools/demo_config/hrnet_w32_coco_256x192.py', 
                                 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
                                device)   
    det_model = init_detector('tools/demo_config/faster_rcnn_r50_fpn_2x_coco.py', 
                              'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                                'faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_'
                                'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth', 
                                device)
    assert det_model is not None, ('Failed to build the detection model. Check if you have installed mmcv-full properly. '
                               'You should first install mmcv-full successfully, then install mmdet, mmpose. ')
    assert det_model.CLASSES[0] == 'person', 'We require you to use a detector trained on COCO'

    mmcv.mkdir_or_exist(args.pose_folder)
    # eval(embedder_fn=embedder_fn, loader=loader, scalar_parameters=scalar_parameters, writer=writer, device=device)

    subject = [9, 11]
    subject_folders = []
    det_score_thr = 0.9
    for sub in subject:
        subject_folders.extend(glob.glob(os.path.join(args.image_folders, f"s_{sub:02d}_*")))

    subject_folders.sort()

    chunk_size = len(subject_folders) // args.chunks if len(subject_folders) > args.chunks else len(subject_folders)
    start = args.gpu_id * chunk_size
    if args.gpu_id + 1 == args.chunks:
        end = len(subject_folders)
    else:
        end = start + chunk_size

    for folder in subject_folders[start: end]:
        print(f"GPU-{args.gpu_id}: {folder}")
        pose_file = os.path.join(args.pose_folder, folder.split('/')[-1] + ".pkl")
        if os.path.exists(pose_file):
            continue
        frame_paths = glob.glob(os.path.join(folder, '*.jpg'))
        frame_paths.sort()
        try:
            det_results = detection_inference(det_score_thr, frame_paths, det_model)
        except Exception as e:
            print("Error occurs in detection_inference:", e)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            pose_results = pose_inference(frame_paths, det_results, pose_model)
        except Exception as e:
            print("Error occurs in pose_inference:", e)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mmcv.dump(pose_results, pose_file)
    

    """ Now we have the estimated poses of each image, we will use our trained embedder to get the embedding vectors
    """
    # save_embeddings(pose_folder=args.pose_folder, embedder=embedder_fn)

    """ Now we have a list (~450k) of dict {pose_key_file: embedding_dict {key: embedding_vec}}
    """
    with open("data/h36m_selected_3d_frames.json","r") as f:
        selected_3d = json.load(f)

    for k in selected_3d:
        selected_3d[k] = set(selected_3d[k])
    
    with torch.no_grad():
        pose_retrieve_test(embedding_file=os.path.join(args.embedding_folder, 'embeddings.pkl'), selected_3d=selected_3d, n_test=5)