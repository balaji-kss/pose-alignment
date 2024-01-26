import torch
from torch.utils.data import DataLoader
from pose_embedding.common import visualization_utils
import matplotlib.pyplot as plt
import time
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from pose_embedding.common import models, keypoint_utils, loss_utils, data_utils, constants, distance_utils
from pose_embedding.dataset.pose_pairs_dataset import PosePairsDataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Pose embedding model evaluation.')
    parser.add_argument(
        '--writer',
        default='runs/model-eval', 
        help='Tensorboard writer logs')
    parser.add_argument(
        '--dataset',
        default= "./data/test-chunk/*.h5", # "./data/test-chunks/*.h5",#
        help='dataset path of hdf5 files')
    parser.add_argument(
        '--resume-ckpt',
        default= './checkpoints/training_model_8_0.pth', # None, # 'checkpoints/model_0905_no_dropout_epoch_3/training_model_2.pth', #
        help='resume training from previous checkpoint')   
    parser.add_argument(
        '--gpu-id', type=int, default=2, help='CUDA device id')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='batch size') 
    parser.add_argument(
        '--keypoint-dropout-rate', type=float, default=0.1, help='apply occlusion augmentation') 
    
    args = parser.parse_args()   
    model_name = args.resume_ckpt.split('/')[-1]
    args.writer = f"{args.writer}-{model_name}"

    return args 

def eval(embedder_fn, loader, scalar_parameters, writer, device):
    with torch.no_grad():
        sigmoid_a, sigmoid_b = loss_utils.get_sigmoid_parameters( 
            raw_a=scalar_parameters["raw_a"],
            raw_b=scalar_parameters["raw_b"],
            a_range=(None, constants.sigmoid_a_max))        
        embedding_sample_distance_fn_kwargs ={
            'L2_SIGMOID_MATCHING_PROB_a': sigmoid_a,
            'L2_SIGMOID_MATCHING_PROB_b': sigmoid_b,
            'SQUARED_L2_SIGMOID_MATCHING_PROB_a': sigmoid_a,
            'SQUARED_L2_SIGMOID_MATCHING_PROB_b': sigmoid_b,
        }    
        # embedding_sample_distance_fn = loss_utils.create_sample_distance_fn(
        #                         pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
        #                         distance_kernel=constants.triplet_distance_kernel,
        #                         pairwise_reduction=constants.triplet_pairwise_reduction,
        #                         componentwise_reduction=constants.triplet_componentwise_reduction,
        #                         **embedding_sample_distance_fn_kwargs)        
        triplet_embedding_sample_distance_fn = loss_utils.create_sample_distance_fn(
                        pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
                        distance_kernel=constants.triplet_distance_kernel,
                        pairwise_reduction=constants.DISTANCE_REDUCTION_MEAN,
                        componentwise_reduction=constants.DISTANCE_REDUCTION_MEAN,
                        **embedding_sample_distance_fn_kwargs)
        positive_pairwise_embedding_sample_distance_fn = loss_utils.create_sample_distance_fn(
                        pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
                        distance_kernel=constants.positive_pairwise_distance_kernel,
                        pairwise_reduction=constants.DISTANCE_REDUCTION_MEAN,
                        componentwise_reduction=(
                            constants.DISTANCE_REDUCTION_MEAN),
                        **embedding_sample_distance_fn_kwargs)
        step = 0
        eval_dict = {
            'triplet_loss/Anchor/Positive/Distance/Mean': 0.0,
            'triplet_loss/Anchor/HardNegative/Distance/Mean': 0.0,
            'triplet_loss/Anchor/HardNegative/Distance/Median': 0.0,
            'triplet_loss/Anchor/SemiHardNegative/Distance/Mean': 0.0,
            'triplet_loss/Anchor/SemiHardNegative/Distance/Median': 0.0
        }
        for i, data in tqdm(enumerate(loader)):
            # if i % 5 !=0:
            #     continue
            step += 1
            summaries = {}
            data['model_inputs']=data['model_inputs'].to(device)
            data[constants.KEY_KEYPOINTS_3D]=data[constants.KEY_KEYPOINTS_3D].to(device)
            outputs, activations = embedder_fn(data['model_inputs'])
            # positive_pairwise_A_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 0, ...]
            # positive_pairwise_B_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 1, ...]            
            # to do ....
            # Prepare keypoint trilet loss
            anchor_keypoint_masks_3d, positive_keypoint_masks_3d = None, None
            anchor_keypoints_3d = data[constants.KEY_KEYPOINTS_3D][:, 0, ...]
            positive_keypoints_3d = data[constants.KEY_KEYPOINTS_3D][:, 1, ...]

            triplet_anchor_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 0, ...]
            triplet_positive_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 1, ...]

            triplet_anchor_mining_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 0, ...]
            triplet_positive_mining_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 1, ...]

            # positive_pairwise_anchor_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 0, ...]
            # positive_pairwise_positive_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 1, ...]
            dist = positive_pairwise_embedding_sample_distance_fn(triplet_anchor_embeddings, triplet_positive_embeddings)
            anchor_match_distance_matrix = distance_utils.compute_distance_matrix(
                triplet_anchor_embeddings,
                triplet_positive_embeddings,
                distance_fn=positive_pairwise_embedding_sample_distance_fn)            
            dist_sort, dist_indices = torch.sort(dist, descending=False) # for pairwise_reduction=constants.DISTANCE_REDUCTION_MEAN,
            # dist_sort, dist_indices = torch.sort(dist, descending=True) # for pairwise_reduction=constants.DISTANCE_REDUCTION_NEG_LOG_MEAN,
            dist_indices = dist_indices.cpu()
            block = 1
            for i_show in range(block):
                print(dist_indices[i_show*block:(i_show+1)*block], dist_sort[i_show*block:(i_show+1)*block])
                fig = visualization_utils.draw_poses_2d(data_utils.flatten_first_dims(data[constants.KEY_PREPROCESSED_KEYPOINTS_2D][dist_indices[i_show*block:(i_show+1)*block]], 
                                                                                      num_last_dims_to_keep=2),
                                                        keypoint_profile_2d=dataset.keypoint_profile_2d,num_cols=block)
                fig.savefig(f"output/eval_show_{i_show}.png")
                plt.close(fig)

                # evaluate and visualize the hardest samples
                for i_hard, hard_id in enumerate(dist_indices[i_show*block:(i_show+1)*block]):
                    neg_sort, neg_indices = torch.sort(anchor_match_distance_matrix[hard_id], descending=True) # for pairwise_reduction=constants.DISTANCE_REDUCTION_MEAN,
                    # neg_sort, neg_indices = torch.sort(anchor_match_distance_matrix[hard_id], descending=False) # for pairwise_reduction=constants.DISTANCE_REDUCTION_NEG_LOG_MEAN,
                    hard_neigbors = []
                    hard_neigbors_mpjpe = []
                    hard_neigbors_matching_prob = []
                    for k in range(block):
                        similars_3d = torch.stack([data[constants.KEY_KEYPOINTS_3D][hard_id][0], data[constants.KEY_KEYPOINTS_3D][neg_indices[k]][1]])
                        np_mpjpe = keypoint_distance_fn(similars_3d[0],similars_3d[1])
                        similars = torch.stack([data[constants.KEY_PREPROCESSED_KEYPOINTS_2D][hard_id][0], data[constants.KEY_PREPROCESSED_KEYPOINTS_2D][neg_indices[k]][1]])
                        hard_neigbors.append(similars)
                        hard_neigbors_mpjpe.append(float(np_mpjpe.cpu().numpy()))
                        hard_neigbors_matching_prob.append(float(neg_sort[k].cpu().numpy()))
                    hard_neigbors = torch.stack(hard_neigbors)
                    fig = visualization_utils.draw_poses_2d(data_utils.flatten_first_dims(hard_neigbors, 
                                                                        num_last_dims_to_keep=2),
                                        keypoint_profile_2d=dataset.keypoint_profile_2d,num_cols=block)
                    hard_neigbors_str = f"{hard_id}: "
                    hard_neigbors_str += ', '.join(['{:.2f}'.format(f) for f in hard_neigbors_mpjpe])
                    hard_neigbors_str += '\n'
                    hard_neigbors_str += ', '.join(['{:.2f}'.format(f) for f in hard_neigbors_matching_prob])
                    fig.suptitle(hard_neigbors_str)
                    plt.tight_layout()
                    fig.savefig(f"output/hard_{i_show}_example_{i_hard}.png")
                    plt.close(fig)


            triplet_loss, triplet_loss_summaries = (
                            loss_utils.compute_keypoint_triplet_losses(
                                anchor_embeddings=triplet_anchor_embeddings,
                                positive_embeddings=triplet_positive_embeddings,
                                match_embeddings=triplet_positive_embeddings,
                                anchor_keypoints=anchor_keypoints_3d,
                                match_keypoints=positive_keypoints_3d,
                                margin=constants.triplet_loss_margin,                   # 0.69314718056
                                min_negative_keypoint_distance=constants.min_negative_keypoint_mpjpe,  # 0.1
                                use_semi_hard=constants.use_semi_hard_triplet_negatives,    # True
                                exclude_inactive_triplet_loss=True,
                                anchor_keypoint_masks=anchor_keypoint_masks_3d,
                                match_keypoint_masks=positive_keypoint_masks_3d,
                                embedding_sample_distance_fn=triplet_embedding_sample_distance_fn,
                                keypoint_distance_fn=keypoint_distance_fn,
                                anchor_mining_embeddings=triplet_anchor_mining_embeddings,
                                positive_mining_embeddings=triplet_positive_mining_embeddings,
                                match_mining_embeddings=triplet_positive_mining_embeddings,
                                summarize_percentiles=True))
            summaries.update(triplet_loss_summaries)            
            # positive_pairwise_loss, positive_pairwise_loss_summaries = (
            #     loss_utils.compute_positive_pairwise_loss(
            #         positive_pairwise_anchor_embeddings,
            #         positive_pairwise_positive_embeddings,
            #         loss_weight=constants.positive_pairwise_loss_weight,
            #         distance_fn=positive_pairwise_embedding_sample_distance_fn
            #         )
            #     )
            # summaries.update(positive_pairwise_loss_summaries)
            # for key, value in summaries.items():
            #     writer.add_scalar(key, value, step)
            for key in eval_dict.keys():
                if key in summaries:
                    eval_dict[key] += summaries[key]
            if i % 100 == 0:
                for k, v in eval_dict.items():
                    print(f"{k}: {v/step: .4f}")


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
    dataset = PosePairsDataset(args.dataset, is_train=False,keypoint_dropout_rate=args.keypoint_dropout_rate)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"Dataset preparation cost {time.time() - t0 : .2f} seconds")

    # We only need sigmoid parameters when a related distance kernel is used.
    raw_a = torch.nn.Parameter(torch.tensor(constants.sigmoid_raw_a_initial, device=device))
    raw_b = torch.nn.Parameter(torch.tensor(constants.sigmoid_b_initial, device=device))
    scalar_parameters = dict(raw_a=raw_a, raw_b=raw_b, epoch=0, epoch_step=0)

    keypoint_distance_fn = keypoint_utils.compute_procrustes_aligned_mpjpes

    embedder_fn = models.get_embedder(
        base_model_type=constants.BASE_MODEL_TYPE_SIMPLE,
        embedding_type=constants.EMBEDDING_TYPE_GAUSSIAN,
        num_embedding_components=constants.num_embedding_components,
        embedding_size=constants.embedding_size,
        num_embedding_samples=constants.num_embedding_samples,
        weight_max_norm=0.0,
        feature_dim=3*dataset.keypoint_profile_2d.keypoint_num,
        seed=0)

    models.load_training_state(embedder_fn=embedder_fn, optimizer=None, scalar_parameters=scalar_parameters, ckpt_resume_path=args.resume_ckpt)        
    embedder_fn.keywords['base_model_fn'].to(device)
    embedder_fn.keywords['base_model_fn'].eval()

    raw_a = scalar_parameters['raw_a']
    raw_b = scalar_parameters['raw_b']
    print(raw_a, raw_b)        
    eval(embedder_fn=embedder_fn, loader=loader, scalar_parameters=scalar_parameters, writer=writer, device=device)