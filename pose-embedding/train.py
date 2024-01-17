import torch
from torch.utils.data import DataLoader
from pose_embedding.common import visualization_utils
import matplotlib.pyplot as plt
import time
import os
import argparse
import mmcv
from torch.utils.tensorboard import SummaryWriter
from pose_embedding.common import models, keypoint_utils, loss_utils, data_utils, constants
from pose_embedding.dataset.pose_pairs_dataset import PosePairsDataset
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = argparse.ArgumentParser(description='Pose embedding model training.')
    parser.add_argument(
        '--writer',
        default='runs/model-train/occlusion_1018_version', 
        help='Tensorboard writer logs')
    parser.add_argument(
        '--keypoint-dropout-rate', type=float,  nargs='*', default=[0.3, 0.2], 
                    help="keypoint dropout rate [apply_rate_instance, drop_out_rate_keypoint]")
    parser.add_argument(
        '--ckpt-path',
        default="./checkpoints/fullbody-occlusion-1018",
        help='Checkpoint path')
    parser.add_argument(
        '--dataset',
        default="./data/chunks/*.h5",
        help='training dataset path of hdf5 files')
    parser.add_argument(
        '--val-dataset',
        default="./data/test-chunks/*.h5",
        help='validation dataset path of hdf5 files')    
    parser.add_argument(
        '--resume-ckpt',
        default= None, # './checkpoints/training_model_0_600.pth', #
        help='resume training from previous checkpoint')   
    parser.add_argument(
        '--eval-interval', type=int, default=30000, help='During one epoch, the interval of doing evaluation')
    parser.add_argument(
        '--gpu-id', type=int, default=7, help='CUDA device id')
    parser.add_argument(
        '--max-epoch', type=int, default=10, help='Total epoch number for training') 
    parser.add_argument(
        '--batch-size', type=int, default=256, help='batch size') 
    parser.add_argument(
        '--num-workers', type=int, default=4, help='number of workers')     
    args = parser.parse_args()   

    return args 

def train_one_epoch(loader, optimizer, device, writer, ep, global_step):
    t0 = time.time()
    epoch_step = scalar_parameters['epoch_step']
    for i, data in tqdm(enumerate(loader)):
        if i + epoch_step > loader.dataset.__len__()//loader.batch_size:
            break
        optimizer.zero_grad()
        summaries = {}
        data['model_inputs']=data['model_inputs'].to(device)
        data[constants.KEY_KEYPOINTS_3D]=data[constants.KEY_KEYPOINTS_3D].to(device)
        data[constants.KEY_PREPROCESSED_KEYPOINT_MASKS_2D] = data[constants.KEY_PREPROCESSED_KEYPOINT_MASKS_2D].to(device)

        sigmoid_a, sigmoid_b = loss_utils.get_sigmoid_parameters( 
            raw_a=raw_a,
            raw_b=raw_b,
            a_range=(None, constants.sigmoid_a_max))

        embedding_sample_distance_fn_kwargs.update({
            'L2_SIGMOID_MATCHING_PROB_a': sigmoid_a,
            'L2_SIGMOID_MATCHING_PROB_b': sigmoid_b,
            'SQUARED_L2_SIGMOID_MATCHING_PROB_a': sigmoid_a,
            'SQUARED_L2_SIGMOID_MATCHING_PROB_b': sigmoid_b,
        })    
        triplet_embedding_sample_distance_fn = loss_utils.create_sample_distance_fn(
                        pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
                        distance_kernel=constants.triplet_distance_kernel,
                        pairwise_reduction=constants.triplet_pairwise_reduction,
                        componentwise_reduction=constants.triplet_componentwise_reduction,
                        **embedding_sample_distance_fn_kwargs)
        positive_pairwise_embedding_sample_distance_fn = loss_utils.create_sample_distance_fn(
                        pair_type=constants.DISTANCE_PAIR_TYPE_ALL_PAIRS,
                        distance_kernel=constants.positive_pairwise_distance_kernel,
                        pairwise_reduction=constants.positive_pairwise_pairwise_reduction,
                        componentwise_reduction=(
                            constants.positive_pairwise_componentwise_reduction),
                        **embedding_sample_distance_fn_kwargs)

        outputs, activations = embedder_fn(data['model_inputs'])

        # Prepare keypoint trilet loss
        # anchor_keypoint_masks_3d, positive_keypoint_masks_3d = None, None
        anchor_keypoint_masks_2d, positive_keypoint_masks_2d = torch.unbind(
                        data[constants.KEY_PREPROCESSED_KEYPOINT_MASKS_2D],
                        dim=1)
        anchor_keypoint_masks_3d = keypoint_utils.transfer_keypoint_masks(
            anchor_keypoint_masks_2d,
            input_keypoint_profile=dataset.keypoint_profile_2d, #configs['keypoint_profile_2d'],
            output_keypoint_profile=dataset.keypoint_profile_3d, #configs['keypoint_profile_3d'],
            enforce_surjectivity=False)
        positive_keypoint_masks_3d = keypoint_utils.transfer_keypoint_masks(
            positive_keypoint_masks_2d,
            input_keypoint_profile=dataset.keypoint_profile_2d, #configs['keypoint_profile_2d'],
            output_keypoint_profile=dataset.keypoint_profile_3d, #configs['keypoint_profile_3d'],
            enforce_surjectivity=False)
    
        anchor_keypoints_3d = data[constants.KEY_KEYPOINTS_3D][:, 0, ...]
        positive_keypoints_3d = data[constants.KEY_KEYPOINTS_3D][:, 1, ...]

        triplet_anchor_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 0, ...]
        triplet_positive_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 1, ...]

        triplet_anchor_mining_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 0, ...]
        triplet_positive_mining_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 1, ...]

        positive_pairwise_anchor_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 0, ...]
        positive_pairwise_positive_embeddings = outputs[constants.KEY_EMBEDDING_SAMPLES][:, 1, ...]
        
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

        # Adds KL regularization loss.
        kl_regularization_loss, kl_regularization_loss_summaries = (
            loss_utils.compute_kl_regularization_loss(
                outputs[constants.KEY_EMBEDDING_MEANS],
                stddevs=outputs[constants.KEY_EMBEDDING_STDDEVS],
                prior_stddev=constants.kl_regularization_prior_stddev,
                loss_weight=constants.kl_regularization_loss_weight))    

        summaries.update(kl_regularization_loss_summaries)

        positive_pairwise_loss, positive_pairwise_loss_summaries = (
            loss_utils.compute_positive_pairwise_loss(
                positive_pairwise_anchor_embeddings,
                positive_pairwise_positive_embeddings,
                loss_weight=constants.positive_pairwise_loss_weight,
                distance_fn=positive_pairwise_embedding_sample_distance_fn
                )
            )
        summaries.update(positive_pairwise_loss_summaries)
        dist = positive_pairwise_embedding_sample_distance_fn(positive_pairwise_anchor_embeddings, positive_pairwise_positive_embeddings)

        total_loss = triplet_loss + kl_regularization_loss + positive_pairwise_loss
        total_loss.backward()
        optimizer.step()

        summaries['train/total_loss'] = total_loss
        for key, value in summaries.items():
            writer.add_scalar(key, value, global_step + i)

        # if i % 500 == 0:
        #     fig = visualization_utils.draw_poses_2d(
        #         data_utils.flatten_first_dims(
        #             data[constants.KEY_PREPROCESSED_KEYPOINTS_2D][:constants.draw_anchor_positive_number,...],
        #             num_last_dims_to_keep=2),
        #         keypoint_profile_2d=dataset.keypoint_profile_2d,
        #         num_cols=2)
        #     # Log figure to TensorBoard
        #     visualization_utils.plot_to_tensorboard(writer, fig, step=i+global_step, tag="my_plot")

        #     # Don't forget to close the plot to free up resources
        #     plt.close(fig)

        if i % 100 == 0:
            writer.add_histogram('Pairwise_distances', dist, i+global_step)            
            print(f"Epoch {ep}-{i}: {time.time() - t0 :.3f} secs. triplet loss: {triplet_loss:.4f}, positive_pairwise_loss loss: {positive_pairwise_loss:.4f}")
            t0 = time.time()

        if i % 500 == 0:
            for k in activations:
                writer.add_histogram(k, activations[k], i+global_step)                       
            for k in outputs:
                writer.add_histogram(k, outputs[k], i+global_step)    

        if i > 0 and i % args.eval_interval == 0:            
            scalar_parameters['epoch_step'] =  i + epoch_step
            models.save_training_state(embedder_fn=embedder_fn, optimizer=optimizer, scalar_parameters=scalar_parameters, ckpt_path=args.ckpt_path)
            # embedder_fn.keywords['base_model_fn'].eval()
            # # do eval on test dataset and save model
            # embedder_fn.keywords['base_model_fn'].train()


if __name__ == '__main__':
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Initialize TensorBoard writer
    args.ckpt_path = args.ckpt_path + f"-{args.keypoint_dropout_rate[0]}-{args.keypoint_dropout_rate[1]}"
    mmcv.mkdir_or_exist(args.ckpt_path)
    args.writer = args.writer + f"-{args.keypoint_dropout_rate[0]}-{args.keypoint_dropout_rate[1]}"
    writer = SummaryWriter(args.writer)
    # To view TensorBoard, run `tensorboard --logdir=runs` in the terminal

    t0 = time.time()
    dataset = PosePairsDataset(args.dataset, keypoint_profile_2d_name='LEGACY_2DCOCO13', 
                               keypoint_dropout_rate=args.keypoint_dropout_rate)
    val_dataset = PosePairsDataset(args.val_dataset, keypoint_profile_2d_name='LEGACY_2DCOCO13')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"Dataset preparation cost {time.time() - t0 : .2f} seconds")

    embedding_sample_distance_fn_kwargs = {
        'EXPECTED_LIKELIHOOD_min_stddev': constants.min_stddev,
        'EXPECTED_LIKELIHOOD_max_squared_mahalanobis_distance': constants.max_squared_mahalanobis_distance,
    }
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
        dropout_rate=constants.dropout_rate,
        seed=0)
    
    embedder_fn.keywords['base_model_fn'].to(device)

    learning_rate = models.get_learning_rate(
                    constants.learning_rate_schedule,   # ""
                    constants.learning_rate,            # 0.02
                    decay_steps=constants.num_steps,    # 5,000,000
                    num_warmup_steps=constants.num_warmup_steps)
    optimizer = models.get_optimizer(
                    constants.optimizer_name.upper(), learning_rate=learning_rate, 
                    model_parameters=[raw_a, raw_b] + list(embedder_fn.keywords['base_model_fn'].parameters())
                    )
    
    if args.resume_ckpt is not None:
        models.load_training_state(embedder_fn=embedder_fn, optimizer=optimizer, scalar_parameters=scalar_parameters, ckpt_resume_path=args.resume_ckpt)
        raw_a = scalar_parameters['raw_a']
        raw_b = scalar_parameters['raw_b']
        print('Resume training parameters: ', raw_a, raw_b, scalar_parameters['epoch'], scalar_parameters['epoch_step'], '\n')
        
    while scalar_parameters['epoch'] < args.max_epoch:
        global_step = scalar_parameters['epoch']*(dataset.__len__()//args.batch_size) + scalar_parameters['epoch_step'] + 1
        train_one_epoch(loader=loader, writer=writer, optimizer=optimizer, 
                        device=device, ep=scalar_parameters['epoch'], 
                        global_step=global_step)
        scalar_parameters['epoch'] += 1
        scalar_parameters['epoch_step'] = 0
        models.save_training_state(embedder_fn=embedder_fn, optimizer=optimizer, scalar_parameters=scalar_parameters, ckpt_path=args.ckpt_path)
        