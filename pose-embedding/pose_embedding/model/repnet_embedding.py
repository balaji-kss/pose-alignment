import math
import time
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from pose_embedding.model.repnet import RepNetHead
from pose_embedding.common import loss_utils, constants, models, keypoint_profiles


class SelfSimilarityProbDistance(torch.nn.Module):
    def __init__(self, device, raw_a=4.0335, raw_b=14.0885, smoothing=0.1, chunk_size=128, 
                 temperature=13.544, path="data/output/vis"):
        super(SelfSimilarityProbDistance, self).__init__()
        self.temperature = temperature
        self.path = path
        if isinstance(raw_a, float) or isinstance(raw_b, float):            
            raw_a = torch.Tensor([raw_a]).to(device)
            raw_b = torch.Tensor([raw_b]).to(device)
        else:
            raw_a = raw_a.to(device)
            raw_b = raw_b.to(device)

        sigmoid_a, sigmoid_b = loss_utils.get_sigmoid_parameters( 
            raw_a=raw_a, raw_b=raw_b,
            a_range=(None, constants.sigmoid_a_max))          
        self.sigmoid_a=sigmoid_a
        self.sigmoid_b=sigmoid_b
        self.smoothing = smoothing
        self.chunk_size = chunk_size

    def fill_tensor(self, A, B, mask):
        # Ensure B is of the same dtype as A
        B = B.to(dtype=A.dtype)

        # Find the positions in A that need to be replaced
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # Replace the values in A at the mask positions with the values from B
        A[mask_indices] = B.flatten()

        return A  # This is optional, as A is modified in-place

    def forward(self, X_batch, mask_1d_batch, temperature=13.544):
        # x in shape of [B, T, seq, samples * dim]: [B, T, 8, 20 * 16]
        # T is padded as power of 2, eg: 512, 1024, 2048, ... 
        tsm_list = []
        for i in range(X_batch.shape[0]):
            X = X_batch[i]
            N = X.shape[0]
            # Expand the 1-D mask to 2-D
            mask_1d = mask_1d_batch[i]
            mask_2d = mask_1d.unsqueeze(1) & mask_1d.unsqueeze(0)

            num_chunks = N // self.chunk_size
            tsm = torch.zeros([N,N], device=X.device)

            row_indices, col_indices = np.meshgrid(np.arange(num_chunks), np.arange(num_chunks))

            for (r, c) in zip(row_indices.ravel(), col_indices.ravel()):
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()                
                # Expand dimensions
                a_exp = X[r*self.chunk_size:(r+1)*self.chunk_size].unsqueeze(1).unsqueeze(2)  # [N, 1, 1, 8, 320]
                b_exp = X[c*self.chunk_size:(c+1)*self.chunk_size].unsqueeze(0).unsqueeze(3)  # [1, N, 8, 1, 320]
                # Compute sequence-to-sequence distances
                distances = loss_utils.probabilistic_distance_torch(a_exp, b_exp, self.sigmoid_a, self.sigmoid_b)  # This should be [N, N, 8, 8]
                # Average over the sequence length dimensions to get the sequence-to-sequence distances
                distances = torch.mean(distances, dim=[-1,-2])  # This will be [N, N]
                tsm[r*self.chunk_size:(r+1)*self.chunk_size, c*self.chunk_size:(c+1)*self.chunk_size] = distances
            # set padding data as very large number, eg 1e10
            tsm[~mask_2d] = 1e10
            # 11/01/2023 add tsm sample-wise normalization
            valid_entries = tsm[mask_2d]
            tsm_mean      = torch.mean(valid_entries)
            tsm_std       = torch.std(valid_entries)
            tsm[mask_2d]  = (tsm[mask_2d] - tsm_mean) / tsm_std
            # Apply the softmax function
            tsm = F.softmax(-tsm / temperature, dim=-1)
            tsm[~mask_2d] = 0
            tsm_list.append(tsm)
        tsm_tensor = torch.stack(tsm_list)
        return tsm_tensor

    
    def forward_mask(self, X_batch, mask_1d_batch, temperature=13.544):
        # x in shape of [B, T, seq, samples * dim]: [B, T, 8, 20 * 16]
        # T is padded as power of 2, eg: 512, 1024, 2048, ... 
        tsm_list = []
        for i in range(X_batch.shape[0]):
            X = X_batch[i]
            N = X.shape[0]
            mask_1d = mask_1d_batch[i]
            num_valid = torch.sum(mask_1d)
            # Expand the 1-D mask to 2-D
            mask_2d = mask_1d.unsqueeze(1) & mask_1d.unsqueeze(0)

            num_chunks = N // self.chunk_size
            tsm = torch.zeros([N,N], device=X.device)

            row_indices, col_indices = np.meshgrid(np.arange(num_chunks), np.arange(num_chunks))

            for (r, c) in zip(row_indices.ravel(), col_indices.ravel()):
                # if torch.cuda.is_available():
                #     torch.cuda.empty_cache()                
                # Expand dimensions
                a_exp = X[r*self.chunk_size:(r+1)*self.chunk_size].unsqueeze(1).unsqueeze(2)  # [N, 1, 1, 8, 320]
                b_exp = X[c*self.chunk_size:(c+1)*self.chunk_size].unsqueeze(0).unsqueeze(3)  # [1, N, 8, 1, 320]
                # Compute sequence-to-sequence distances
                distances = loss_utils.probabilistic_distance_torch(a_exp, b_exp, self.sigmoid_a, self.sigmoid_b)  # This should be [N, N, 8, 8]
                # Average over the sequence length dimensions to get the sequence-to-sequence distances
                distances = torch.mean(distances, dim=[-1,-2])  # This will be [N, N]
                tsm[r*self.chunk_size:(r+1)*self.chunk_size, c*self.chunk_size:(c+1)*self.chunk_size] = distances
            # Apply the softmax function
            tsm[~mask_2d] = 0.0
            tsm_soft = F.softmax(-tsm[mask_2d].reshape([num_valid, num_valid])  / temperature, dim=-1)
            self.fill_tensor(tsm, tsm_soft, mask_2d)
            tsm_list.append(tsm)
        tsm_tensor = torch.stack(tsm_list)
        return tsm_tensor

    def save_tsm_fig(self, tsm, path, name, start, end):
        import matplotlib.pyplot as plt
        import cv2
        import os

        fig, axes = plt.subplots(figsize=(10, 6))
        bk_sim_img = tsm.cpu().detach().numpy()

        image_eq = ((bk_sim_img - bk_sim_img.min()) * (255/(bk_sim_img.max()-bk_sim_img.min()))).astype(np.uint8)
        X = image_eq
        X = cv2.equalizeHist(image_eq)
        
        # Use imshow to display X as an image
        axes.imshow(X, cmap='jet')
        axes.set_title(f"{name}") 

        plt.savefig(os.path.join(path, f"{name}_[{start}-{end}]_sim.png"), format='png')
        plt.close(fig)

    """ 
    embeddings is a dict {meta_info: {fps, raw_a, raw_b}, embeddings: [list of embedding_dict]}
    """
    def test_by_second(self, embeddings, win_size=0.5, stride=0.25, name="test", start=0, end=20, device="cuda"):
        fps = embeddings['meta_info']['fps']
        stride = int(fps * stride)
        win_size_frames = min(8, int(fps * win_size))  # For high fps, we use 8 frames per window
        cand_s, cand_e = 0, int(fps * win_size)
        seq_embedding_list = []
        embedding_list = embeddings['embeddings']
        # t0 = time.time()
        while cand_e < len(embedding_list) and len(seq_embedding_list) < end:
            sequence_b = embedding_list[cand_s:cand_e]
            sequence_b = np.stack([em[constants.KEY_EMBEDDING_SAMPLES] for em in sequence_b], axis=0)
            # down-sampling the template if pose_win_template > alignment_win
            if cand_e-cand_s > win_size_frames:
                sampling_idx = np.linspace(0, cand_e-cand_s-1, win_size_frames).astype(int)
                sequence_b = sequence_b[sampling_idx]      

            sequence_b = sequence_b.reshape(sequence_b.shape[0], -1)
            seq_embedding_list.append(torch.Tensor(sequence_b))

            cand_s += stride
            cand_e += stride
        # print(f"Matched {candidate_vid.frame_cnt/candidate_vid.fps:.2f} seconds video cost {time.time() - t0: .4f} seconds")
        seq_embedding = torch.stack(seq_embedding_list, dim=0)
        seq_embedding = seq_embedding.to(device=device)
        parts_embedding = seq_embedding[start:end]
        t0=time.time()
        pad_emb = parts_embedding.new_full([512, *parts_embedding.shape[1:]], 0.0)
        pad_emb[:parts_embedding.shape[0]].copy_(parts_embedding)
        # generate the mask
        mask = torch.arange(512) < parts_embedding.shape[0]

        tsm_matrix = self.forward(pad_emb, mask) #  parts_embedding

        print(f"forward costs: {time.time() - t0:.4f} seconds on {device}")
        self.save_tsm_fig(tsm_matrix,path=self.path, name=name, start=start, end=end)
        return tsm_matrix
    
    def test(self, embeddings, max_seq_len=512, name="test", device="cuda"):
        embeddings = torch.Tensor(embeddings).to(device)
        pad_emb = embeddings.new_full([max_seq_len, *embeddings.shape[1:]], 0.0)
        pad_emb[:embeddings.shape[0]].copy_(embeddings)
        # generate the mask
        mask = torch.arange(max_seq_len) < embeddings.shape[0]

        t0=time.time()
        tsm_matrix = self.forward(pad_emb, mask)
        print(f"forward costs: {time.time() - t0:.4f} seconds on {device}")
        self.save_tsm_fig(tsm_matrix,path=self.path, name=name, start=0, end=max_seq_len)
        return tsm_matrix


class CrossSimilarityProbDistance(SelfSimilarityProbDistance):
    def __init__(self, device, raw_a=4.0335, raw_b=14.0885, smoothing=0.1, path='data/output/vis'):
        super(CrossSimilarityProbDistance, self).__init__(device, raw_a, raw_b, smoothing, path)
    
    def forward(self, X, Y):
        Nx = X.shape[0] # X # [Nx, 8, 320]
        Ny = Y.shape[0] # Y # [Ny, 8, 320]

        a_exp = X.unsqueeze(1).unsqueeze(2)  # [Nx, 1, 1, 8, 320]
        b_exp = Y.unsqueeze(0).unsqueeze(3)  # [1, Ny, 8, 1, 320]
        # Compute sequence-to-sequence distances
        distances = loss_utils.probabilistic_distance_torch(a_exp, b_exp, self.sigmoid_a, self.sigmoid_b, self.smoothing)  # This should be [N, N, 8, 8]
        # Average over the sequence length dimensions to get the sequence-to-sequence distances
        distances = torch.mean(distances, dim=[-1,-2])  # This will be [N, N]
        
        return distances
    
class CrossFrameSimilarityProbDistance(SelfSimilarityProbDistance):
    def __init__(self, device, raw_a=4.0335, raw_b=14.0885, smoothing=0.1, path='data/output/vis'):
        super(CrossFrameSimilarityProbDistance, self).__init__(device, raw_a, raw_b, smoothing, path)
    
    def forward(self, X, Y):
        Nx = X.shape[0] # X # [Nx, 8, 320]
        Ny = Y.shape[0] # Y # [Ny, 8, 320]

        a_exp = X.unsqueeze(1).unsqueeze(2)  # [Nx, 1, 1, 8, 320]
        b_exp = Y.unsqueeze(0).unsqueeze(3)  # [1, Ny, 8, 1, 320]
        # Compute sequence-to-sequence distances
        distances = loss_utils.probabilistic_distance_torch(a_exp, b_exp, self.sigmoid_a, self.sigmoid_b, self.smoothing)  # This should be [N, N, 8, 8]
        # Average over the sequence length dimensions to get the sequence-to-sequence distances
        distances = torch.mean(distances, dim=[-1,-2])  # This will be [N, N]
        
        return distances    

class RepNetEmbedding(nn.Module):
    """
        Temporal Pose Embedding based RepNet
    """
    def __init__(
        self,
        max_seq_len,           # max sequence length (used for training)
        embedding_ckpt,        # loading embedding ekpt for raw_a and raw_b parameters
        device,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_div_factor = 1

        # We only need sigmoid parameters when a related distance kernel is used.
        raw_a = torch.nn.Parameter(torch.tensor(constants.sigmoid_raw_a_initial, device=device))
        raw_b = torch.nn.Parameter(torch.tensor(constants.sigmoid_b_initial, device=device))
        scalar_parameters = dict(raw_a=raw_a, raw_b=raw_b, epoch=0, epoch_step=0)
        keypoint_profile_2d = keypoint_profiles.create_keypoint_profile("LEGACY_2DCOCO13")

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
        
        embedder_fn.keywords['base_model_fn'].eval()

        models.load_training_state(embedder_fn=embedder_fn, optimizer=None, scalar_parameters=scalar_parameters, ckpt_resume_path=embedding_ckpt)        
        raw_a = scalar_parameters['raw_a']
        raw_b = scalar_parameters['raw_b']

        self.sims = SelfSimilarityProbDistance(device=device, raw_a=raw_a, raw_b=raw_b)

        self.rep_head = RepNetHead(max_seq_len=max_seq_len, device=device)

        # MSE is not a suitable loss for this scenario because only a fraction of range supports which is a sparse recovery problem
        self.lossMAE = torch.nn.SmoothL1Loss().cuda(device=device) if torch.cuda.is_available() else torch.nn.SmoothL1Loss() 
        self.lossBCE = torch.nn.BCEWithLogitsLoss().cuda(device=device) if torch.cuda.is_available() else torch.nn.BCEWithLogitsLoss()

        self.use_count_error = True

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def get_tsm(self, video_list, scale_range):
        # batch the video list into feats (B, T, S, C) and masks (B, T, 1)
        feats, masks = self.preprocessing(video_list)

        sim_output = []

        # goint to fix 
        for i in scale_range:
            x = F.relu(self.sims(feats, masks))
            sim_output.append(x)

        return dict(sims=sim_output, masks=masks)            

    def forward(self, video_list, scale_range=None):
        if scale_range is not None:
            return self.get_tsm(video_list, scale_range)
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        feats, masks = self.preprocessing(video_list)

        sim_output = []
        pred_period, pred_periodicity = [], []

        with torch.no_grad():
            x = F.relu(self.sims(feats, masks))
        sim_output.append(x)

        x = x.unsqueeze(dim=1)

        p_period, p_periodicity = self.rep_head(x)

        pred_period.append(p_period)
        pred_periodicity.append(p_periodicity)

        # generate segment/lable List[N x 2] / List[N] with length = B
        assert video_list[0]['segments'] is not None, "GT action labels does not exist"
        assert video_list[0]['labels'] is not None, "GT action labels does not exist"
        gt_segments = [x['segments'].to(self.device) for x in video_list]
        gt_labels = [x['labels'].to(self.device) for x in video_list]

        gt_period, gt_periodicity = self.label_points(
            gt_segments, gt_labels)

        # return loss during training
        if self.training:
            # compute the loss and return
            losses = 0
            losses = self.losses(masks,
                pred_period, pred_periodicity,
                gt_period, gt_periodicity
            )
            losses.update(sims=sim_output, pred_period=pred_period, pred_periodicity=pred_periodicity,
                           gt_periodicity=gt_periodicity, gt_period=gt_period, masks=masks)
            # losses.update(sim_output=sim_output,pred_period=pred_period, pred_periodicity=pred_periodicity)
            return losses
        else:
            results = self.inference(
                video_list, masks,
                pred_period, pred_periodicity,
                gt_period, gt_periodicity
            )
            results.update(sims=sim_output, pred_period=pred_period, pred_periodicity=pred_periodicity,
                           gt_periodicity=gt_periodicity, gt_period=gt_period, masks=masks)
            return results            

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[0] for feat in feats]) # feat : T x S x D
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape [B, T, S, D]
            batch_shape = [len(feats), max_len, feats[0].shape[1], feats[0].shape[2]] # [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[:feat.shape[0], ...].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

            pad_feat = feats[0].new_full([max_len, *feats[0].shape[1:]], 0.0)
            pad_feat[:feats[0].shape[0]].copy_(feats[0])
            batched_inputs = pad_feat.unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.to(self.device)# batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    def losses(self, masks,
            pred_period, pred_periodicity,
            gt_period, gt_periodicity):
        # masks: list of different scale of masks, each is [N, 1, T//stride]
        # pred_period : list of different scale prediction, each is [N, T//stride,1]
        # pred_periodicity: list of different scale prediction, each is [N, T//stride,1]
        # gt_period: list of concat of different scale gt for each video, [960] * N
        # gt_periodicity: list of concat of different scale gt for each video, [960] * N

        gt_period = torch.stack(gt_period).to(self.device)  # [N, 960]
        gt_periodicity = torch.stack(gt_periodicity).to(self.device)   # [N, 960]
        valid_masks = masks # torch.cat(masks[:len(pred_period)], dim=2).squeeze(dim=1)    # [N, 1, 960] --> [N, 960]

        # count_error = []
        # gt_period_cur = gt_period[:, :len(masks[-1,:])]
        # gt_periodicity_cur = gt_periodicity[:, :len(masks[-1, :])]
        # countpred = torch.sum((pred_periodicity[0][masks] > 0) / (pred_period[0][masks] + 1e-1), dim=0).squeeze()
        # count = torch.sum((gt_periodicity_cur[masks] > 0) / (gt_period_cur[masks] + 1e-1), dim=0)
        # count_error.append(self.lossMAE(count, countpred) / gt_period.shape[0])
            # count_error.append(torch.div(torch.abs(countpred - count), (count + 1e-1)))

        pred_period = torch.cat(pred_period, dim=1).squeeze(dim=2) # [N, 960,1] --> [N, 960]
        pred_periodicity = torch.cat(pred_periodicity, dim=1).squeeze(dim=2) # [N, 960,1] --> [N, 960]

        # loss_mae = self.lossMAE(pred_period[valid_masks], gt_period[valid_masks]) # use mask 11/01/2023
        loss_mae = self.lossMAE(pred_period, gt_period) # 
        losses = [torch.nn.functional.binary_cross_entropy_with_logits(p[m], t[m]) for p, t, m in zip(pred_periodicity, gt_periodicity, valid_masks)]
        loss_bce = torch.mean(torch.stack(losses))
        # loss_bce = self.lossBCE(pred_periodicity[valid_masks], gt_periodicity[valid_masks])        

        # countpred = torch.sum((pred_periodicity[valid_masks] > 0) / (pred_period[valid_masks] + 1e-1), dim=0)
        # count = torch.sum((gt_periodicity[valid_masks] > 0) / (gt_period[valid_masks] + 1e-1), dim=0)
        countpred = self.count_by_masking(periodicity=pred_periodicity, period=pred_period, valid_masks=valid_masks)
        count = self.count_by_masking(periodicity=gt_periodicity, period=gt_period, valid_masks=valid_masks)

        # loss_count = torch.sum(torch.div(torch.abs(countpred - count), (count + 1e-1)))
        loss_count = self.lossMAE(count, countpred)

        final_loss = loss_mae + 10*loss_bce
        if self.use_count_error:
            final_loss += 0.01 * loss_count
        return dict(loss_mae=loss_mae, loss_bce=loss_bce, final_loss=final_loss, loss_count=loss_count,
                    countpred=[countpred], count=[count], gt_periodicity_out=[gt_periodicity])

    def count_by_masking(self, periodicity, period, valid_masks):
        masked_gt_a = torch.where(valid_masks, periodicity, torch.tensor(0.0, device=periodicity.device))
        masked_gt_b = torch.where(valid_masks, period, torch.tensor(1e-1, device=period.device))
        count = torch.sum((masked_gt_a > 0) / (masked_gt_b + 1e-1), dim=1)
        return count
            
    @torch.no_grad()
    def inference(self, video_list, masks,
                pred_period, pred_periodicity,
                gt_period, gt_periodicity):
        gt_period = torch.stack(gt_period).to(self.device)  # [N, 960]
        gt_periodicity = torch.stack(gt_periodicity).to(self.device)   # [N, 960]

        # decode each scale of ground truth
        st = 0
        preds, gts, gt_periodicity_out = [], [], []
        for i in range(len(pred_period)):
            mask = masks
            gt_period_cur = gt_period[:,st:st+len(mask[-1,:])]
            gt_periodicity_cur = gt_periodicity[:,st:st+len(mask[-1, :])]
            st += len(mask[-1, :])
            countpred = torch.sum((pred_periodicity[i][mask] > 0) / (pred_period[i][mask] + 1e-1), dim=0)
            count = torch.sum((gt_periodicity_cur[mask] > 0) / (gt_period_cur[mask] + 1e-1), dim=0)
            preds.append(countpred)
            gts.append(count)
            gt_periodicity_out.append(gt_periodicity_cur)
        # valid_masks = torch.cat(masks[:len(pred_period)], dim=2).squeeze(dim=1)    # [N, 1, 960] --> [N, 960]
        # pred_period = torch.cat(pred_period, dim=1).squeeze(dim=2) # [N, 960,1] --> [N, 960]
        # pred_periodicity = torch.cat(pred_periodicity, dim=1).squeeze(dim=2) # [N, 960,1] --> [N, 960]
        # countpred = torch.sum((pred_periodicity[valid_masks] > 0) / (pred_period[valid_masks] + 1e-1), dim=0)
        # count = torch.sum((gt_periodicity[valid_masks] > 0) / (gt_period[valid_masks] + 1e-1), dim=0)
        return dict(countpred=preds, count=gts, gt_periodicity_out=gt_periodicity_out)

    @torch.no_grad()
    def label_points(self, gt_segments, gt_labels):
        # concat points on all fpn levels List[T x 4] -> F T x 4
        # This is shared for all samples in the mini-batch
        # num_levels = len(points)
        # concat_points = torch.cat(points, dim=0)
        gt_period, gt_periodicity = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            period_targets, periodicity_targets = self.label_period_single_video(
                gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_period.append(period_targets)
            gt_periodicity.append(periodicity_targets)

        return gt_period, gt_periodicity
    
    @torch.no_grad()
    def label_period_single_video(self, gt_segment, gt_label):
        # label the period and periodicity for one video
        # input: 
        #   gt_segment [[start, end]; [start, end]; ...; [start,end]]: [k, 2]
        #   gt_label [count, count, count, ..., count]: [k]
        # output: 
        #   period: [...512...256...128...64...] of pooled period for each level
        #   periodicity: [...512...256...128...64...] of pooled periodicity for each level
        periods = []
        period = torch.zeros(self.max_seq_len)
        for i in range(gt_segment.shape[0]):
            st, ed = int(gt_segment[i, 0]), int(gt_segment[i, 1])
            st = max(0, st)
            period[st:ed] = (ed - st) / gt_label[i]
        periods.append(period)
        period_targets = torch.cat(periods, dim=0)
        periodicity_targets = self.getPeriodicity(period_targets)

        return period_targets, periodicity_targets

    @torch.no_grad()
    def getPeriodicity(self, periodLength):
        periodicity = torch.nn.functional.threshold(periodLength, 2, 0)
        periodicity = -torch.nn.functional.threshold(-periodicity, -1, -1)
        return periodicity	