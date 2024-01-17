import torch
import numpy as np
import time
from torch.nn import functional as F
from pose_embedding.common import constants, loss_utils



""" Code for temporal similarity search
"""
def temporal_averaging(dist_matrix, kernel, rate):
    # Modify the kernel to account for the dilation rate
    if rate > 1:
        dilated_kernel = np.zeros(1 + (len(kernel)-1) * rate)
        dilated_kernel[::rate] = kernel
    else:
        dilated_kernel = kernel
    
    # Apply convolution
    return np.apply_along_axis(lambda m: np.convolve(m, dilated_kernel, mode='same'), axis=0, arr=dist_matrix)

"""  distance_fn is similar to following
def probabilistic_distance(A, B, sigmoid_a=1, sigmoid_b=0):
    # Reshape A and B to [20, 16]
    A_reshaped = A.reshape(20, 16)
    B_reshaped = B.reshape(20, 16)

    # Compute pairwise L2 distances
    distances = np.linalg.norm(A_reshaped[:, np.newaxis] - B_reshaped, axis=2)
    
    # Apply the scaled sigmoid function
    distances = scaled_sigmoid(distances, sigmoid_a, sigmoid_b)
    
    # Return the mean distance
    # return -np.mean(distances, axis=(-2, -1))
    return -np.log(np.mean(distances, axis=(-2, -1)))
"""
# def compute_pairwise_distances(sequence_a, sequence_b, distance_fn):
#     distances = np.zeros((len(sequence_a), len(sequence_b)))
#     for i in range(len(sequence_a)):
#         for j in range(len(sequence_b)):
#             distances[i, j] = distance_fn(sequence_a[i], sequence_b[j])
#     return distances

def compute_pairwise_distances(sequence_a, sequence_b, distance_fn):
    # Expand dimensions to use broadcasting
    a_exp = sequence_a[:, np.newaxis, :]
    b_exp = sequence_b[np.newaxis, :, :]
    
    # Apply distance function
    distances = distance_fn(a_exp, b_exp)
    
    return distances

def dtw_with_precomputed_distances(D):
    N, M = D.shape
    cost = np.zeros((N, M))
    cost[0, 0] = D[0, 0]
    
    for i in range(1, N):
        cost[i, 0] = cost[i-1, 0] + D[i, 0]
    for j in range(1, M):
        cost[0, j] = cost[0, j-1] + D[0, j]
    for i in range(1, N):
        for j in range(1, M):
            cost[i, j] = np.min([cost[i-1, j], cost[i, j-1], cost[i-1, j-1]]) + D[i, j]

    # Extracting the warping path
    i, j = N-1, M-1
    path = [(i, j)]
    
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            argmin_cost = np.argmin([cost[i-1, j], cost[i, j-1], cost[i-1, j-1]])
            if argmin_cost == 0:
                i -= 1
            elif argmin_cost == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
    path.reverse()
    return cost[N-1, M-1], path


class SelfSimilarityProbDistance(torch.nn.Module):
    def __init__(self, device, raw_a=4.0335, raw_b=14.0885, chunk_size=128, 
                 temperature=13.544, path="data/output/vis"):
        super(SelfSimilarityProbDistance, self).__init__()
        self.temperature = temperature
        self.path = path
        raw_a = torch.Tensor([raw_a]).to(device)
        raw_b = torch.Tensor([raw_b]).to(device)
        sigmoid_a, sigmoid_b = loss_utils.get_sigmoid_parameters( 
            raw_a=raw_a, raw_b=raw_b,
            a_range=(None, constants.sigmoid_a_max))          
        self.sigmoid_a=sigmoid_a
        self.sigmoid_b=sigmoid_b
        self.chunk_size = chunk_size

    def fill_tensor(self, A, B, mask):
        # Ensure B is of the same dtype as A
        B = B.to(dtype=A.dtype)

        # Find the positions in A that need to be replaced
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # Replace the values in A at the mask positions with the values from B
        A[mask_indices] = B.flatten()

        return A  # This is optional, as A is modified in-place
    
    def forward_direct_eatup_memory(self, X, mask_1d, temperature=13.544):
        # x in shape of [N, seq, samples * dim]: [N, 8, 20 * 16]
        # N will be padded as power of 2, eg: 512, 1024, 2048, ... 
        N = X.shape[0]
        # Expand the 1-D mask to 2-D
        num_valid = torch.sum(mask_1d)
        mask_2d = mask_1d.unsqueeze(1) & mask_1d.unsqueeze(0)

        # Expand dimensions
        a_exp = X.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, 8, 320]
        b_exp = X.unsqueeze(0).unsqueeze(3)  # [1, N, 8, 1, 320]

        # Compute sequence-to-sequence distances
        distances = loss_utils.probabilistic_distance_torch(a_exp, b_exp, self.sigmoid_a, self.sigmoid_b)  # This should be [N, N, 8, 8]

        # Average over the sequence length dimensions to get the sequence-to-sequence distances
        # distances = -torch.log(pairwise_seq_distances.mean(dim=(2,3)))  # This will be [N, N]
        distances = torch.mean(distances, dim=[-1,-2])

        # distances = distances.unsqueeze(1)  # insert a new dimension [n,s,s] --> [n,1,s,s]
        # Apply the softmax function
        distances[~mask_2d] = 0.0
        distances_soft = F.softmax(-distances[mask_2d].reshape([num_valid, num_valid]) / temperature, dim=-1)
        
        self.fill_tensor(distances, distances_soft, mask_2d)

        return distances

    def forward(self, X, mask_1d, temperature=13.544):
        # x in shape of [N, seq, samples * dim]: [N, 8, 20 * 16]
        # N will be padded as power of 2, eg: 512, 1024, 2048, ... 
        N = X.shape[0]
        num_valid = torch.sum(mask_1d)
        # Expand the 1-D mask to 2-D
        mask_2d = mask_1d.unsqueeze(1) & mask_1d.unsqueeze(0)

        num_chunks = N // self.chunk_size
        tsm = torch.zeros([N,N], device=X.device)

        row_indices, col_indices = np.meshgrid(np.arange(num_chunks), np.arange(num_chunks))

        for (r, c) in zip(row_indices.ravel(), col_indices.ravel()):
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

        return tsm

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