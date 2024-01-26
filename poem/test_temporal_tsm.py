import pickle
import torch
import torch.nn.functional as F
import numpy as np
import time
import glob
import os
from pose_embedding.common.utils.dtw import SelfSimilarityProbDistance

from tumeke_action.models.repnet_former import MultiRepHead
from actionformer.core import load_config
from actionformer.modeling import make_meta_arch

# Use only the first head for forward propagation or other operations
def forward_with_first_head(model, features):
    return model.heads[0](features)  # use only the first head


embedding_path = "output/fullbody-occlusion-1018-0.5-0.2/embedding-ep10-repcount" # "-ep10-custom-upper/" repcount
embedding_file_list = glob.glob(os.path.join(embedding_path, "*.npy"))

al_config = "/data/junhe/ActionRecognition/configs/tumeke_repcount_pose_0523.yaml"
ckpt_file = "/data/junhe/checkpoints/repnet/tumeke_aug_salient_pose_ep250.pth.tar"
rep_head_ckpt_file = "/data/junhe/checkpoints/repnet/rephead_pose_ep250.pth.tar"

rep_cfg = load_config(al_config)

new_state_dict = torch.load(rep_head_ckpt_file,
    map_location='cpu')  
# Initialize the new model (or you can use the same model structure)
scale_range = range(min(rep_cfg['model']['backbone_arch'][-1] + 1, 2))
new_model = MultiRepHead([rep_cfg['dataset']['max_seq_len']//rep_cfg['model']['scale_factor']**i for i in scale_range])

model_dict = new_model.state_dict()
model_dict.update(new_state_dict)
new_model.load_state_dict(model_dict)

max_seq_len = rep_cfg['dataset']['max_seq_len']

# device = "cpu" # 
device = "cuda:7" # 
new_model = new_model.to(device)
for temporal_embedding_file in embedding_file_list:
    feats = np.load(temporal_embedding_file).astype(np.float32)

    sims_prob = SelfSimilarityProbDistance(device=device,
                                           path="/".join(temporal_embedding_file.split('/')[:-1]))
    t0=time.time()
    tsm_matrix = sims_prob.test(feats, max_seq_len,
                                name=temporal_embedding_file.split('/')[-1],
                                device=device)
    print(f"{feats.shape[0]}*{feats.shape[0]} TSM : {time.time()-t0:.2f} seconds (using {device}).")
    # tsm_matrix.unsqueeze_(0).unsqueeze_(0)
    # if tsm_matrix.shape[-1] < max_seq_len:
    #     # padding to [1,1,max_seq_len,max_seq_len]
    #     pad_left = pad_top = (512 - tsm_matrix.shape[-1]) // 2
    #     pad_right = pad_bottom = 512 - tsm_matrix.shape[-1] - pad_left
    #     padded_tensor = F.pad(tsm_matrix, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    # else:
    #     padded_tensor = tsm_matrix
    # with torch.no_grad():
    #     output = forward_with_first_head(new_model, padded_tensor)

    # print(output)    