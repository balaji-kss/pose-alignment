import pickle
import torch
import torch.nn.functional as F
import numpy as np
import time
import glob
import os
import mmcv
import copy as cp
from pose_embedding.common.utils.dtw import SelfSimilarityProbDistance
from pose_embedding.common import constants
from pose_embedding.api.temporal_embedding_feature import extract_embedding_temporal_feature
from pose_embedding.common.default_config import load_config

from pose_embedding.model.repnet_embedding import RepNetEmbedding

tumeke_action_root = '/data/junhe/ActionRecognition/'
repcount_cfg = os.path.join(tumeke_action_root, 'configs/tumeke_repcount_embedding_1025.yaml')
rep_ckpt = os.path.join(tumeke_action_root, f"data/repcount/embedding-rep-aug-rephead-as-paper/epoch_400.pth.tar")

dataset_path = '/data/junhe/data/gpu-compute-output/2people-scraping'
embedding_path = os.path.join(dataset_path, "")
embedding_file_list = glob.glob(os.path.join(embedding_path, "*.pkl"))
feature_store_path = os.path.join(dataset_path, "feature-store")
track_id = 1
mmcv.mkdir_or_exist(feature_store_path)

# load repcount model
trunk_size = 512 # 256
rep_cfg = load_config(repcount_cfg)
rep_cfg['model_name'] = 'RepNetEmbedding'
rep_model = RepNetEmbedding(max_seq_len=rep_cfg['model']['max_seq_len'], 
                            embedding_ckpt=rep_cfg['model']['embedding_ckpt'],
                            device=rep_cfg['devices'][0])
rep_model = torch.nn.DataParallel(rep_model, device_ids=rep_cfg['devices'])    # first wrap model by using DataParallel

# load ckpt, reset epoch / best rmse
if torch.cuda.is_available():
    checkpoint = torch.load(rep_ckpt,
        map_location = lambda storage, loc: storage.cuda(
            rep_cfg['devices'][0]))  
else:
    checkpoint = torch.load(rep_ckpt,
        map_location='cpu')  
rep_model.load_state_dict(checkpoint['state_dict'])   
rep_model.eval() 

for candidate_embedding_file in embedding_file_list:
    with open(candidate_embedding_file,'rb') as f:
        embeddings = pickle.load(f)

        seq_embedding = extract_embedding_temporal_feature(embeddings, track_id=track_id)

        name = candidate_embedding_file.split('/')[-1].split('.')[0]
        temporal_emb_file = os.path.join(feature_store_path, f"{name}_{track_id}.npy")
        print(f"{temporal_emb_file}: temporal embedding length {seq_embedding.shape[0]}, FPS={embeddings['meta_info']['fps']:.2f}")
        np.save(temporal_emb_file, seq_embedding)

        with torch.no_grad():
            seq_embedding_tensor = torch.from_numpy(seq_embedding)
            data_dict = {'video_id'       : "test",
                        # 'feats'           : seq_embedding_tensor,      # C x T ==> T x S x D 
                        'segments'        : torch.Tensor([[0,1]]),       # N x 2
                        'labels'          : torch.Tensor([1]),           # N
                        'fps'             : embeddings['meta_info']['fps'],     # 'fps' is unused in model part
                        'feat_stride'     : 8,           # 'feat_stride' is unused in model part
                        'feat_num_frames' : 12}          # 'feat_num_frames' is unused in model part

            # Keep the length of temporal feature smaller than 256 which is better for our model            
            trunks = seq_embedding.shape[0] // trunk_size    
            results = []
            for i in range(trunks+1):
                if i == trunks:
                    data_dict['feats'] = seq_embedding_tensor[trunk_size*i:]
                else:
                    data_dict['feats'] = seq_embedding_tensor[trunk_size*i : trunk_size*(i+1)]
                result = rep_model([data_dict])
                results.append(result)

            msg = ""
            count_pred = 0
            for result in results:
                pred_count = result['countpred'][0].cpu().detach().numpy()
                count_pred += float(pred_count)
            print(f"{name}: estimate {count_pred:.2f} repetitions")
