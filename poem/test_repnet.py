import torch
import numpy as np

import os
import pickle

from tumeke_action.models.repnet_former import MultiRepHead
from actionformer.core import load_config
from actionformer.modeling import make_meta_arch
from actionformer.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_scheduler, AverageMeter, #make_optimizer, 
                        fix_random_seed, ModelEma)

build_rephead_flag = False

al_config = "/data/junhe/ActionRecognition/configs/tumeke_repcount_pose_0523.yaml"
ckpt_file = "/data/junhe/checkpoints/repnet/tumeke_aug_salient_pose_ep250.pth.tar"
rep_head_ckpt_file = "/data/junhe/checkpoints/repnet/rephead_pose_ep250.pth.tar"

rep_cfg = load_config(al_config)

if build_rephead_flag:
    # model
    rep_cfg['devices'] = ['cpu']
    rep_cfg['model_name'] = 'RepNetTransformer'
    rep_model = make_meta_arch(rep_cfg['model_name'], **rep_cfg['model'])
    rep_model = torch.nn.DataParallel(rep_model, device_ids=rep_cfg['devices'])    # first wrap model by using DataParallel
    print("Using model EMA ...")
    model_ema = ModelEma(rep_model)
    resume_epoch = 0  

    print(f"=> loading checkpoint {ckpt_file}")
    # load ckpt, reset epoch / best rmse
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_file,
            map_location = lambda storage, loc: storage.cuda(
                rep_cfg['devices'][0]))  
    else:
        checkpoint = torch.load(ckpt_file,
            map_location='cpu')  
    resume_epoch = checkpoint['epoch']
    rep_model.load_state_dict(checkpoint['state_dict'])
    model_ema.module.load_state_dict(checkpoint['state_dict_ema'])           
    del checkpoint

    # Extract the Count_Head weights
    count_head_weights = {k: v for k, v in rep_model.state_dict().items() if "rep_heads" in k}
    # Remove "module.rep_heads" prefix
    new_state_dict = {}
    for key, value in count_head_weights.items():
        new_key = key.replace("module.rep_heads.", "")
        new_state_dict[new_key] = value

    torch.save(new_state_dict, rep_head_ckpt_file)

new_state_dict = torch.load(rep_head_ckpt_file,
    map_location='cpu')  
# Initialize the new model (or you can use the same model structure)
scale_range = range(min(rep_cfg['model']['backbone_arch'][-1] + 1, 2))
new_model = MultiRepHead([rep_cfg['dataset']['max_seq_len']//rep_cfg['model']['scale_factor']**i for i in scale_range])

model_dict = new_model.state_dict()
model_dict.update(new_state_dict)
new_model.load_state_dict(model_dict)

# Use only the first head for forward propagation or other operations
def forward_with_first_head(model, features):
    return model.heads[0](features)  # use only the first head

input_tensor = torch.randn(1, 1, 512, 512)
with torch.no_grad():
    output = forward_with_first_head(new_model, input_tensor)

print(output)