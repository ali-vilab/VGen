import os
import json
import torch
import logging
import collections

from utils.registry_class import PRETRAIN

@PRETRAIN.register_function()
def pretrain_specific_strategies(
        model, 
        resume_checkpoint,
        sd_keys_path=None,
        grad_scale=1,
        fix_weight=False,
        **kwargs
    ):

    state_dict = torch.load(resume_checkpoint, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # [1] load model
    try:
        ret = model.load_state_dict(state_dict, strict=False)
        logging.info(f'load a fixed model with {ret}')
    except:
        model_dict = model.state_dict()
        key_list = list(state_dict.keys())
        for skey, item in state_dict.items():
            if skey not in model_dict:
                logging.info(f'Skip {skey}')
                continue
            if item.shape != model_dict[skey].shape:
                logging.info(f'Skip {skey} with different shape {item.shape} {model_dict[skey].shape}')
                continue
            model_dict[skey].copy_(item)
        model.load_state_dict(model_dict)
    
    # [2] assign strategies
    total_size = 0
    state_dict = {} if sd_keys_path is None else json.load(open(sd_keys_path))
    for k, p in model.named_parameters():
        if k in state_dict:
            total_size += p.numel()
            if fix_weight:
                p.requires_grad=False
            else:
                p.register_hook(lambda grad: grad_scale * grad)
    
    resume_step = int(os.path.basename(resume_checkpoint).split('_')[-1].split('.')[0])
    logging.info(f'Successfully load step {resume_step} model from {resume_checkpoint}')
    logging.info(f'load a fixed model with {int(total_size / (1024 ** 2))}M parameters')
    return model, resume_step



@PRETRAIN.register_function()
def pretrain_from_sd():
    pass


@PRETRAIN.register_function()
def pretrain_ema_model():
    pass
