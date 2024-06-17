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
def pretrain_dreamvideo(
        model, 
        resume_checkpoint,
        sd_keys_path=None,
        grad_scale=1,
        fix_spatial_weight=False,
        fix_temporal_weight=False,
        train_adapter=False,
        **kwargs
    ):

    state_dict = torch.load(resume_checkpoint, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # [1] load model
    try:
        mismatch = model.load_state_dict(state_dict, strict=False)
        logging.info("Keys in model not matched: {}".format(mismatch[0]))
        logging.info("Keys in checkpoint not matched: {}".format(mismatch[1]))
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
        if train_adapter and "adapter" in k:
            logging.info(f"train adapter param: {k}")
        elif k in state_dict:
            total_size += p.numel()
            if fix_spatial_weight:
                p.requires_grad=False
            else:
                p.register_hook(lambda grad: grad_scale * grad)
        elif fix_temporal_weight:
            p.requires_grad=False
    
    resume_step = 0
    logging.info(f'Successfully load step {resume_step} model from {resume_checkpoint}')
    logging.info(f'load a fixed model with {int(total_size / (1024 ** 2))}M parameters')
    return model, resume_step


@PRETRAIN.register_function()
def pretrain_instructvideo(
        model,
        optimizer,
        scaler,
        resume_checkpoint,
        pretrained_image_keys,
        fix_weight,
        grad_scale,
        # cfg,
        **kwargs
    ):

    if resume_checkpoint.startswith('workspace') or resume_checkpoint.startswith('cache') or resume_checkpoint.startswith('models'):
        checkpoint_dict = torch.load(resume_checkpoint, map_location='cpu')
    else:
        assert False
    
    if 'state_dict' in checkpoint_dict:
        state_dict = checkpoint_dict['state_dict']
    else:
         state_dict = checkpoint_dict
    if 'state_dict' in checkpoint_dict:
        resume_step = checkpoint_dict['step']
    else:
        # Parse #step from the file name
        resume_step = int(resume_checkpoint.split('.')[-2].split('_')[-1])
    
    mismatch = model.module.load_state_dict(state_dict, strict=False)
    logging.info("Keys in model not matched: {}".format(mismatch[0]))
    logging.info("Keys in checkpoint not matched: {}".format(mismatch[1]))
    state_dict = json.load(open(pretrained_image_keys))

    if "optimizer" in checkpoint_dict:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint_dict["optimizer"])
        else:
            print("Resume function does not receive the 'optimizer' param.")
    else:
        print("Resumed checkpoint does not contain optimizer.")
    
    if "scaler" in checkpoint_dict:
        if scaler is not None:
            scaler.load_state_dict(checkpoint_dict["scaler"])
        else:
            print("Resume function does not receive the 'scaler' param.")
    else:
        print("Resumed checkpoint does not contain scaler.")
    
    total_size = 0
    for k, p in model.module.named_parameters():
        if k in state_dict: # spatial
            total_size += p.numel()
            if fix_weight:
                p.requires_grad=False
            else:
                if p.requires_grad:
                    p.register_hook(lambda grad: grad_scale['spatial'] * grad)
        else:               # temporal
            if fix_weight:
                p.requires_grad=False
            else:
                if p.requires_grad:
                    p.register_hook(lambda grad: grad_scale['temporal'] * grad)
    
    def compute_lora_stat(model):
        total_size = 0
        lora_size = 0
        for k, p in model.module.named_parameters():
            total_size += p.numel()
            if 'lora' in k:
                lora_size += p.numel()

        logging.info(f'Total params (Spatio-Temporal): {total_size / (1024 ** 2):.2f}M.')
        logging.info(f'LoRA params (Spatio-Temporal): {lora_size / (1024 ** 2):.2f}M.')
       
    compute_lora_stat(model)
    # INFO: Total params (Spatio-Temporal): 1347.44M.
    # INFO: LoRA params (Spatio-Temporal): 1.58M.
    logging.info(f'Successfully load step {resume_step} model from {resume_checkpoint}')
    logging.info(f'load a fixed model with {int(total_size / (1024 ** 2))}M parameters')
    return model, resume_step



@PRETRAIN.register_function()
def pretrain_from_sd():
    pass


@PRETRAIN.register_function()
def pretrain_ema_model():
    pass