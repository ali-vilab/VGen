# ------------------------------------------------------------------------
# InstructVideo: Instructing Video Diffusion Models with Human Feedback
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# ----------------------------- Notice -----------------------------------
# If you find it useful, please consider citing InstructVideo.
# ------------------------------------------------------------------------

import os
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import logging
import json
import math
import random
import torch
import pynvml
import datetime
import open_clip
import itertools
import numpy as np
from PIL import Image
import torch.optim as optim
import torch.cuda.amp as amp
from importlib import reload
from copy import deepcopy, copy
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from io import BytesIO
from easydict import EasyDict
from functools import partial
from einops import rearrange
from collections import defaultdict
import torchvision.transforms as T
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# import self packge
import utils.transforms as data
from utils.distributed import generalized_all_gather, all_reduce

from ..modules.config import cfg
from utils.seed import setup_seed
from utils.multi_port import find_free_port
from utils.registry_class import INFER_ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION
from utils.video_op import save_video_local
import time


@INFER_ENGINE.register_function()
def inference_instructvideo_entrance(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) # 0
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    setup_seed(cfg.seed)

    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg


def worker(gpu, cfg):
    '''
    Training workder for each gpu
    '''
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    
    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # [logging] Save logging
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    exp_name = '.'.join(osp.dirname(cfg.cfg_file).split("/")[1:] + [osp.basename(cfg.cfg_file).split('.')[0]]) + '_' + obtain_time()
    cfg.log_dir = osp.join(cfg.log_dir, exp_name)
    cfg.webvid_dir_save = cfg.webvid_dir_save + str(cfg.webvid_test_caps)
    os.makedirs(cfg.webvid_dir_save, exist_ok=True)
    if cfg.rank == 0:
        log_file = osp.join(cfg.log_dir, 'log.txt')
        log_local = osp.join(cfg.webvid_dir_save, 'log.txt')
        cfg.log_file = log_file
        cfg.log_local = log_local
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=log_local),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)
    
    cfg.batch_size = 1
    cfg.max_frames = max(cfg.frame_lens)
    # # [data] imagedataset and videodataset
    # len_frames = len(cfg.frame_lens)
    # len_fps = len(cfg.sample_fps)
    # cfg.max_frames = cfg.frame_lens[cfg.rank % (len_frames * len_fps)// len_fps] # 计算
    # cfg.data_type = cfg.data_type[cfg.rank % (len_frames * len_fps)// len_fps]
    # cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]
    # if cfg.data_type == 'vid_reward':
    #     cfg.batch_size = cfg.batch_size // 2
    #     # If we use vid_reward, the computational requirement increase.
    # cfg.sample_fps = cfg.sample_fps[cfg.rank % len_fps]

    # train_trans = data.Compose([
    #     data.CenterCropWide(size=cfg.resolution),
    #     data.ToTensor(),
    #     data.Normalize(mean=cfg.mean, std=cfg.std)])
    vit_trans = data.Compose([
        data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])),
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    # [model]auotoencoder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    zero_feature = clip_encoder.zero_feature.to(gpu)
    
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()

    model = MODEL.build(cfg.UNet)
    model = model.to(gpu)
    model.eval()
    
    load_dict = torch.load(cfg.infer_checkpoint, map_location='cpu')
    if 'state_dict' in load_dict.keys():
        load_dict = load_dict['state_dict']
    load_info = model.load_state_dict(load_dict, strict=True)
    print(f"We are loading from {cfg.infer_checkpoint}, which outputs {load_info}.")
    
    if cfg.UNet['use_lora']:
        model = get_lora_merged_model(model = model, cfg_UNet = cfg.UNet, gpu = gpu)
        print(f'Use_lora for the inference model: {model.use_lora}')
    
    torch.cuda.empty_cache()

    
    # betas = beta_schedule('linear_sd', cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120)
    # diffusion = ops.GaussianDiffusionReward(
    #     betas=betas,
    #     mean_type=cfg.mean_type,
    #     var_type=cfg.var_type,
    #     loss_type=cfg.loss_type,
    #     rescale_timesteps=False)

    # [Diffusion]  build diffusion settings
    diffusion = DIFFUSION.build(cfg.Diffusion)
    
    caps_dict = {}
    with open(os.path.join(cfg.webvid_dir, cfg.webvid_cap_file, f'{cfg.webvid_eval_text}.txt'), 'r') as f:
        line = f.readline()
        while(line):
            video_path, video_cap, _ = line.split('|||')
            video_path = video_path.split('/')[-1]
            caps_dict[video_path] = video_cap 
            line = f.readline()
    print(caps_dict)

    num_videos = len(caps_dict)
    logging.info(f'There are {num_videos} videos.')

    saved_videos = []
    for idx, (vid, cap) in enumerate(caps_dict.items()):
        if idx >= cfg.webvid_test_caps:
            break

        logging.info(f"[{idx}]/[{num_videos}] {cap}")

        ### Embed the caption
        captions = [cap.strip() + cfg.suffix]
        with torch.no_grad():
            ### Embeddins texts
            y = clip_encoder(captions).detach() # bs * 77 *1024 [80, 77, 1024]
            y0 = y.clone()
            zero_y = clip_encoder("").detach() # 1 * 77 * 1024

            target_h, target_w = cfg.resolution[1]//8, cfg.resolution[0]//8
            with amp.autocast(enabled=cfg.use_fp16):
                if cfg.share_noise:
                    # b, c, f, h, w= video_data.shape
                    b, c, f, h, w = cfg.batch_size, 4, cfg.max_frames, target_h, target_w
                    noise = torch.randn((b,c,h,w), device=gpu)
                    noise=noise.repeat_interleave(repeats=f, dim=0) ###share noise
                    noise = rearrange(noise, '(b f) c h w->b c f h w', b=b)
                    noise = noise.contiguous()
                elif cfg.temporal_offset_noise:
                    b, c, f, h, w = cfg.batch_size, 4, cfg.max_frames, target_h, target_w
                    noise = torch.randn((b, c, f, h, w), device=gpu)
                    offset_noise = torch.randn(b, c, f, 1, 1, device=gpu)
                    noise = noise + cfg.temporal_offset_noise_strength * offset_noise
                else:
                    b, c, f, h, w = cfg.batch_size, 4, cfg.max_frames, target_h, target_w
                    noise = torch.randn((b, c, f, h, w), device=gpu)
                    print('Do not use shared noise and temporal offset noise.')

                pynvml.nvmlInit()
                handle=pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
                logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')

                ### Text conditions
                model_kwargs = [
                    {'y': y0},
                    {'y': zero_y}
                ]
                gen_videos = diffusion.ddim_sample_loop(
                    noise = noise,
                    model = model,
                    model_kwargs = model_kwargs,
                    guide_scale = cfg.guide_scale,
                    ddim_timesteps = cfg.ddim_timesteps,
                    eta = 0.0)
        
        gen_videos = 1. / cfg.scale_factor * gen_videos # [1, 4, 32, 46]
        gen_videos = rearrange(gen_videos, 'b c f h w -> (b f) c h w')
        chunk_size = min(cfg.decoder_bs, gen_videos.shape[0])
        gen_videos_list = torch.chunk(gen_videos, gen_videos.shape[0]//chunk_size, dim=0)
        decode_prior = []
        for vd_data in gen_videos_list:
            gen_frames = autoencoder.decode(vd_data)
            decode_prior.append(gen_frames)
        gen_videos = torch.cat(decode_prior, dim=0)
        gen_videos = rearrange(gen_videos, '(b f) c h w -> b c f h w', b = cfg.batch_size)

        text_size = cfg.resolution[-1]
        # file_name = f'{idx:04d}_rank_{cfg.world_size:02d}-{cfg.rank:02d}-{cfg.seed:06d}.mp4'
        file_name = f"{idx:04d}_{vid.replace('.mp4', '')}_{cap.replace(' ', '-')}.mp4"
        oss_key = os.path.join(cfg.log_dir, f'{file_name}')
        local_path = os.path.join(cfg.webvid_dir_save, f'{file_name}')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            video_numpy_list = save_video_local(local_path, gen_videos.cpu(), cfg.mean, cfg.std)
            video_tensor = torch.stack([torch.from_numpy(img) for img in video_numpy_list])
            saved_videos.append(video_tensor)
            logging.info(f'Save video to {oss_key}')
        except Exception as e:
            logging.info(f'Step: save text or video error with {e}')
        

    logging.info('Congratulations! The inference is completed!')

    saved_path = os.path.join(cfg.webvid_dir_save, cfg.webvid_dir_save.split('/')[-1] + '.pt')
    torch.save(torch.stack(saved_videos), saved_path)


def obtain_time():
    cur_time = time.localtime()
    month = str(cur_time.tm_mon).zfill(2)
    day = str(cur_time.tm_mday).zfill(2)
    hour = str(cur_time.tm_hour).zfill(2)
    minute = str(cur_time.tm_min).zfill(2)
    str_time = f'{month}{day}-{hour}-{minute}'
    return str_time


def check_for_nan_or_inf(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:  # Check if gradient is not None
                # print(f"Gradient available for {name}")
                has_nan = torch.isnan(param.grad).any()
                has_inf = torch.isinf(param.grad).any()
                if has_nan or has_inf:
                    print(f"Gradient problem in {name}: NaN: {has_nan}, Inf: {has_inf}")
            else:
                print(f"No gradient for {name}")


def get_lora_merged_model(model, cfg_UNet, gpu):
    # base_model: model without LoRA
    # model: model with LoRA

    if cfg_UNet['use_lora']:
        lora_cfg = deepcopy(cfg_UNet)
        lora_cfg['use_lora'] = False
        # lora_cfg['lora_rank'] = None
        # lora_cfg['lora_alpha'] = None
        base_model = MODEL.build(lora_cfg) 
        base_model = base_model.to(gpu).eval()
        # base_model = freeze_all(base_model)
        base_model_dict = base_model.state_dict()
        model_dict = model.state_dict()
        alpha = 1.0 if lora_cfg['lora_alpha'] is None else lora_cfg["lora_alpha"] / lora_cfg["lora_rank"]

        # Merge lora parameters
        # pdb.set_trace()
        ### Copy the base parameters
        merged_keys = 0
        for name, param in model_dict.items():
            if 'lora' not in name:
                base_model_dict[name] = deepcopy(param)
        
        ### Merge the LoRA parameters to the base
        for name, param in model_dict.items():
            if 'lora' in name:
                # Only process lora down key
                if "up_linear" in name: continue
                lora_weight_up_name = name.replace("down_linear", "up_linear")
                lora_weight_down = deepcopy(model_dict[name])
                # print('lora_weight_down ', lora_weight_down.requires_grad)
                lora_weight_up   = deepcopy(model_dict[lora_weight_up_name])
                if 'to_out' in name:
                    # 'to_out' is defined using "Sequential", thus, we need to add '0.'.
                    attn_param_name = name.replace('_lora', '.0').replace("down_linear.", "").replace("up_linear.", "")
                else:
                    attn_param_name = name.replace('_lora', '').replace("down_linear.", "").replace("up_linear.", "")
                
                base_model_dict[attn_param_name] += alpha * torch.mm(lora_weight_up, lora_weight_down)
                merged_keys += 1
                # logging.info(f'merge lora parameter |{name}| and |{lora_weight_up_name}| to attn parameter |{attn_param_name}|')
                # 'input_blocks.0.1.transformer_blocks.0.attn1.to_q_lora.down_linear.weight', 
                # 'input_blocks.0.1.transformer_blocks.0.attn1.to_q_lora.up_linear.weight'

        logging.info(f'We merge {merged_keys*2} lora keys to prepare for equivalent inference.')
        ### Remove 'module.'
        no_module_base_model_dict = {}
        for name, param in base_model_dict.items():
            no_module_base_model_dict[name.replace('module.', '')] = param
        base_model.load_state_dict(no_module_base_model_dict)

        return base_model
    else:
        return model
