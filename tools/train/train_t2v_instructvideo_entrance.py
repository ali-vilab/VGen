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

from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.nn.data_parallel import ShardedDataParallel
from torchvision.transforms.functional import InterpolationMode
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP

# import self packge
import utils.transforms as data
from utils.optim import AnnealingLR
from utils.distributed import generalized_all_gather, all_reduce
from utils.util import to_device
from utils.reward import DiffRewardModel

from ..modules.config import cfg
from utils.seed import setup_seed
from utils.multi_port import find_free_port
from utils.registry_class import ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION, PRETRAIN
import time
import cv2
import pdb

@ENGINE.register_function()
def t2v_instructvideo_entrance(cfg_update,  **kwargs):
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
    cfg.temp_dir = osp.join(cfg.temp_dir, exp_name)
    os.makedirs(cfg.temp_dir, exist_ok=True)
    if cfg.rank == 0:
        log_file = osp.join(cfg.log_dir, 'log.txt')
        log_local = osp.join(cfg.temp_dir, 'log.txt')
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
    
    # [data] imagedataset and videodataset
    len_frames = len(cfg.frame_lens)
    len_fps = len(cfg.sample_fps)
    cfg.max_frames = cfg.frame_lens[cfg.rank % (len_frames * len_fps)// len_fps] # 计算
    cfg.data_type = cfg.data_type[cfg.rank % (len_frames * len_fps)// len_fps]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]
    # if cfg.data_type == 'vid_diff_reward':
    #     cfg.batch_size = cfg.batch_size // 2
    #     # If we use vid_reward, the computational requirement increase.
    cfg.sample_fps = cfg.sample_fps[cfg.rank % len_fps]

    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])
    vit_trans = data.Compose([
        data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])),
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    if cfg.max_frames == 1 or cfg.data_type == 'img':
        logging.info(f"Loading a img dataset:{cfg.img_dataset['data_list']}")
        # print(f"Loading a img dataset:{cfg.img_dataset['data_list']}")
        dataset = DATASETS.build(cfg.img_dataset, transforms=train_trans, vit_transforms=vit_trans, max_frames=cfg.max_frames)
    elif cfg.data_type == 'vid':
        logging.info(f"Loading a vid dataset:{cfg.vid_dataset['data_list']}")
        # print(f"Loading a vid dataset:{cfg.vid_dataset['data_list']}")
        dataset = DATASETS.build(cfg.vid_dataset, sample_fps=cfg.sample_fps, transforms=train_trans, vit_transforms=vit_trans, max_frames=cfg.max_frames)
    elif cfg.data_type == 'vid_diff_reward':
        logging.info(f"Loading a vid_diff_reward dataset:{cfg.vid_reward_dataset['data_list']}")
        # print(f"Loading a vid_reward dataset:{cfg.vid_reward_dataset['data_list']}")
        dataset = DATASETS.build(cfg.vid_reward_dataset, sample_fps=cfg.sample_fps, transforms=train_trans, vit_transforms=vit_trans, max_frames=cfg.max_frames)
    

    sampler = None
    if cfg.world_size > 1 and cfg.debug:
       sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank)
    dataloader = DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor)
    rank_iter = iter(dataloader) 

    # [model]auotoencoder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    zero_feature = clip_encoder.zero_feature
    # white_feature = clip_encoder(clip_encoder.white_image).unsqueeze(1) # [1, 1, 1024]
    # zero_feature = torch.zeros_like(white_feature, device=white_feature.device) 

    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()

    model = MODEL.build(cfg.UNet)
    lora_params = [param for name, param in model.named_parameters() if 'lora' in name]
    print(f"There are {len(lora_params)} keys with LoRA.")
    model = model.to(gpu)
    if cfg.UNet['use_lora']:
        model = freeze_all_except_lora(model)

    print(f"There are {len([name for name, param in model.named_parameters() if param.requires_grad is True])} keys remaining to be optimized.")

    optimizer = optim.AdamW(params=model.parameters(), 
        lr=cfg.lr, weight_decay=cfg.weight_decay)

    # optimizer & acceleration
    scaler = amp.GradScaler(enabled=cfg.use_fp16)
    if cfg.use_fsdp:
        config = {}
        config['compute_dtype'] = torch.float32
        config['mixed_precision'] = True
        model = FSDP(model, **config)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model.to(gpu)
        # model = DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True) if not cfg.debug else model.to(gpu)

    # scheduler
    scheduler = AnnealingLR(
        optimizer=optimizer,
        base_lr=cfg.lr,
        warmup_steps=cfg.warmup_steps, # 10
        total_steps=cfg.num_steps, # 200000
        decay_mode=cfg.decay_mode) # 'cosine'

    resume_step = 1
    if cfg.Pretrain['resume_checkpoint'] != '':
        model, resume_step = PRETRAIN.build(cfg.Pretrain, model=model, optimizer=optimizer, scaler=scaler, grad_scale=cfg.grad_scale) # , cfg=cfg
    else:
        logging.info('Train from scratch.')
    torch.cuda.empty_cache()
    print(f"There are {len([name for name, param in model.named_parameters() if param.requires_grad is True])} keys remaining to be optimized.")

    if cfg.use_ema:
        ema = model.module.state_dict() if not cfg.debug else model.state_dict()
        ema = type(ema)([(k, ema[k].data.clone()) for k in list(ema.keys())[cfg.rank::cfg.world_size]])

    # [Diffusion]  build diffusion settings
    diffusion = DIFFUSION.build(cfg.Diffusion)
    
    # Define the reward model
    if cfg.reward_type == "HPSv2":
        reward_scorer = DiffRewardModel(
            reward_type = cfg.reward_type,
            segments = cfg.segments,
            cfg = cfg,
            autoencoder = autoencoder,
            reward_normalization = cfg.reward_normalization,
            positive_reward = cfg.positive_reward)
    else:
        print(f"{cfg.reward_type} haven't been implemented yet.")
        assert False
    
    # [Visual]
    viz_num = min(cfg.batch_size, 8)
    visual_func = VISUAL.build(
        cfg.visual_train,
        cfg_global=cfg,  
        viz_num=viz_num, 
        diffusion=diffusion, 
        autoencoder=autoencoder
    ) 
    
    for step in range(resume_step, cfg.num_steps + 1): # resume_step:1 cfg.num_steps: 200000
        # print(step)
        model.train()
        
        try:
            batch = next(rank_iter)
        except StopIteration:
            rank_iter = iter(dataloader)
            batch = next(rank_iter)
        # continue

        batch = to_device(batch, gpu, non_blocking=True)
        ref_frame, vit_frame, video_data, captions, video_key = batch
        
        batch_size, frames_num, _, _, _ = video_data.shape
        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
        
        if video_data.shape[0] % cfg.chunk_size != 0:
            if cfg.chunk_size >= video_data.shape[0]:
                cfg.chunk_size = video_data.shape[0]
            else:
                cfg.chunk_size = frames_num

        video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
        with torch.no_grad():  # test point for torch.no_grad()
            decode_data = []
            for chunk_data in video_data_list:
                # encoder_posterior = autoencoder.encode(chunk_data)
                # latent_z = get_first_stage_encoding(encoder_posterior, cfg.scale_factor).detach()
                # decode_data.append(latent_z) # [64, 4, 32, 48]
                latent_z = autoencoder.encode_firsr_stage(chunk_data, cfg.scale_factor).detach()
                decode_data.append(latent_z) # [B, 4, 32, 56]

            video_data = torch.cat(decode_data,dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = batch_size) # [80, 4, 16, 32, 48]

        ### Text conditions
        # preprocess
        with torch.no_grad():
            ### Text processing
            y = clip_encoder(captions).detach() #bs * 77 *1024 [80, 77, 1024]
            y0 = y.clone()
            zero_y = clip_encoder("").detach() # 1 * 77 * 1024
            y[torch.rand(y.size(0)) < cfg.p_zero, :] = zero_y # cfg.p_zero: 0.9

            ### Image processing
            vb, vc, vh, vw = vit_frame.size()
            vt = 1
            y_visual = clip_encoder.forward_image(vit_frame).view(vb, vt, -1) # bs * 1 *1024 [B, 1, 1024]
            y_visual_0 = y_visual.clone()
        
        # forward
        # model_kwargs={'y': y}
        

        ### Diff Training
        if cfg.use_fsdp:
            # TODO: This is left to be implemented.
            assert False
        else:
            if 'diff_reward' not in cfg.data_type:
                # TODO: To be implemented
                assert False
            else:
                t_round = torch.ones((batch_size, ), dtype=torch.long, device=gpu) * cfg.ddim_steps[-int(cfg.starting_partial*cfg.ddim_timesteps)] # int(cfg.starting_partial*cfg.num_timesteps)
                model_kwargs_ddpm = {'y': y0}
                model_kwargs_ddim = [{'y': y0},
                                {'y': zero_y.repeat(batch_size, 1, 1)}]
                with amp.autocast(enabled=cfg.use_fp16):

                    loss_recon = None
                    if cfg.share_noise:
                        b, c, f, h, w= video_data.shape
                        noise = torch.randn((b,c,h,w), device=gpu)
                        noise=noise.repeat_interleave(repeats=f, dim=0)###share noise
                        noise = rearrange(noise, '(b f) c h w->b c f h w', b=b)
                        noise = noise.contiguous()
                        if cfg.data_align_method == 'ddpm':
                            with torch.no_grad():
                                loss_recon, denoised_x0, log_prob = diffusion.loss(x0=video_data, 
                                    t=t_round, model=model, model_kwargs=model_kwargs_ddpm, noise=noise, 
                                    use_div_loss=cfg.use_div_loss)
                    elif cfg.temporal_offset_noise:
                        noise = torch.randn_like(video_data)
                        b, c, f, h, w= video_data.shape
                        offset_noise = torch.randn(b, c, f, 1, 1, device=video_data.device)
                        noise = noise + cfg.temporal_offset_noise_strength * offset_noise
                        if cfg.data_align_method == 'ddpm':
                            with torch.no_grad():
                                loss_recon, denoised_x0, log_prob = diffusion.loss(x0=video_data, 
                                        t=t_round, model=model, model_kwargs=model_kwargs_ddpm, noise=noise, 
                                        use_div_loss=cfg.use_div_loss)
                    else:
                        assert False
                        # noise = None
                        # loss_recon, denoised_x0, log_prob = diffusion.loss(x0=video_data, 
                        #         t=t_round, model=model, model_kwargs=model_kwargs_ddpm, 
                        #         use_div_loss=cfg.use_div_loss) # cfg.use_div_loss: False    loss: [80]
                         
                    

                    ### For differentiable reward
                    noise = torch.randn_like(video_data) if noise is None else noise # [80, 4, 8, 32, 32]
                    noised_image = diffusion.q_sample(x0=video_data,
                                                      t = t_round,
                                                      noise=noise)

                    gen_videos = diffusion.ddim_sample_loop_partial(
                                        noise = noised_image,
                                        model = model,
                                        starting_partial = cfg.starting_partial,
                                        grad_checkpointing = cfg.grad_checkpointing,
                                        truncated_backprop = cfg.truncated_backprop,
                                        trunc_backprop_timestep = cfg.trunc_backprop_timestep,
                                        model_kwargs = model_kwargs_ddim,
                                        guide_scale = cfg.guide_scale,
                                        ddim_timesteps = cfg.ddim_timesteps,
                                        eta = 0.0)

                    loss = reward_scorer.reward_scorer(captions = captions,
                                                    denoised_x0 = gen_videos,
                                                    # log_prob = log_prob,
                                                    loss_recon = loss_recon,
                                                    t_round = t_round)

        # backward
        if cfg.use_fsdp:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            # check_for_nan_or_inf(model)
            scaler.step(optimizer)
            scaler.update()
        
        if not cfg.use_fsdp:
            scheduler.step()

        # ema update
        if cfg.use_ema:
            for k, v in ema.items():
                v.copy_((model if cfg.debug else model.module).state_dict()[k].lerp(v, cfg.ema_decay))

        # metrics
        all_reduce(loss)
        loss = loss / cfg.world_size

        # logging
        if cfg.rank == 0 and step % cfg.log_interval == 0: # cfg.log_interval: 100
            logging.info(f'Step: {step}/{cfg.num_steps} Loss: {loss.item():.3f} scale: {scaler.get_scale():.1f} LR: {scheduler.get_lr():.7f}')

        # visualization
        if step == resume_step or step == cfg.num_steps or step % cfg.viz_interval == 0:
            with torch.no_grad():
                try:
                    # Visualization using image featues
                    # visual_func.run(model = model, ref_frame = ref_frame, vit_frame = vit_frame, 
                    #                 video_data = video_data, captions = captions, 
                    #                 y0 = y_visual_0, zero_y = zero_feature, step = step)
                    
                    # Visualization using text featues
                    lora_merged_model = get_lora_merged_model(model = model, cfg_UNet = cfg.UNet, gpu = gpu)
                    visual_func.run(model = lora_merged_model,
                                    ref_frame = ref_frame, vit_frame = vit_frame, 
                                    video_data = video_data, captions = captions, 
                                    y0 = y0, zero_y = zero_y, step = step)
                    del lora_merged_model
                except Exception as e:
                    logging.info(f'Save videos with exception {e}')
        
        # checkpoint
        if (step == cfg.num_steps or step % cfg.save_ckp_interval == 0) and resume_step != step:
            if cfg.rank == 0:
                local_key = osp.join(cfg.temp_dir, f'checkpoints/non_ema_{step:07d}.pth')
                state_dict = {
                    'state_dict': model.module.state_dict() if not cfg.debug else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'step': step}
                os.makedirs(osp.dirname(local_key), exist_ok=True)
                torch.save(state_dict, local_key)
                logging.info(f'Save model to {local_key}')
    
    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')

    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()
    

@torch.no_grad()
def viz_first_frame_latent_video(video_data, autoencoder, save_path):
    '''
        video_data: shape 'b f c h w'
    '''
    scale_factor = 0.18215 
    video_data = 1. / scale_factor * video_data
    decode_data = []
    video_num = video_data.shape[0]
    for vid in video_data:
        gen_frames = autoencoder.decode(vid) # [f, c, h, w]
        # gen_frames = autoencoder.decode(video_data[0])
        decode_data.append(gen_frames) 
    video_data = torch.cat(decode_data, dim=0)
    video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = video_num)

    vid_mean = [0.5, 0.5, 0.5]
    vid_std = [0.5, 0.5, 0.5]
    vid_mean = torch.tensor(vid_mean, device=video_data.device).view(1, -1, 1, 1, 1) # ncfhw
    vid_std = torch.tensor(vid_std, device=video_data.device).view(1, -1, 1, 1, 1) # ncfhw
    video_data.clamp_(0, 1)
    video_data = video_data * 255.0
    video_data = rearrange(video_data, 'b c f h w -> b f h w c', b = video_num)
    
    if video_data.requires_grad:
        video_data = video_data.detach().cpu().numpy()
    else:
        video_data = video_data.cpu().numpy()
    mid_index = video_data.shape[1]//2
    first_frame = Image.fromarray(np.uint8(video_data[-1][mid_index]))
    first_frame.save(save_path)


@torch.no_grad()
def viz_latent_video(video_data, autoencoder, viz_tmp_path, caption, suffix):
    '''
        video_data: shape 'b f c h w'
    '''
    scale_factor = 0.18215 
    video_data = 1. / scale_factor * video_data
    decode_data = []
    video_num = video_data.shape[0]
    for vid in video_data:
        gen_frames = autoencoder.decode(vid) # [f, c, h, w]
        # gen_frames = autoencoder.decode(video_data[0])
        decode_data.append(gen_frames) 
    video_data = torch.cat(decode_data, dim=0)
    video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = video_num)

    vid_mean = [0.5, 0.5, 0.5]
    vid_std = [0.5, 0.5, 0.5]
    vid_mean = torch.tensor(vid_mean, device=video_data.device).view(1, -1, 1, 1, 1) # ncfhw
    vid_std = torch.tensor(vid_std, device=video_data.device).view(1, -1, 1, 1, 1) # ncfhw
    video_data.clamp_(0, 1)
    video_data = video_data * 255.0
    video_data = rearrange(video_data, 'b c f h w -> b f h w c', b = video_num)
    
    if video_data.requires_grad:
        video_data = video_data.detach().cpu().numpy()
    else:
        video_data = video_data.cpu().numpy()
    
    # [(img.numpy()).astype('uint8') for img in images]

    for video, cap in zip(video_data, caption):
        video = [img.astype('uint8') for img in video]
        save_fps = 8
        local_path = os.path.join(viz_tmp_path, f'{obtain_time()}_' + cap.replace(' ', '_') + f'_{suffix}.mp4')

        # if os.path.exists(local_path):
        #     os.makedirs(local_path)
        frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
        os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
        for fid, frame in enumerate(video):
            tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
            cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
        os.system(cmd); os.system(f'rm -rf {frame_dir}')
    
    # mid_index = video_data.shape[1]//2
    # first_frame = Image.fromarray(np.uint8(video_data[-1][mid_index]))
    # first_frame.save(save_path)


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
                print(f"Gradient for {name}")
                # print(f"Gradient available for {name}")
                has_nan = torch.isnan(param.grad).any()
                has_inf = torch.isinf(param.grad).any()
                if has_nan or has_inf:
                    print(f"Gradient problem in {name}: NaN: {has_nan}, Inf: {has_inf}")
            else:
                print(f"No gradient for {name}")


def freeze_all_except_lora(model):
    if hasattr(model, 'module'):
        tmp_model = 'model.module'
    else:
        tmp_model = 'model'
    for name, param in eval(tmp_model).named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def freeze_all(model):
    if hasattr(model, 'module'):
        tmp_model = 'model.module'
    else:
        tmp_model = 'model'
    for name, param in eval(tmp_model).named_parameters():
        param.requires_grad = False
    return model


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

        logging.info(f'We merge {merged_keys*2} lora keys to prepare for equivalent inference.')
        ### Remove 'module.'
        no_module_base_model_dict = {}
        for name, param in base_model_dict.items():
            no_module_base_model_dict[name.replace('module.', '')] = param
        base_model.load_state_dict(no_module_base_model_dict)

        return base_model
    else:
        return model