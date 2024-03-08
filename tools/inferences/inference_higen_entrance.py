'''
/* 
*Copyright (c) 2021, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/
'''

import os
import re
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import torch
import pynvml
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.cuda.amp as amp
from importlib import reload
import torch.distributed as dist
import torch.multiprocessing as mp

from einops import rearrange
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.nn.parallel import DistributedDataParallel

import utils.transforms as data
from ..modules.config import cfg
from utils.seed import setup_seed
from utils.multi_port import find_free_port
from utils.assign_cfg import assign_signle_cfg
from utils.distributed import generalized_all_gather, all_reduce
from utils.video_op import save_i2vgen_video, save_t2vhigen_video_safe
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION


@INFER_ENGINE.register_function()
def inference_higen_entrance(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) 
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    
    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg, cfg_update)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, cfg_update))
    return cfg


def worker(gpu, cfg, cfg_update):
    '''
    Inference worker for each gpu
    '''
    cfg = assign_signle_cfg(cfg, cfg_update, 'vldm_cfg')
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg.gpu = gpu
    cfg.seed = int(cfg.seed)
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    setup_seed(cfg.seed + cfg.rank)

    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # [Log] Save logging and make log dir
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    exp_name = osp.basename(cfg.test_list_path).split('.')[0]
    inf_name = osp.basename(cfg.cfg_file).split('.')[0]
    test_model = osp.basename(cfg.test_model).split('.')[0].split('_')[-1]
    
    cfg.log_dir = osp.join(cfg.log_dir, '%s' % (exp_name))
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = osp.join(cfg.log_dir, 'log_%02d.txt' % (cfg.rank))
    cfg.log_file = log_file
    reload(logging)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(filename=log_file),
            logging.StreamHandler(stream=sys.stdout)])
    logging.info(cfg)
    logging.info(f"Going into inference_text2video_entrance inference on {gpu} gpu")
    
    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Data] Data Transform    
    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])

    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    _, _, zero_y = clip_encoder(text="")
    zero_y = zero_y.detach()

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    
    # [Model] UNet 
    model = MODEL.build(cfg.UNet)
    state_dict = torch.load(cfg.test_model, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if 'step' in state_dict:
        resume_step = state_dict['step']
    else:
        resume_step = 0
    status = model.load_state_dict(state_dict, strict=True)
    logging.info('Load model from {} with status {}'.format(cfg.test_model, status))
    model = model.to(gpu)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model
    torch.cuda.empty_cache()
    
    # [Test List]
    test_list = open(cfg.test_list_path).readlines()
    test_list = [item.strip() for item in test_list]
    num_videos = len(test_list)
    logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    test_list = [item for item in test_list for _ in range(cfg.round)]
    
    for idx, caption in enumerate(test_list):
        if caption.startswith('#'):
            logging.info(f'Skip {caption}')
            continue
        if '|' in caption:
            caption, manual_seed = caption.split('|')
            manual_seed = int(manual_seed)
        else:
            manual_seed = 0
        logging.info(f"[{idx}]/[{num_videos}] Begin to sample {caption} ...")
        if caption == "": 
            logging.info(f'Caption is null of {caption}, skip..')
            continue

        captions = [caption]
        with torch.no_grad():
            _, y_text, y_words = clip_encoder(text=captions) # bs * 1 *1024 [B, 1, 1024]

        with torch.no_grad():
            pynvml.nvmlInit()
            handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
            logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')
            # sample images (DDIM)
            with amp.autocast(enabled=cfg.use_fp16):
                setup_seed(cfg.seed + cfg.rank + idx % cfg.round + manual_seed)
                logging.info(f"Setup seed to {cfg.seed + cfg.rank + idx % cfg.round + manual_seed} ...")
                # setup_seed(cfg.seed + cfg.rank + idx + manual_seed)
                # logging.info(f"Setup seed to {cfg.seed + cfg.rank + idx} ...")
                cur_seed = torch.initial_seed()
                logging.info(f"Current seed {cur_seed} ...")

                spat_noise = torch.randn([1, 4, 1, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)]).to(gpu)
                spat_prior = torch.zeros_like(spat_noise).squeeze(2)
                motion_cond = torch.tensor([0],dtype=torch.long, device=gpu)
                appearance_cond = torch.Tensor([[[1.0]]]).repeat(1, 1, max(cfg.frame_lens)).to(gpu)
                model_kwargs=[
                    {'y': y_words, 'spat_prior': spat_prior, 'motion_cond': motion_cond, "appearance_cond": appearance_cond},
                    {'y': zero_y, 'spat_prior': spat_prior, 'motion_cond': motion_cond, "appearance_cond": appearance_cond}]
                
                spat_data = diffusion.ddim_sample_loop(
                    noise=spat_noise,
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                
                spat_key_frames = autoencoder.decode(1. / cfg.scale_factor * spat_data.squeeze(2))
                spat_data = spat_data.squeeze(2)

                temp_noise = torch.randn([1, 4, cfg.max_frames, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)]).to(gpu)
                b, c, f, h, w= temp_noise.shape
                offset_noise = torch.randn(b, c, f, 1, 1, device=gpu)
                temp_noise = temp_noise + cfg.noise_strength * offset_noise
                
                motion_cond = torch.tensor([[cfg.motion_factor] * (cfg.max_frames-1)], dtype=torch.long, device=gpu)

                sim_list = torch.cat([torch.linspace(1.0-cfg.appearance_factor, 1.0, cfg.max_frames)[:-1], torch.linspace(1.0, 1.0-cfg.appearance_factor, cfg.max_frames)])
                # sim_list = (torch.cos(sim_list * 3.1415926 + 3.1415926) + 1) / 2 # consine
                appearance_cond = torch.stack([sim_list[i:i+cfg.max_frames] for i in range(len(sim_list)-cfg.max_frames, -1, -1)]).to(gpu)
                # appearance_cond = CLIPSim().load_vid_sim('/mnt/data-nas-workspace/qingzhiwu/code/video_generation/workspace/temp_dir/cvpr2024_1.vidldm_15_pub_midj_unclip_basemodel_img_text_e003_eval_725000_pikachu_turn_back_g12/sample_000001/cvpr2024_1.vidldm_15_pub_midj_unclip_basemodel_img_text_e003_eval_725000_pikachu_turn_back_g12_s01_diff_0.0_500_.mp4')
                model_kwargs=[
                    {'y': y_words, 'spat_prior': spat_data, 'motion_cond': motion_cond, 'appearance_cond': appearance_cond[None, :]},
                    {'y': zero_y, 'spat_prior': spat_data, 'motion_cond': motion_cond, 'appearance_cond': appearance_cond[None, :]}]
                video_data = diffusion.ddim_sample_loop(
                    noise=temp_noise,
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
        
        video_data = 1. / cfg.scale_factor * video_data # [1, 4, 32, 46]
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
        chunk_size = min(cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
        decode_data = []
        for vd_data in video_data_list:
            gen_frames = autoencoder.decode(vd_data)
            decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = cfg.batch_size)
        # video_data = torch.cat([spat_key_frames[:, :, None, :, :], video_data], dim=2)
        
        text_size = cfg.resolution[-1]
        cap_name = re.sub(r'[^\w\s]', '', caption).replace(' ', '_')
        file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{idx:04d}_{cap_name}.mp4'
        local_path = os.path.join(cfg.log_dir, f'{file_name}')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            save_t2vhigen_video_safe(local_path, video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            logging.info('Save video to dir %s:' % (local_path))
        except Exception as e:
            logging.info(f'Step: save text or video error with {e}')
    
    logging.info('Congratulations! The inference is completed!')
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()
