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
from utils.video_op import save_i2vgen_video, save_i2vgen_video_safe
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION
from diffusers.schedulers import LCMScheduler, DDIMScheduler, DDPMScheduler

@INFER_ENGINE.register_function()
def inference_videolcm_entrance(cfg_update,  **kwargs):
    
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
    
    

    # [Data] Data Transform    
    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])
    
    vit_trans = data.Compose([
        data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])),
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    # [Model] embedder
    # from ipdb import set_trace; set_trace()
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    _, _, zero_y = clip_encoder(text="")
    _, _, zero_y_negative = clip_encoder(text=cfg.negative_prompt)
    zero_y, zero_y_negative = zero_y.detach(), zero_y_negative.detach()

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()

    
    # [Model] UNet 
    if "config" in cfg.UNet:
        cfg.UNet["config"] = cfg
    model = MODEL.build(cfg.UNet)
    state_dict = torch.load(cfg.test_model, map_location='cpu')
    if 'state_dict' in state_dict:
        resume_step = state_dict['step']
        state_dict = state_dict['state_dict']
    else:
        resume_step = 0
    status = model.load_state_dict(state_dict, strict=True)
    logging.info('Load model from {} with status {}'.format(cfg.test_model, status))
    model = model.to(gpu)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model
    torch.cuda.empty_cache()

    # [Diffusion]
    # diffusion = DIFFUSION.build(cfg.Diffusion)
    diffusion = LCMScheduler(prediction_type="v_prediction", beta_schedule= "scaled_linear", clip_sample=False, timestep_spacing="linspace", rescale_betas_zero_snr=True)
    num_inference_steps = 4
    if hasattr(cfg, "num_inference_steps"):
        num_inference_steps = cfg.num_inference_steps
    # num_inference_steps = 4
    # DDIM
    # diffusion.config.timestep_spacing = "linspace"

    diffusion.set_timesteps(num_inference_steps, device=model.device)
    
    # [Test List]
    test_list = open(cfg.test_list_path).readlines()
    test_list = [item.strip() for item in test_list]
    num_videos = len(test_list)
    logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    test_list = [item for item in test_list for _ in range(cfg.round)]
    
    for idx, caption in enumerate(test_list):
        setup_seed(cfg.seed + cfg.rank)
        if caption.startswith('#'):
            logging.info(f'Skip {caption}')
            continue
        logging.info(f"[{idx}]/[{num_videos}] Begin to sample {caption} ...")
        if caption == "": 
            logging.info(f'Caption is null of {caption}, skip..')
            continue

        captions = [caption+cfg.positive_prompt]
        # captions = [caption]
        with torch.no_grad():
            _, y_text, y_words = clip_encoder(text=captions) # bs * 1 *1024 [B, 1, 1024]
        fps_tensor =  torch.tensor([cfg.target_fps], dtype=torch.long, device=gpu)

        with torch.no_grad():
            pynvml.nvmlInit()
            handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
            logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')
            # sample images (DDIM)
            with amp.autocast(enabled=cfg.use_fp16):
                cur_seed = torch.initial_seed()
                logging.info(f"Current seed {cur_seed} ...")
                noise = torch.randn([1, 4, cfg.max_frames, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)])
                noise = noise.to(gpu)
                diffusion._step_index = None
                timesteps = diffusion.timesteps
                do_classifier_free_guidance = False
                batch_size = noise.shape[0]

                model_kwargs=[
                    {'y': y_words, 'fps': fps_tensor},
                    {'y': zero_y_negative, 'fps': fps_tensor}]
                # video_data = diffusion.ddim_sample_loop(
                #     noise=noise,
                #     model=model.eval(),
                #     model_kwargs=model_kwargs,
                #     guide_scale=cfg.guide_scale,
                #     ddim_timesteps=cfg.ddim_timesteps,
                #     eta=0.0)
                latents = noise
                # with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    # latents = noise_one
                    latent_model_input = latents # torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = diffusion.scale_model_input(latent_model_input, t)

                    
                    noise_pred_text =  model(latent_model_input, t.repeat(batch_size).to(device=latents.device, dtype=latents.dtype), t_w=None, **model_kwargs[0])

                    # perform guidance
                    if do_classifier_free_guidance:
                        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred_uncond = model(latent_model_input, t.repeat(batch_size).to(device=latents.device, dtype=latents.dtype), t_w=None, **model_kwargs[1])
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_text

                    # if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    extra_step_kwargs = {}
                    latents = diffusion.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    
                video_data = latents
        
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
        
        text_size = cfg.resolution[-1]
        cap_name = re.sub(r'[^\w\s]', '', caption).replace(' ', '_')
        file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{idx:04d}_{cap_name}.mp4'
        local_path = os.path.join(cfg.log_dir, f'{file_name}')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            save_i2vgen_video_safe(local_path, video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            logging.info('Save video to dir %s:' % (local_path))
        except Exception as e:
            logging.info(f'Step: save text or video error with {e}')
    
    logging.info('Congratulations! The inference is completed!')
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

