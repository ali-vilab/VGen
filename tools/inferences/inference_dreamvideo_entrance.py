import os
import re
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import random
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
from utils.registry_class import INFER_ENGINE, MODEL, EMBEDDER, AUTO_ENCODER, DIFFUSION, EMBEDMANAGER


@INFER_ENGINE.register_function()
def inference_dreamvideo_entrance(cfg_update,  **kwargs):
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
    cfg_prefix = getattr(cfg, 'cfg_prefix', '')
    if hasattr(cfg, 'subject_cfg'):
        cfg = assign_signle_cfg(cfg, cfg_update, 'subject_cfg')
        subject_log_dir = os.path.join(cfg_prefix, cfg.log_dir)
    if hasattr(cfg, 'motion_cfg'):
        cfg = assign_signle_cfg(cfg, cfg_update, 'motion_cfg')
        motion_log_dir = os.path.join(cfg_prefix, cfg.log_dir)
    
    if hasattr(cfg, 'subject_cfg'):
        cfg['use_textInversion'] = True
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    cfg.gpu = gpu
    if cfg.use_random_seed:
        cfg.seed = random.randint(0, 10000)
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
    base_model = osp.basename(cfg.base_model).split('.')[0].split('_')[-1]
    
    cfg.log_dir = osp.join(cfg.log_dir, '%s' % (inf_name))
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
    logging.info(f"Going into inference_dreamvideo_entrance inference on {gpu} gpu")
    
    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

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
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    white_feature, _, zero_y = clip_encoder(image=clip_encoder.white_image, text="")
    _, _, zero_y_negative = clip_encoder(text=cfg.negative_prompt)
    zero_y, zero_y_negative = zero_y.detach(), zero_y_negative.detach()
    white_feature = white_feature.unsqueeze(1)
    zero_feature = torch.zeros_like(white_feature, device=white_feature.device) 

    embedding_manager = None
    if cfg.use_textInversion:
        cfg.embedmanager['embedder'] = clip_encoder   
        embedding_manager = EMBEDMANAGER.build(cfg.embedmanager)
        if hasattr(cfg, 'text_embedding_path'):
            text_embedding_path = os.path.join(cfg_prefix, cfg.text_embedding_path)
            embedding_manager.load(text_embedding_path)
        embedding_manager.cuda()
        for name, param in embedding_manager.named_parameters():
            param.requires_grad = False

    # [Model] auotoencoder 
    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.cuda()
    
    # [Model] UNet 
    model = MODEL.build(cfg.UNet)

    state_dict = torch.load(cfg.base_model, map_location='cpu')
    if 'state_dict' in state_dict:
        resume_step = state_dict['step']
        state_dict = state_dict['state_dict']
    else:
        resume_step = 0

    merged_state_dict = state_dict.copy()
    if hasattr(cfg, 'identity_adapter_index') and hasattr(cfg, 'identity_adapter_path'):
        raise Exception("Both identity_adapter_index and identity_adapter_path are used, please set only one.")
    elif hasattr(cfg, 'identity_adapter_index'):
        subject_cfg_name = cfg.subject_cfg.split('/')[-1].split('.')[0]
        identity_adapter_name = f'adapter_{cfg.identity_adapter_index:08d}.pth'
        identity_adapter_path = os.path.join(subject_log_dir, subject_cfg_name, 'checkpoints', identity_adapter_name)
        id_adapter_state_dict = torch.load(identity_adapter_path, map_location='cpu')
        merged_state_dict.update(id_adapter_state_dict['state_dict'])
    elif hasattr(cfg, 'identity_adapter_path'):
        id_adapter_state_dict = torch.load(cfg.identity_adapter_path, map_location='cpu')
        merged_state_dict.update(id_adapter_state_dict['state_dict'])

    if hasattr(cfg, 'motion_adapter_index') and hasattr(cfg, 'motion_adapter_path'):
        raise Exception("Both motion_adapter_index and motion_adapter_path are used, please set only one.")
    elif hasattr(cfg, 'motion_adapter_index'):
        motion_cfg_name = cfg.motion_cfg.split('/')[-1].split('.')[0]
        motion_adapter_name = f'adapter_{cfg.motion_adapter_index:08d}.pth'
        motion_adapter_path = os.path.join(motion_log_dir, motion_cfg_name, 'checkpoints', motion_adapter_name)
        motion_adapter_state_dict = torch.load(motion_adapter_path, map_location='cpu')
        merged_state_dict.update(motion_adapter_state_dict['state_dict'])
    elif hasattr(cfg, 'motion_adapter_path'):
        motion_adapter_state_dict = torch.load(cfg.motion_adapter_path, map_location='cpu')
        merged_state_dict.update(motion_adapter_state_dict['state_dict'])

    status = model.load_state_dict(merged_state_dict, strict=True)
    logging.info('Load model from {} with status {}'.format(cfg.base_model, status))
    model = model.to(gpu)
    model.eval()
    model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model
    torch.cuda.empty_cache()

    inverse_noise_strength = getattr(cfg, 'inverse_noise_strength', 0)
    if inverse_noise_strength > 0:
        latents = torch.load(cfg.latents_path, map_location='cpu')
        latents = latents.to(gpu)
        inverse_noise = diffusion.ddim_reverse_sample_loop(
            x0=latents,
            model=model.eval(),
            model_kwargs={'y': zero_y},
            guide_scale=None,
            ddim_timesteps=cfg.ddim_timesteps,
        )
    
    # [Test List]
    test_list = open(cfg.test_list_path).readlines()
    test_list = [item.strip() for item in test_list]
    num_videos = len(test_list)
    logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    test_list = [item for item in test_list for _ in range(cfg.round)]
    
    for idx, line in enumerate(test_list):
        if line.startswith('#'):
            logging.info(f'Skip {line}')
            continue
        logging.info(f"[{idx+1}]/[{num_videos * cfg.round}] Begin to sample {line} ...")
        img_key, caption = line.split('|||')
        img_name = os.path.basename(img_key).split('.')[0]
        if caption == "":
            logging.info(f'Caption is null of {caption}, skip..')
            continue
        captions = [caption]

        img_path = os.path.join(cfg.test_data_dir, img_key)
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        with torch.no_grad():
            image_tensor = vit_trans(image)
            image_tensor = image_tensor.unsqueeze(0)
            y_visual, y_text, y_words = clip_encoder(image=image_tensor, text=captions, embedding_manager=embedding_manager)
            y_visual = y_visual.unsqueeze(1)

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
                if cfg.noise_strength > 0:
                    b, c, f, *_ = noise.shape
                    offset_noise = torch.randn(b, c, f, 1, 1, device=noise.device)
                    noise = noise + cfg.noise_strength * offset_noise
                
                if inverse_noise_strength > 0:
                    noise = (inverse_noise_strength) ** 0.5 * inverse_noise + (1 - inverse_noise_strength) ** 0.5 * noise
                
                model_kwargs=[
                    {'y': y_words,},
                    {'y': zero_y_negative,}]
                if cfg.use_clip_adapter_condition:
                    model_kwargs[0]['y_image'] = y_visual
                    model_kwargs[1]['y_image'] = zero_feature

                    model_kwargs[0]['ag_strength'] = getattr(cfg, 'appearance_guide_strength_cond', 1)
                    model_kwargs[1]['ag_strength'] = getattr(cfg, 'appearance_guide_strength_uncond', 1)

                video_data = diffusion.ddim_sample_loop(
                    noise=noise,
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
        
        video_data = 1. / cfg.scale_factor * video_data
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
        cap_name = re.sub(r'[^\w\s\*]', '', caption).replace(' ', '_')
        file_name = f'{cap_name}_{cfg.seed}_{idx}.mp4'
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

