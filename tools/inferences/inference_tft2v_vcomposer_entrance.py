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
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.cuda.amp as amp
from importlib import reload
import torch.distributed as dist
import torch.multiprocessing as mp
import random
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
from copy import copy

# condition for videocomposer
from tools.annotator.canny import CannyDetector
from tools.annotator.sketch import pidinet_bsd, sketch_simplification_gan
from tools.annotator.depth import midas_v3


@INFER_ENGINE.register_function()
def inference_tft2v_vcomposer_entrance(cfg_update,  **kwargs):
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



# __all__ = ['make_irregular_mask', 'make_rectangle_mask', 'make_uncrop']

def make_irregular_mask(w, h, max_angle=4, max_length=200, max_width=100, min_strokes=1, max_strokes=5, mode='line'):
    # initialize mask
    assert mode in ['line', 'circle', 'square']
    mask = np.zeros((h, w), np.float32)

    # draw strokes
    num_strokes = np.random.randint(min_strokes, max_strokes + 1)
    for i in range(num_strokes):
        x1 = np.random.randint(w)
        y1 = np.random.randint(h)
        for j in range(1 + np.random.randint(5)):
            angle = 0.01 + np.random.randint(max_angle)
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 10 + np.random.randint(max_length)
            radius = 5 + np.random.randint(max_width)
            x2 = np.clip((x1 + length * np.sin(angle)).astype(np.int32), 0, w)
            y2 = np.clip((y1 + length * np.cos(angle)).astype(np.int32), 0, h)
            if mode == 'line':
                cv2.line(mask, (x1, y1), (x2, y2), 1.0, radius)
            elif mode == 'circle':
                cv2.circle(mask, (x1, y1), radius=radius, color=1.0, thickness=-1)
            elif mode == 'square':
                radius = radius // 2
                mask[y1 - radius:y1 + radius, x1 - radius:x1 + radius] = 1
            x1, y1 = x2, y2
    return mask

def make_rectangle_mask(w, h, margin=10, min_size=30, max_size=150, min_strokes=1, max_strokes=4):
    # initialize mask
    mask = np.zeros((h, w), np.float32)

    # draw rectangles
    num_strokes = np.random.randint(min_strokes, max_strokes + 1)
    for i in range(num_strokes):
        box_w = np.random.randint(min_size, max_size)
        box_h = np.random.randint(min_size, max_size)
        x1 = np.random.randint(margin, w - margin - box_w + 1)
        y1 = np.random.randint(margin, h - margin - box_h + 1)
        mask[y1:y1 + box_h, x1:x1 + box_w] = 1
    return mask

def make_uncrop(w, h):
    # initialize mask
    mask = np.zeros((h, w), np.float32)

    # randomly halve the image
    side = np.random.choice([0, 1, 2, 3])
    if side == 0:
        mask[:h // 2, :] = 1
    elif side == 1:
        mask[h // 2:, :] = 1
    elif side == 2:
        mask[:, :w // 2] = 1
    elif side == 3:
        mask[:, w // 2:] = 1
    return mask

def make_masked_images(imgs, masks):
    masked_imgs = []
    for i, mask in enumerate(masks):        
        # concatenation
        masked_imgs.append(torch.cat([imgs[i] * (1 - mask), (1 - mask)], dim=1))
    return torch.stack(masked_imgs, dim=0)

def load_video_frames(vid_path, train_trans, vit_transforms, max_frames=16, sample_fps = 4, resolution=[448, 256], get_first_frame=True, vit_resolution=[224, 224]):
    
    file_path = vid_path
    for _ in range(5):
        try:
            capture = cv2.VideoCapture(file_path)
            _fps = capture.get(cv2.CAP_PROP_FPS)
            _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            stride = round(_fps / sample_fps)
            cover_frame_num = (stride * max_frames)
            if _total_frame_num < cover_frame_num + 5:
                start_frame = 0
                end_frame = _total_frame_num
            else:
                start_frame = random.randint(0, _total_frame_num-cover_frame_num-5)
                end_frame = start_frame + cover_frame_num
            
            pointer, frame_list = 0, []
            while(True):
                ret, frame = capture.read()
                pointer +=1 
                if (not ret) or (frame is None): break
                if pointer < start_frame: continue
                if pointer >= end_frame - 1: break
                if (pointer - start_frame) % stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame_list.append(frame)
            break
        except Exception as e:
            logging.info('{} read video frame failed with error: {}'.format(vid_path, e))
            continue

    video_data = torch.zeros(max_frames, 3,  resolution[1], resolution[0])
    if get_first_frame:
        ref_idx = 0
    else:
        ref_idx = int(len(frame_list)/2)
    try:
        if len(frame_list)>0:
            mid_frame = copy(frame_list[ref_idx])
            vit_frame = vit_transforms(mid_frame)
            frames = train_trans(frame_list)
            video_data[:len(frame_list), ...] = frames
        else:
            vit_frame = torch.zeros(3, vit_resolution[1], vit_resolution[0])
    except:
        vit_frame = torch.zeros(3, vit_resolution[1], vit_resolution[0])
    ref_frame = copy(frames[ref_idx])
    # ref_frame = vit_frame
    p = random.random()
    if p < 0.7:
        mask = make_irregular_mask(512, 512)
    elif p < 0.9:
        mask = make_rectangle_mask(512, 512)
    else:
        mask = make_uncrop(512, 512)
    # mask = torch.from_numpy(cv2.resize(mask, (self.misc_size,self.misc_size), interpolation=cv2.INTER_NEAREST)).unsqueeze(0).float()
    mask = torch.from_numpy(cv2.resize(mask, (resolution[0], resolution[1]), interpolation=cv2.INTER_NEAREST)).unsqueeze(0).float()

    mask = mask.unsqueeze(0).repeat_interleave(repeats=max_frames, dim=0)
    return ref_frame, vit_frame, video_data, mask



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
    logging.info(f"Going into inference_sr600_entrance inference on {gpu} gpu")
    
    # [Diffusion]
    diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Data] Data Transform    
    train_trans = data.Compose([
        data.Resize(cfg.resolution),
        data.ToTensor(),
        # data.Normalize(mean=cfg.mean, std=cfg.std)
        ])

    vit_transforms = T.Compose([
                data.Resize(cfg.resolution),
                data.Resize(cfg.vit_resolution),
                T.ToTensor(),
                T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    # [Model] embedder
    clip_encoder = EMBEDDER.build(cfg.embedder)
    clip_encoder.model.to(gpu)
    _, _, zero_y = clip_encoder(text=cfg.negative_prompt)
    zero_y = zero_y.detach()
    
    # # [Model] visual embedder
    # clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(layer='penultimate',pretrained = DOWNLOAD_TO_CACHE(cfg.clip_checkpoint))
    # clip_encoder_visual.model.to(gpu)

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


    ### Generators for various conditions
    if 'depthmap' in cfg.video_compositions:
        midas = midas_v3(pretrained=True).eval().requires_grad_(False).to(
            memory_format=torch.channels_last).half().to(gpu)
    if 'canny' in cfg.video_compositions:
        canny_detector = CannyDetector()
    if 'sketch' in cfg.video_compositions or 'single_sketch' in cfg.video_compositions:
        pidinet = pidinet_bsd(pretrained=True, vanilla_cnn=True).eval().requires_grad_(False).to(gpu)
        cleaner = sketch_simplification_gan(pretrained=True).eval().requires_grad_(False).to(gpu)
        pidi_mean = torch.tensor(cfg.sketch_mean).view(1, -1, 1, 1).to(gpu)
        pidi_std = torch.tensor(cfg.sketch_std).view(1, -1, 1, 1).to(gpu)
    
    
    # [Test List]
    test_list = open(cfg.test_list_path).readlines()
    test_list = [item.strip() for item in test_list]
    num_videos = len(test_list)
    logging.info(f'There are {num_videos} videos. with {cfg.round} times')
    test_list = [item for item in test_list for _ in range(cfg.round)]
    
    for idx, file_path in enumerate(test_list):
        setup_seed(cfg.seed + cfg.rank)
        video_key, caption = file_path.split('|||')
        video_local_path = os.path.join(cfg.data_dir, video_key)
        if caption.startswith('#'):
            logging.info(f'Skip {caption}')
            continue
        if '|' in caption:
            caption, manual_seed = caption.split('|')
            manual_seed = int(manual_seed)
        else:
            manual_seed = int(cfg.seed)
        logging.info(f"[{idx}]/[{num_videos}] Begin to sample {caption}, seed {manual_seed} ...")
        if caption == "": 
            logging.info(f'Caption is null of {caption}, skip..')
            continue

        captions = [caption + cfg.positive_prompt]
        # from ipdb import set_trace; set_trace()
        with torch.no_grad():
            _, y_text, y_words = clip_encoder(text=captions) # bs * 1 *1024 [B, 1, 1024]

        
        
        # ref_frame, vit_frame, video_data, mask
        ref_imgs, vit_frame, misc_data, mask = load_video_frames(video_local_path, train_trans, vit_transforms, max_frames=cfg.max_frames, sample_fps =cfg.sample_fps, resolution=cfg.resolution)
        misc_data = misc_data.unsqueeze(0).to(gpu)
        vit_frame = vit_frame.unsqueeze(0).to(gpu)
        mask = mask.unsqueeze(0).to(gpu)
        ref_imgs = ref_imgs.to(gpu)
        # from ipdb import set_trace; set_trace()

        ### save for visualization
        misc_backups = copy(misc_data)
        frames_num = misc_data.shape[1]
        misc_backups = rearrange(misc_backups, 'b f c h w -> b c f h w')
        mv_data_video = []
        # if 'motion' in cfg.video_compositions: # need pip install motion-vector-extractor==1.0.6
        #     mv_data_video = rearrange(mv_data, 'b f c h w -> b c f h w')

        ### mask images
        masked_video = []
        if 'mask' in cfg.video_compositions:
            masked_video = make_masked_images(misc_data.sub(0.5).div_(0.5), mask)
            masked_video = rearrange(masked_video, 'b f c h w -> b c f h w')
        
        image_local = []
        if 'local_image' in cfg.video_compositions:
            frames_num = misc_data.shape[1]
            bs_vd_local = misc_data.shape[0]
            image_local = misc_data[:,:1].clone().repeat(1,frames_num,1,1,1)
            image_local = rearrange(image_local, 'b f c h w -> b c f h w', b = bs_vd_local)

        
        ### encode the video_data
        
        bs_vd = misc_data.shape[0]
        # video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')
        misc_data = rearrange(misc_data, 'b f c h w -> (b f) c h w')

        # video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
        misc_data_list = torch.chunk(misc_data, misc_data.shape[0]//cfg.chunk_size,dim=0)

        with torch.no_grad():
            # decode_data = []
            # for vd_data in video_data_list:
            #     encoder_posterior = autoencoder.encode(vd_data)
            #     tmp = get_first_stage_encoding(encoder_posterior).detach()
            #     decode_data.append(tmp)
            # video_data = torch.cat(decode_data,dim=0)
            # video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = bs_vd)

            depth_data = []
            if 'depthmap' in cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    depth = midas(misc_imgs.sub(0.5).div_(0.5).to(memory_format=torch.channels_last).half())
                    depth = (depth / cfg.depth_std).clamp_(0, cfg.depth_clamp)
                    depth_data.append(depth)
                depth_data = torch.cat(depth_data, dim = 0)
                depth_data = rearrange(depth_data, '(b f) c h w -> b c f h w', b = bs_vd)
            
            sketch_data = []
            if 'sketch' in cfg.video_compositions:
                for misc_imgs in misc_data_list:
                    sketch = pidinet(misc_imgs.sub(pidi_mean).div_(pidi_std))
                    sketch = 1.0 - cleaner(1.0 - sketch)
                    sketch_data.append(sketch)
                sketch_data = torch.cat(sketch_data, dim = 0)
                sketch_data = rearrange(sketch_data, '(b f) c h w -> b c f h w', b = bs_vd)
            
            single_sketch_data = []
            if 'single_sketch' in cfg.video_compositions:
                if 'sketch' not in cfg.video_compositions:
                    sketch_data_c = []
                    for misc_imgs in misc_data_list:
                        sketch = pidinet(misc_imgs.sub(pidi_mean).div_(pidi_std))
                        sketch = 1.0 - cleaner(1.0 - sketch)
                        sketch_data_c.append(sketch)
                    sketch_data_c = torch.cat(sketch_data_c, dim = 0)
                    sketch_data_c = rearrange(sketch_data_c, '(b f) c h w -> b c f h w', b = bs_vd)
                    single_sketch_data = sketch_data_c.clone()[:,:,:1].repeat(1,1,frames_num,1,1)
                
                else:
                    single_sketch_data = sketch_data.clone()[:,:,:1].repeat(1,1,frames_num,1,1)
            
            y_visual = []
            if 'image' in cfg.video_compositions:
                with torch.no_grad():
                    vit_frame = vit_frame.squeeze(1)
                    y_visual = clip_encoder.encode_image(vit_frame).unsqueeze(1) # [60, 1024]
                    y_visual0 = y_visual.clone()
                
       

        with amp.autocast(enabled=True):
            pynvml.nvmlInit()
            handle=pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
            # total_noise_levels = cfg.total_noise_levels
            # setup_seed(0)
            cur_seed = torch.initial_seed()
            logging.info(f"Current seed {cur_seed} ...")
            noise = torch.randn([1, 4, cfg.max_frames, int(cfg.resolution[1]/cfg.scale), int(cfg.resolution[0]/cfg.scale)])
            noise = noise.to(gpu)
            
            full_model_kwargs=[{
                                        'y': y_words,
                                        "local_image": None if len(image_local) == 0 else image_local,
                                        'image': None if len(y_visual) == 0 else y_visual0,
                                        'depth': None if len(depth_data) == 0 else depth_data,
                                        # 'canny': None if len(canny_data) == 0 else canny_data,
                                        'sketch': None if len(sketch_data) == 0 else sketch_data,
                                        # 'histogram': None if len(hist_data) == 0 else hist_data,
                                        'masked': None if len(masked_video) == 0 else masked_video,
                                        'motion': None if len(mv_data_video) == 0 else mv_data_video,
                                        'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data,
                                        
                                       }, 
                                       {
                                        
                                        'y': zero_y.repeat(y_words.shape[0],1,1),
                                        "local_image": None if len(image_local) == 0 else image_local,
                                        'image': None if len(y_visual) == 0 else torch.zeros_like(y_visual0),
                                        'depth': None if len(depth_data) == 0 else depth_data,
                                        # 'canny': None if len(canny_data) == 0 else canny_data,
                                        'sketch': None if len(sketch_data) == 0 else sketch_data,
                                        # 'histogram': None if len(hist_data) == 0 else hist_data,
                                        'masked': None if len(masked_video) == 0 else masked_video,
                                        'motion': None if len(mv_data_video) == 0 else mv_data_video,
                                        'single_sketch': None if len(single_sketch_data) == 0 else single_sketch_data,
                                        
                                       }]

            # model_kwargs=[ {'y': y_words}, {'y': zero_y}]
            partial_keys = [
                    ['y', 'depth'],
                    ['y', 'sketch'],
                ]
            if hasattr(cfg, "partial_keys") and cfg.partial_keys:
                partial_keys = cfg.partial_keys

            for partial_keys_one in partial_keys:
                model_kwargs_one = prepare_model_kwargs(partial_keys = partial_keys_one,
                                    full_model_kwargs = full_model_kwargs,
                                    use_fps_condition = cfg.use_fps_condition)
                noise_one = noise
                video_data = diffusion.ddim_sample_loop(
                    noise=noise_one,
                    model=model.eval(), #.requires_grad_(False),
                    model_kwargs=model_kwargs_one,
                    guide_scale=9.0,
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
                video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = cfg.batch_size).float()
                # video_data = torch.cat([spat_key_frames[:, :, None, :, :], video_data], dim=2)
                
                text_size = cfg.resolution[-1]
                cap_name = re.sub(r'[^\w\s]', '', caption).replace(' ', '_')
                name = 'condition'
                for ii in partial_keys_one:
                    name = name + "_" + ii
                file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{idx:04d}_{name}_{cap_name}.mp4'
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

def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition=False):
    
    if use_fps_condition is True:
        partial_keys.append('fps')

    partial_model_kwargs = [{}, {}]
    for partial_key in partial_keys:
        partial_model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
        partial_model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]

    return partial_model_kwargs