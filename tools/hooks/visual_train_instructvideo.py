# ------------------------------------------------------------------------
# InstructVideo: Instructing Video Diffusion Models with Human Feedback
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# ----------------------------- Notice -----------------------------------
# If you find it useful, please consider citing InstructVideo.
# ------------------------------------------------------------------------


import os
import torch
import pynvml
import logging
from einops import rearrange
import torch.cuda.amp as amp

from utils.registry_class import VISUAL
from utils.video_op import save_video_refimg_and_text

@VISUAL.register_class()
class VisualVideoTextDuringTrainUnClip(object):
    def __init__(self, cfg_global, autoencoder, diffusion, viz_num, **kwargs):
        super(VisualVideoTextDuringTrainUnClip, self).__init__(**kwargs)
        self.cfg = cfg_global
        self.viz_num = viz_num
        self.diffusion = diffusion
        self.autoencoder = autoencoder
    
    @torch.no_grad()
    def run(self, 
            model, 
            ref_frame,
            vit_frame, 
            video_data, 
            captions,
            y0, 
            zero_y, 
            step,
            **kwargs):
        cfg = self.cfg
        viz_num = self.viz_num
        captions = captions[:viz_num]
        vit_frame = vit_frame[:viz_num]
        # print memory
        pynvml.nvmlInit()
        handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
        logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')
        # sample images (DDIM)
        with amp.autocast(enabled=cfg.use_fp16):
            if cfg.share_noise:
                b, c, f, h, w= video_data.shape
                noise = torch.randn((viz_num, c, h, w), device=video_data.device)
                noise = noise.repeat_interleave(repeats=f,dim=0) ###share noise
                noise = rearrange(noise, '(b f) c h w->b c f h w',b=viz_num)
                noise = noise.contiguous()
            else:
                noise=torch.randn_like(video_data[:viz_num]) # viz_num: 8
            
            model_kwargs=[ {'y': y0[:viz_num]}, {'y': zero_y.repeat(viz_num, 1, 1)}]
            video_data = self.diffusion.ddim_sample_loop(
                noise=noise,
                model=model.eval(), #.requires_grad_(False),
                model_kwargs=model_kwargs,
                guide_scale=cfg.guide_scale,
                ddim_timesteps=cfg.ddim_timesteps,
                eta=0.0)

        video_data = 1. / cfg.scale_factor * video_data # [64, 4, 32, 48]
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
        chunk_size = min(cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size,dim=0)
        decode_data = []
        for vd_data in video_data_list:
            gen_frames = self.autoencoder.decode(vd_data)
            decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = viz_num)

        text_size = cfg.resolution[-1]
        ref_frame = ref_frame[:viz_num]
        file_name = f'rank_{cfg.world_size:02d}-{cfg.rank:02d}.gif'
        local_path = os.path.join(cfg.temp_dir, f'sample_{step:06d}/{file_name}')
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            # ops.save_video_image_not_gif(self.bucket, oss_key, local_path, ref_frame.cpu(), video_data.cpu(), captions, cfg.mean, cfg.std, cfg.vit_mean, cfg.vit_std, text_size)
            save_video_refimg_and_text(local_path, ref_frame.cpu(), video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
        except Exception as e:
            logging.info(f'Step: {step} save text or video error with {e}')
        