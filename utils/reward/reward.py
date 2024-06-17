# ------------------------------------------------------------------------
# InstructVideo: Instructing Video Diffusion Models with Human Feedback
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# ----------------------------- Notice -----------------------------------
# If you find it useful, please consider citing InstructVideo.
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
import math

from .open_clip import create_model_and_transforms, get_tokenizer
from .stat_tracking import PerPromptStatTracker

from PIL import Image
from einops import rearrange
import numpy as np
import cv2
from skimage.metrics import structural_similarity
import torchvision
from piq import ssim, SSIMLoss
import pdb
import os

# By default, RewardModel uses HPSv2 as a reward model.
__all__ = ['DiffRewardModel', 'RWRRewardModel', 'DDPORewardModel']


class DiffRewardModel(object):
    def __init__(self,
                 reward_type,
                 cfg,
                 autoencoder,
                 segments = 8,
                 step_thresold = 300,
                 modulating_reward = 'hard_clip',
                 tracking_strategy = 'hard_clip',
                 reward_normalization = True,
                 positive_reward = False,
                 ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # reward initialization
        if reward_type == "HPSv2":
            model_name = "ViT-H-14"
            if cfg.reward_precision == 'fp16':
                precision = 'fp16'
            else:
                precision = 'amp'

            print(f'reward {reward_type}:{model_name}')
            self.reward_model, self.reward_preprocess_train, self.reward_preprocess_val = \
                create_model_and_transforms(
                model_name,
                'laion2B-s32B-b79K',
                precision=precision,
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False,
                cache_dir='models/.cache/',
            )

            self.diff_normalize = torchvision.transforms.Normalize(
                                                mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
            self.diff_resize = torchvision.transforms.Resize(224)

            checkpoint = torch.load('models/HPS_v2.pt')
            self.reward_model.load_state_dict(checkpoint['state_dict'])
            self.tokenizer = get_tokenizer(model_name)
            self.reward_model.eval()

            vid_mean = [0.5, 0.5, 0.5]
            vid_std = [0.5, 0.5, 0.5]
            self.vid_mean = torch.tensor(vid_mean, device=self.device).view(1, -1, 1, 1, 1) # ncfhw
            self.vid_std = torch.tensor(vid_std, device=self.device).view(1, -1, 1, 1, 1) # ncfhw
        
        self.cfg = cfg
        self.autoencoder = autoencoder
        self.segments = segments
        print(f"We use {self.segments} segments for spatial rewards.")
        self.selection_method = cfg.selection_method
        self.exponential_TSN = self.cfg.exponential_TSN
        self.lambda_TAR = self.cfg.lambda_TAR
        if self.selection_method == "TSN":
            print(f"We use exponential for TSN reward: {self.exponential_TSN}")
        self.modulating_reward = modulating_reward
        print(f"The modulating strategy for the reward: {self.modulating_reward}")
        self.tracking_strategy = tracking_strategy
        print(f"The tracking strategy for the reward: {self.tracking_strategy}")
        self.reward_normalization = reward_normalization
        self.positive_reward = positive_reward

        if self.reward_normalization:
            self.reward_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                    min_count = cfg.reward_tracker['min_count'])
        if 'mean' in self.cfg.temporal_reward_type:
            if self.reward_normalization:
                self.motion_mean_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                        min_count = cfg.reward_tracker['min_count'])
        if 'std' in self.cfg.temporal_reward_type:
            if self.reward_normalization:
                self.motion_std_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                        min_count = cfg.reward_tracker['min_count'])

        self.adv_clip_max = cfg.adv_clip_max
        self.adv_clip_min = - cfg.adv_clip_max
        self.reward_weights = cfg.reward_weights
        self.ST_reward_weights = cfg.ST_reward_weights
        self.step_count = 0
        self.motion_rep = cfg.motion_rep
        print(f'Motion representation: {self.motion_rep}')
        self.low_penal_threshold = cfg.low_penal_threshold 

    def reward_scorer(self, captions,
                            denoised_x0,
                            loss_recon,
                            # log_prob,
                            t_round):
        ### Video data pre-processing
        video_data = 1. / self.cfg.scale_factor * denoised_x0 # [64, 4, 32, 48]
        video_num, _, frame_num, _, _ = video_data.shape
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
        
        chunk_size = min(self.cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
        decode_data = []
        for vd_data in video_data_list:
            gen_frames = self.autoencoder.decode(vd_data)
            decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = video_num)

        video_data = video_data.mul_(self.vid_std).add_(self.vid_mean)  # 8x3x16x256x384
        video_data.clamp_(0, 1)
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')

        video_data_resize = self.diff_resize(video_data)
        video_data_norm = self.diff_normalize(video_data_resize).to(video_data.dtype)

        ### Frame selection
        video_data_norm = rearrange(video_data_norm, '(b f) c h w -> b f c h w', b = video_num)
        segment_span = frame_num // self.segments
        if self.selection_method == 'fixed_first':
            pre_video_data = video_data_norm[:, ::segment_span]
        elif self.selection_method == 'TSN':
            local_index = torch.randint(segment_span, (video_num, self.segments)).to(video_data_norm.device)
            global_index = local_index + (torch.tensor(range(self.segments))[None,] * segment_span).to(video_data_norm.device)
            pre_video_data = video_data_norm[torch.arange(video_num).unsqueeze(-1), global_index, :, :, :]
        else:
            print("Not implemented yet.")
            assert False
            

        pre_video_chunk = torch.chunk(pre_video_data, pre_video_data.shape[0]//chunk_size, dim=0)
   
        ### Text data pre-processing
        pre_captions = self.tokenizer(captions)
        pre_captions = pre_captions.to(self.device)
        pre_captions_chunk = torch.chunk(pre_captions, pre_captions.shape[0]//chunk_size, dim = 0)
        if self.exponential_TSN:
            TSN_coef = torch.exp(-torch.abs(global_index - frame_num//2) * self.lambda_TAR)
            TSN_coef_chunk = torch.chunk(TSN_coef, TSN_coef.shape[0]//chunk_size, dim = 0)
            

        reward_scores = []
        with torch.cuda.amp.autocast(enabled=self.cfg.use_fp16):
            for chunk_idx, (vid_chunk, cap_chunk) in enumerate(zip(pre_video_chunk, pre_captions_chunk)):
                vid_chunk = rearrange(vid_chunk, 'k s c h w -> (k s) c h w')
                outputs = self.reward_model(vid_chunk, cap_chunk)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                # image_features: (chunk_size*segments) * dim, text_features: chunk_size * dim
                logits_per_image = image_features @ text_features.T
                logits_per_image = rearrange(logits_per_image, '(k s) x-> s k x', s = self.segments)
                diag = torch.diagonal(logits_per_image, dim1 = 1, dim2 = 2) # s, k
                if self.exponential_TSN:
                    diag = TSN_coef_chunk[chunk_idx].permute(1, 0) * diag
                diag = torch.mean(diag, dim = 0) # Average over 'self.segments' dimension to obtain: (chunk_size, )
                reward_scores.append(diag)
        

        ########################## Implementation 2 ###############################
        # GT videos as reward and improve the calculation of log_prob
        reward_scores = torch.cat(reward_scores, dim = 0)

        advantages_spatial = 1 - reward_scores

        ### Improved temporal reward
        advantages_motion = 0
        advantages = self.ST_reward_weights['spatial'] * advantages_spatial + self.ST_reward_weights['temporal'] * advantages_motion

        ### Modulating the rewards according to the step index
        if self.cfg.data_align_method == 'ddpm':
            assert loss_recon is not None
            recon_degree = loss_recon.detach()
            data_align_coef = torch.exp(- self.cfg.data_align_coef * recon_degree)
            print(recon_degree, data_align_coef, advantages)
            reward_loss = (data_align_coef * advantages).mean()
        elif self.cfg.data_align_method is None:
            reward_loss = (advantages).mean()

        if torch.any(torch.isinf(reward_loss)):
            reward_loss = reward_loss.clamp(self.adv_clip_min, self.adv_clip_max)
            print("reward_loss has Inf values! Clamp to (self.adv_clip_min, self.adv_clip_max).")
        
        loss = self.reward_weights['reward'] * reward_loss # + self.reward_weights['reg'] * recon_loss
        
        self.step_count += 1
        if self.step_count % 5 == 0:
            if len(self.cfg.temporal_reward_type) > 0:
                obs_str = f"advantages_spatial: {advantages_spatial.mean().item():.5f}, advantages_motion: {advantages_motion.mean().item():.5f}, reward_loss: {reward_loss.item():.5f}"
            else:
                obs_str = f"advantages: {advantages.mean().item():.5f}, reward_loss: {reward_loss.item():.5f}"
            if loss_recon is not None:
                obs_str += f', recon_loss: {recon_loss.item():.5f}'
            print(obs_str)

        return loss


class RWRRewardModel(object):
    def __init__(self,
                 reward_type,
                 cfg,
                 autoencoder,
                 segments = 8,
                 step_thresold = 300,
                 modulating_reward = 'hard_clip',
                 tracking_strategy = 'hard_clip',
                 reward_normalization = True,
                 positive_reward = False,
                 ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # reward initialization
        if reward_type == "HPSv2":
            model_name = "ViT-H-14"
            if cfg.reward_precision == 'fp16':
                precision = 'fp16'
            else:
                precision = 'amp'

            print(f'reward {reward_type}:{model_name}')
            self.reward_model, self.reward_preprocess_train, self.reward_preprocess_val = \
                create_model_and_transforms(
                model_name,
                'laion2B-s32B-b79K',
                precision=precision,
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False,
                cache_dir='models/.cache/',
            )

            self.diff_normalize = torchvision.transforms.Normalize(
                                                mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
            self.diff_resize = torchvision.transforms.Resize(224)

            checkpoint = torch.load('models/HPS_v2.pt')
            self.reward_model.load_state_dict(checkpoint['state_dict'])
            self.tokenizer = get_tokenizer(model_name)
            self.reward_model.eval()

            vid_mean = [0.5, 0.5, 0.5]
            vid_std = [0.5, 0.5, 0.5]
            self.vid_mean = torch.tensor(vid_mean, device=self.device).view(1, -1, 1, 1, 1) # ncfhw
            self.vid_std = torch.tensor(vid_std, device=self.device).view(1, -1, 1, 1, 1) # ncfhw
        
        self.cfg = cfg
        self.autoencoder = autoencoder
        self.segments = segments
        print(f"We use {self.segments} segments for spatial rewards.")
        self.selection_method = cfg.selection_method
        self.exponential_TSN = self.cfg.exponential_TSN
        if self.selection_method == "TSN":
            print(f"We use exponential for TSN reward: {self.exponential_TSN}")
        self.modulating_reward = modulating_reward
        print(f"The modulating strategy for the reward: {self.modulating_reward}")
        self.tracking_strategy = tracking_strategy
        print(f"The tracking strategy for the reward: {self.tracking_strategy}")
        self.reward_normalization = reward_normalization
        self.positive_reward = positive_reward

        if self.reward_normalization:
            self.reward_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                    min_count = cfg.reward_tracker['min_count'])
        if 'mean' in self.cfg.temporal_reward_type:
            if self.reward_normalization:
                self.motion_mean_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                        min_count = cfg.reward_tracker['min_count'])
        if 'std' in self.cfg.temporal_reward_type:
            if self.reward_normalization:
                self.motion_std_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                        min_count = cfg.reward_tracker['min_count'])
        self.adv_clip_max = cfg.adv_clip_max
        self.adv_clip_min = - cfg.adv_clip_max
        self.reward_weights = cfg.reward_weights
        self.ST_reward_weights = cfg.ST_reward_weights
        self.step_count = 0
        self.motion_rep = cfg.motion_rep
        print(f'Motion representation: {self.motion_rep}')
        self.low_penal_threshold = cfg.low_penal_threshold 

    def reward_scorer(self, captions,
                            denoised_x0,
                            loss_recon,
                            # log_prob,
                            t_round):
        ### Video data pre-processing
        video_data = 1. / self.cfg.scale_factor * denoised_x0 # [64, 4, 32, 48]
        video_num, _, frame_num, _, _ = video_data.shape
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
        
        chunk_size = min(self.cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
        decode_data = []
        with torch.no_grad():  # test point for torch.no_grad()
            for vd_data in video_data_list:
                gen_frames = self.autoencoder.decode(vd_data)
                decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = video_num)

        video_data = video_data.mul_(self.vid_std).add_(self.vid_mean)  # 8x3x16x256x384
        video_data.clamp_(0, 1)
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')


        video_data_resize = self.diff_resize(video_data)
        video_data_norm = self.diff_normalize(video_data_resize).to(video_data.dtype)

        ### Frame selection
        video_data_norm = rearrange(video_data_norm, '(b f) c h w -> b f c h w', b = video_num)
        segment_span = frame_num // self.segments
        if self.selection_method == 'fixed_first':
            pre_video_data = video_data_norm[:, ::segment_span]
        elif self.selection_method == 'TSN':
            local_index = torch.randint(segment_span, (video_num, self.segments)).to(video_data_norm.device)
            global_index = local_index + (torch.tensor(range(self.segments))[None,] * segment_span).to(video_data_norm.device)
            pre_video_data = video_data_norm[torch.arange(video_num).unsqueeze(-1), global_index, :, :, :]
        else:
            print("Not implemented yet.")
            assert False
            
        pre_video_chunk = torch.chunk(pre_video_data, pre_video_data.shape[0]//chunk_size, dim=0)

        ### Text data pre-processing
        pre_captions = self.tokenizer(captions)
        pre_captions = pre_captions.to(self.device)
        pre_captions_chunk = torch.chunk(pre_captions, pre_captions.shape[0]//chunk_size, dim = 0)
        if self.exponential_TSN:
            TSN_coef = torch.exp(-torch.abs(global_index - frame_num//2) * self.lambda_TAR)
            TSN_coef_chunk = torch.chunk(TSN_coef, TSN_coef.shape[0]//chunk_size, dim = 0)
            

        reward_scores = []
        with torch.cuda.amp.autocast(enabled=self.cfg.use_fp16):
            for chunk_idx, (vid_chunk, cap_chunk) in enumerate(zip(pre_video_chunk, pre_captions_chunk)):
                vid_chunk = rearrange(vid_chunk, 'k s c h w -> (k s) c h w')
                outputs = self.reward_model(vid_chunk, cap_chunk)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                # image_features: (chunk_size*segments) * dim, text_features: chunk_size * dim
                logits_per_image = image_features @ text_features.T
                logits_per_image = rearrange(logits_per_image, '(k s) x-> s k x', s = self.segments)
                diag = torch.diagonal(logits_per_image, dim1 = 1, dim2 = 2) # s, k
                if self.exponential_TSN:
                    diag = TSN_coef_chunk[chunk_idx].permute(1, 0) * diag
                diag = torch.mean(diag, dim = 0) # Average over 'self.segments' dimension to obtain: (chunk_size, )
                reward_scores.append(diag)
        

        ########################## Implementation 2 ###############################
        # GT videos as reward and improve the calculation of log_prob
        reward_scores = torch.cat(reward_scores, dim = 0)
        advantages_spatial = reward_scores

        ### Improved temporal reward
        advantages_motion = 0
        advantages = self.ST_reward_weights['spatial'] * advantages_spatial + self.ST_reward_weights['temporal'] * advantages_motion
        
        reward_loss = (advantages * loss_recon).mean()

        if torch.any(torch.isinf(reward_loss)):
            reward_loss = reward_loss.clamp(self.adv_clip_min, self.adv_clip_max)
            print("reward_loss has Inf values! Clamp to (self.adv_clip_min, self.adv_clip_max).")
        
        loss = self.reward_weights['reward'] * reward_loss # + self.reward_weights['reg'] * recon_loss
        
        self.step_count += 1
        if self.step_count % 5 == 0:
            if len(self.cfg.temporal_reward_type) > 0:
                obs_str = f"advantages_spatial: {advantages_spatial.mean().item():.5f}, advantages_motion: {advantages_motion.mean().item():.5f}, reward_loss: {reward_loss.item():.5f}"
            else:
                obs_str = f"advantages: {advantages.mean().item():.5f}, reward_loss: {reward_loss.item():.5f}"
            if loss_recon is not None:
                obs_str += f', recon_loss: {recon_loss.item():.5f}'
            print(obs_str)

        return loss


class DDPORewardModel(object):
    def __init__(self,
                 reward_type,
                 cfg,
                 autoencoder,
                 segments = 8,
                 step_thresold = 300,
                 modulating_reward = 'hard_clip',
                 tracking_strategy = 'hard_clip',
                 reward_normalization = True,
                 positive_reward = False,
                 ):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # reward initialization
        if reward_type == "HPSv2":
            model_name = "ViT-H-14"
            if cfg.reward_precision == 'fp16':
                precision = 'fp16'
            else:
                precision = 'amp'

            print(f'reward {reward_type}:{model_name}')
            self.reward_model, self.reward_preprocess_train, self.reward_preprocess_val = \
                create_model_and_transforms(
                model_name,
                'laion2B-s32B-b79K',
                precision=precision,
                device=self.device,
                jit=False,
                force_quick_gelu=False,
                force_custom_text=False,
                force_patch_dropout=False,
                force_image_size=None,
                pretrained_image=False,
                image_mean=None,
                image_std=None,
                light_augmentation=True,
                aug_cfg={},
                output_dict=True,
                with_score_predictor=False,
                with_region_predictor=False,
                cache_dir='models/.cache/',
            )

            self.diff_normalize = torchvision.transforms.Normalize(
                                                mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
            self.diff_resize = torchvision.transforms.Resize(224)

            checkpoint = torch.load('models/HPS_v2.pt')
            self.reward_model.load_state_dict(checkpoint['state_dict'])
            self.tokenizer = get_tokenizer(model_name)
            self.reward_model.eval()

            vid_mean = [0.5, 0.5, 0.5]
            vid_std = [0.5, 0.5, 0.5]
            self.vid_mean = torch.tensor(vid_mean, device=self.device).view(1, -1, 1, 1, 1) # ncfhw
            self.vid_std = torch.tensor(vid_std, device=self.device).view(1, -1, 1, 1, 1) # ncfhw
        
        self.cfg = cfg
        self.autoencoder = autoencoder
        self.segments = segments
        print(f"We use {self.segments} segments for spatial rewards.")
        self.selection_method = cfg.selection_method
        self.exponential_TSN = self.cfg.exponential_TSN
        if self.selection_method == "TSN":
            print(f"We use exponential for TSN reward: {self.exponential_TSN}")
        self.modulating_reward = modulating_reward
        print(f"The modulating strategy for the reward: {self.modulating_reward}")
        self.tracking_strategy = tracking_strategy
        print(f"The tracking strategy for the reward: {self.tracking_strategy}")
        self.reward_normalization = reward_normalization
        self.positive_reward = positive_reward

        if self.reward_normalization:
            self.reward_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                    min_count = cfg.reward_tracker['min_count'])
        if 'mean' in self.cfg.temporal_reward_type:
            if self.reward_normalization:
                self.motion_mean_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                        min_count = cfg.reward_tracker['min_count'])
        if 'std' in self.cfg.temporal_reward_type:
            if self.reward_normalization:
                self.motion_std_tracker = PerPromptStatTracker(buffer_size = cfg.reward_tracker['buffer_size'],
                                                        min_count = cfg.reward_tracker['min_count'])
        self.adv_clip_max = cfg.adv_clip_max
        self.adv_clip_min = - cfg.adv_clip_max
        self.reward_weights = cfg.reward_weights
        self.ST_reward_weights = cfg.ST_reward_weights
        self.step_count = 0
        self.motion_rep = cfg.motion_rep
        print(f'Motion representation: {self.motion_rep}')
        self.low_penal_threshold = cfg.low_penal_threshold 
        self.ddim_timesteps = cfg.ddim_timesteps
        self.num_timesteps = cfg.num_timesteps
        self.guide_scale = cfg.guide_scale
        self.DDPO_eta = cfg.DDPO_eta

    def reward_scorer(self, captions,
                        denoised_x0,
                        model,
                        sample_data,
                        loss_recon,
                        # log_prob,
                        # t_round
                        ):
        ### Video data pre-processing
        video_data = 1. / self.cfg.scale_factor * denoised_x0 # [64, 4, 32, 48]
        video_num, _, frame_num, _, _ = video_data.shape
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
        
        chunk_size = min(self.cfg.decoder_bs, video_data.shape[0])
        video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
        decode_data = []
        with torch.no_grad():  # test point for torch.no_grad()
            for vd_data in video_data_list:
                gen_frames = self.autoencoder.decode(vd_data)
                decode_data.append(gen_frames)
        video_data = torch.cat(decode_data, dim=0)
        video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = video_num)

        video_data = video_data.mul_(self.vid_std).add_(self.vid_mean)  # 8x3x16x256x384
        video_data.clamp_(0, 1)
        video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')

        video_data_resize = self.diff_resize(video_data)
        video_data_norm = self.diff_normalize(video_data_resize).to(video_data.dtype)

        ### Frame selection
        video_data_norm = rearrange(video_data_norm, '(b f) c h w -> b f c h w', b = video_num)
        segment_span = frame_num // self.segments
        if self.selection_method == 'fixed_first':
            pre_video_data = video_data_norm[:, ::segment_span]
        elif self.selection_method == 'TSN':
            local_index = torch.randint(segment_span, (video_num, self.segments)).to(video_data_norm.device)
            global_index = local_index + (torch.tensor(range(self.segments))[None,] * segment_span).to(video_data_norm.device)
            pre_video_data = video_data_norm[torch.arange(video_num).unsqueeze(-1), global_index, :, :, :]
        else:
            print("Not implemented yet.")
            assert False
            
        pre_video_chunk = torch.chunk(pre_video_data, pre_video_data.shape[0]//chunk_size, dim=0)
   
        ### Text data pre-processing
        pre_captions = self.tokenizer(captions)
        pre_captions = pre_captions.to(self.device)
        pre_captions_chunk = torch.chunk(pre_captions, pre_captions.shape[0]//chunk_size, dim = 0)
        if self.exponential_TSN:
            TSN_coef = torch.exp(-torch.abs(global_index - frame_num//2) * self.lambda_TAR)
            TSN_coef_chunk = torch.chunk(TSN_coef, TSN_coef.shape[0]//chunk_size, dim = 0)
            

        reward_scores = []
        with torch.cuda.amp.autocast(enabled=self.cfg.use_fp16):
            for chunk_idx, (vid_chunk, cap_chunk) in enumerate(zip(pre_video_chunk, pre_captions_chunk)):
                vid_chunk = rearrange(vid_chunk, 'k s c h w -> (k s) c h w')
                outputs = self.reward_model(vid_chunk, cap_chunk)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                # image_features: (chunk_size*segments) * dim, text_features: chunk_size * dim
                logits_per_image = image_features @ text_features.T
                logits_per_image = rearrange(logits_per_image, '(k s) x-> s k x', s = self.segments)
                diag = torch.diagonal(logits_per_image, dim1 = 1, dim2 = 2) # s, k
                if self.exponential_TSN:
                    diag = TSN_coef_chunk[chunk_idx].permute(1, 0) * diag
                diag = torch.mean(diag, dim = 0) # Average over 'self.segments' dimension to obtain: (chunk_size, )
                reward_scores.append(diag)
        

        ########################## Implementation 2 ###############################
        # GT videos as reward and improve the calculation of log_prob
        reward_scores = torch.cat(reward_scores, dim = 0)
        if self.reward_normalization:
            advantages_spatial = self.reward_tracker.update(prompts = ['']*len(reward_scores), rewards = reward_scores.detach().cpu().numpy())
            advantages_spatial = torch.as_tensor(advantages_spatial).to(self.device)
            advantages_spatial = torch.clamp(advantages_spatial, self.adv_clip_min, self.adv_clip_max)
        else:
            # TODO
            # Not implemented
            # assert False
            advantages_spatial = reward_scores

        ### Improved temporal reward
        advantages_motion = 0
        advantages = self.ST_reward_weights['spatial'] * advantages_spatial + self.ST_reward_weights['temporal'] * advantages_motion
        
        obs_str = f"advantages: {advantages.mean().item():.5f}"

        return advantages


def compute_smoothness_metric(video_data, metric = []):
    b, f, h, w, c = video_data.shape
    
    grey_videos = []
    for b_idx in range(b):
        one_grey_video = []
        for f_idx in range(f):
            one_grey_video.append(cv2.cvtColor(video_data[b_idx][f_idx], cv2.COLOR_RGB2GRAY))
        grey_videos.append(one_grey_video)
    
    ssim_videos = []
    for one_grey_video in grey_videos:
        one_ssim_video = []
        for f_idx in range(f-1):
            one_ssim_video.append(structural_similarity(one_grey_video[f_idx], one_grey_video[f_idx+1]))
        ssim_videos.append(one_ssim_video)
    
    ### Ensure that SSIM is generally high.
    return_metric = {}
    if 'mean' in metric:
        avg_ssim = [np.mean(one_ssim_video) for one_ssim_video in ssim_videos]
        return_metric['mean'] = avg_ssim
    if 'std' in metric:
        std_ssim = [np.std(one_ssim_video) for one_ssim_video in ssim_videos]
        return_metric['std'] = std_ssim

    return return_metric
