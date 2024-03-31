import os
import re
import torch
import pynvml
import logging
from einops import rearrange
import torch.cuda.amp as amp
from PIL import Image

from utils.video_op import save_video_refimg_and_text, save_i2vgen_video_safe
from utils.registry_class import VISUAL


@VISUAL.register_class()
class VisualTrainDreamVideo(object):
    def __init__(self, cfg_global, autoencoder, clip_encoder, diffusion, embedding_manager, vit_transforms, viz_num, use_clip_adapter_condition, partial_keys=[], guide_scale=9.0, use_offset_noise=None, 
                    infer_with_custom_text=False, data_list=[], data_dir_list=[], **kwargs):
        super(VisualTrainDreamVideo, self).__init__(**kwargs)
        self.cfg = cfg_global
        self.viz_num = viz_num
        self.diffusion = diffusion
        self.autoencoder = autoencoder
        self.clip_encoder = clip_encoder
        self.vit_trans = vit_transforms
        self.embedding_manager = embedding_manager
        self.use_clip_adapter_condition = use_clip_adapter_condition
        self.guide_scale = guide_scale
        self.partial_keys_list = partial_keys
        self.use_offset_noise = use_offset_noise
        self.infer_with_custom_text = infer_with_custom_text

        if infer_with_custom_text:
            image_list = []
            for item_path, data_dir in zip(data_list, data_dir_list):
                lines = open(item_path, 'r').readlines()
                lines = [[data_dir, item.strip()] for item in lines]
                image_list.extend(lines)
            self.image_list = image_list

    def prepare_model_kwargs(self, partial_keys, full_model_kwargs):
        """
        """
        model_kwargs = [{}, {}]
        for partial_key in partial_keys:
            model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
            model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]
        return model_kwargs
    
    @torch.no_grad()
    def run(self, 
            model, 
            video_data,
            captions,
            step=0,
            ref_frame=None,
            visual_kwards=[],
            zero_y=None,
            zero_feature=None,
            **kwargs):
        cfg = self.cfg
        viz_num = self.viz_num

        if video_data.shape[2] == 1:
            # repeat image to generate video
            video_data = video_data.repeat(1, 1, cfg.gen_frames, 1, 1)

        noise = torch.randn_like(video_data[:viz_num])
        if self.use_offset_noise:
            noise_strength = getattr(cfg, 'noise_strength', 0)
            b, c, f, *_ = video_data[:viz_num].shape
            noise = noise + noise_strength * torch.randn(b, c, f, 1, 1, device=video_data.device)
        
        # print memory
        pynvml.nvmlInit()
        handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
        logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')

        for keys in self.partial_keys_list:
            model_kwargs = self.prepare_model_kwargs(keys, visual_kwards)
            pre_name = '_'.join(keys)
            with amp.autocast(enabled=cfg.use_fp16):
                video_data = self.diffusion.ddim_sample_loop(
                    noise=noise.clone(),
                    model=model.eval(),
                    model_kwargs=model_kwargs,
                    guide_scale=self.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
            
            video_data = 1. / cfg.scale_factor * video_data
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
            file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{cfg.sample_fps:02d}_{pre_name}'
            local_path = os.path.join(cfg.log_dir, f'sample_{step:06d}/{file_name}')
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                save_video_refimg_and_text(local_path, ref_frame.cpu(), video_data.cpu(),  captions, cfg.mean, cfg.std, text_size)
            except Exception as e:
                logging.info(f'Step: {step} save text or video error with {e}')

            if self.infer_with_custom_text:
                self.infer_custom_text(model, zero_y, zero_feature, step)

    def infer_custom_text(self, model, zero_y, zero_feature, step):
        cfg = self.cfg
        noise_strength = getattr(cfg, 'noise_strength', 0)

        logging.info(f'There are {len(self.image_list)} videos for inference.')
        
        for idx, line in enumerate(self.image_list):
            data_dir = line[0]
            img_key, caption = line[1].split('|||')
            captions = [caption]
            file_path = os.path.join(data_dir, img_key)
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            vit_tensor = self.vit_trans(image)
            vit_tensor = vit_tensor.unsqueeze(0)
            
            with torch.no_grad():
                y_image, _, y = self.clip_encoder(text=captions, image=vit_tensor, embedding_manager=self.embedding_manager)
                y_image = y_image.unsqueeze(1)

                pynvml.nvmlInit()
                handle=pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo=pynvml.nvmlDeviceGetMemoryInfo(handle)
                logging.info(f'GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB')
                # sample images (DDIM)
                with amp.autocast(enabled=cfg.use_fp16):
                    batch_size = vit_tensor.shape[0]

                    noise = torch.randn([batch_size, 4, cfg.gen_frames, cfg.resolution[1]//8, cfg.resolution[0]//8])
                    noise = noise.cuda()
                    if noise_strength > 0:
                        b, c, f, *_ = noise.shape
                        offset_noise = torch.randn(b, c, f, 1, 1, device=noise.device)
                        noise = noise + noise_strength * offset_noise
                    noise = noise.contiguous()
                    
                    model_kwargs=[ {'y': y}, {'y': zero_y.repeat(batch_size, 1, 1)}]
                    if self.use_clip_adapter_condition:
                        model_kwargs[0]['y_image'] = y_image
                        model_kwargs[1]['y_image'] = zero_feature.repeat(batch_size, 1, 1)
                    video_data = self.diffusion.ddim_sample_loop(
                        noise=noise.clone(),
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
                gen_frames = self.autoencoder.decode(vd_data)
                decode_data.append(gen_frames)
            video_data = torch.cat(decode_data, dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = batch_size)

            text_size = cfg.resolution[-1]
            cap_name = re.sub(r'[^\w\s\*]', '', caption).replace(' ', '_')
            file_name = f'{cap_name}_{cfg.seed}_{idx}.mp4'
            local_path = os.path.join(cfg.log_dir, f'sample_{step:06d}/{file_name}')
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            try:
                save_i2vgen_video_safe(local_path, video_data.cpu(), captions, cfg.mean, cfg.std, text_size)
            except Exception as e:
                logging.info(f'Step: {step} save text or video error with {e}')
