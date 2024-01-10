import os
import torch
import pynvml
import logging
from einops import rearrange
import torch.cuda.amp as amp

from utils.video_op import save_video_refimg_and_text
from utils.registry_class import VISUAL


@VISUAL.register_class()
class VisualTrainTextToVideo(object):
    def __init__(self, cfg_global, autoencoder, diffusion, viz_num, partial_keys=[], guide_scale=9.0, use_offset_noise=None, **kwargs):
        super(VisualTrainTextToVideo, self).__init__(**kwargs)
        self.cfg = cfg_global
        self.viz_num = viz_num
        self.diffusion = diffusion
        self.autoencoder = autoencoder
        self.guide_scale = guide_scale
        self.partial_keys_list = partial_keys
        self.use_offset_noise = use_offset_noise

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
            **kwargs):
        cfg = self.cfg
        viz_num = self.viz_num

        noise = torch.randn_like(video_data[:viz_num]) # viz_num: 8
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
            file_name = f'rank_{cfg.world_size:02d}_{cfg.rank:02d}_{cfg.sample_fps:02d}_{pre_name}'
            local_path = os.path.join(cfg.log_dir, f'sample_{step:06d}/{file_name}')
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                save_video_refimg_and_text(local_path, ref_frame.cpu(), video_data.cpu(),  captions, cfg.mean, cfg.std, text_size)
            except Exception as e:
                logging.info(f'Step: {step} save text or video error with {e}')
    

