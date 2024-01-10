import torch
import logging
import os.path as osp
from datetime import datetime
from easydict import EasyDict
import os

cfg = EasyDict(__name__='Config: VideoLDM Decoder')

# -------------------------------distributed training--------------------------
pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
gpus_per_machine = torch.cuda.device_count()
world_size = pmi_world_size * gpus_per_machine
# -----------------------------------------------------------------------------


# ---------------------------Dataset Parameter---------------------------------
cfg.mean = [0.5, 0.5, 0.5]
cfg.std = [0.5, 0.5, 0.5]
cfg.max_words = 1000
cfg.num_workers = 8
cfg.prefetch_factor = 2

# PlaceHolder
cfg.resolution = [448, 256]
cfg.vit_out_dim = 1024
cfg.vit_resolution = 336
cfg.depth_clamp = 10.0
cfg.misc_size = 384
cfg.depth_std = 20.0

cfg.frame_lens = [32, 32, 32, 1]
cfg.sample_fps = [4, ]
cfg.vid_dataset = {
    'type': 'VideoBaseDataset',
    'data_list': [],
    'max_words': cfg.max_words,
    'resolution': cfg.resolution}
cfg.img_dataset = {
    'type': 'ImageBaseDataset',
    'data_list': ['laion_400m',],
    'max_words': cfg.max_words,
    'resolution': cfg.resolution}

cfg.batch_sizes = {
    str(1):256,
    str(4):4,
    str(8):4,
    str(16):4}
# -----------------------------------------------------------------------------


# ---------------------------Mode Parameters-----------------------------------
# Diffusion
cfg.Diffusion = {
    'type': 'DiffusionDDIM',
    'schedule': 'cosine', # cosine
    'schedule_param': {
        'num_timesteps': 1000,
        'cosine_s': 0.008,
        'zero_terminal_snr': True,
    },
    'mean_type': 'v',           # [v, eps]
    'loss_type': 'mse',
    'var_type': 'fixed_small',
    'rescale_timesteps': False,
    'noise_strength': 0.1,
    'ddim_timesteps': 50
}
cfg.ddim_timesteps = 50  # official: 250
cfg.use_div_loss = False
# classifier-free guidance
cfg.p_zero = 0.9
cfg.guide_scale = 3.0

# clip vision encoder
cfg.vit_mean = [0.48145466, 0.4578275, 0.40821073]
cfg.vit_std = [0.26862954, 0.26130258, 0.27577711]

# Model
cfg.scale_factor = 0.18215  
cfg.use_checkpoint = True
cfg.use_sharded_ddp = False
cfg.use_fsdp = False 
cfg.use_fp16 = True
cfg.temporal_attention = True

cfg.UNet = {
    'type': 'UNetSD',
    'in_dim': 4,
    'dim': 320,
    'y_dim': cfg.vit_out_dim,
    'context_dim': 1024,
    'out_dim': 8,
    'dim_mult': [1, 2, 4, 4],
    'num_heads': 8,
    'head_dim': 64,
    'num_res_blocks': 2,
    'attn_scales': [1 / 1, 1 / 2, 1 / 4],
    'dropout': 0.1,
    'temporal_attention': cfg.temporal_attention,
    'temporal_attn_times': 1,
    'use_checkpoint': cfg.use_checkpoint,
    'use_fps_condition': False,
    'use_sim_mask': False
}

# auotoencoder from stabel diffusion
cfg.guidances = []
cfg.auto_encoder = {
    'type': 'AutoencoderKL',
    'ddconfig': {
        'double_z': True, 
        'z_channels': 4,
        'resolution': 256, 
        'in_channels': 3,
        'out_ch': 3, 
        'ch': 128, 
        'ch_mult': [1, 2, 4, 4],
        'num_res_blocks': 2, 
        'attn_resolutions': [], 
        'dropout': 0.0,
        'video_kernel_size': [3, 1, 1]
    },
    'embed_dim': 4,
    'pretrained': 'models/v2-1_512-ema-pruned.ckpt'
}
# clip embedder
cfg.embedder = {
    'type': 'FrozenOpenCLIPEmbedder',
    'layer': 'penultimate',
    'pretrained': 'models/open_clip_pytorch_model.bin'
}
# -----------------------------------------------------------------------------

# ---------------------------Training Settings---------------------------------
# training and optimizer
cfg.ema_decay = 0.9999
cfg.num_steps = 600000
cfg.lr = 5e-5
cfg.weight_decay = 0.0
cfg.betas = (0.9, 0.999)
cfg.eps = 1.0e-8
cfg.chunk_size = 16
cfg.decoder_bs = 8
cfg.alpha = 0.7
cfg.save_ckp_interval = 1000

# scheduler
cfg.warmup_steps = 10
cfg.decay_mode = 'cosine'

# acceleration
cfg.use_ema = True  
if world_size<2:
    cfg.use_ema = False
cfg.load_from = None
# -----------------------------------------------------------------------------


# ----------------------------Pretrain Settings---------------------------------
cfg.Pretrain = {
    'type': 'pretrain_specific_strategies',
    'fix_weight': False,
    'grad_scale': 0.2,
    'resume_checkpoint': 'models/jiuniu_0267000.pth',
    'sd_keys_path': 'models/stable_diffusion_image_key_temporal_attention_x1.json',
}
# -----------------------------------------------------------------------------


# -----------------------------Visual-------------------------------------------
# Visual videos
cfg.viz_interval = 1000
cfg.visual_train = {
    'type': 'VisualTrainTextImageToVideo',
}
cfg.visual_inference = {
    'type': 'VisualGeneratedVideos',
}
cfg.inference_list_path = ''

# logging
cfg.log_interval = 100

### Default log_dir
cfg.log_dir = 'workspace/temp_dir'
# -----------------------------------------------------------------------------


# ---------------------------Others--------------------------------------------
# seed 
cfg.seed = 8888
cfg.negative_prompt = 'Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms'
# -----------------------------------------------------------------------------

