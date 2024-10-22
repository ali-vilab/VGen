import os
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
import json
import math
import random
import torch
import logging
import datetime
import numpy as np
from PIL import Image
import torch.optim as optim 
from einops import rearrange
import torch.cuda.amp as amp
from importlib import reload
from copy import deepcopy, copy
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import utils.transforms as data
from utils.util import to_device
from ..modules.config import cfg
from utils.seed import setup_seed
from utils.optim import AnnealingLR
from utils.multi_port import find_free_port
from utils.distributed import generalized_all_gather, all_reduce
from utils.registry_class import ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION, PRETRAIN

from ..modules.diffusions.schedules import beta_schedule
from diffusers.schedulers import LCMScheduler, DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
MAX_SEQ_LENGTH = 77

@ENGINE.register_function()
def train_videolcm_t2v_entrance(cfg_update,  **kwargs):
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





def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f

def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    return tensor[t].view(shape).to(x)





# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb



def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def predicted_origin(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, use_auth_token=True
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")



# Adapted from pipelines.StableDiffusionPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device))[0]

    return prompt_embeds



def worker(gpu, cfg):
    '''
    Training worker for each gpu
    '''
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu
    
    if not cfg.debug:
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)
    
    # [Log] Save logging
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    exp_name = osp.basename(cfg.cfg_file).split('.')[0]
    cfg.log_dir = osp.join(cfg.log_dir, exp_name)
    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.rank == 0:
        log_file = osp.join(cfg.log_dir, 'log.txt')
        cfg.log_file = log_file
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)
        logging.info(f'Save all the file in to dir {cfg.log_dir}')
        logging.info(f"Going into i2v_img_fullid_vidcom function on {gpu} gpu")

    # [Diffusion]  build diffusion settings
    # diffusion = DIFFUSION.build(cfg.Diffusion)

    # [Dataset] imagedataset and videodataset
    len_frames = len(cfg.frame_lens)
    len_fps = len(cfg.sample_fps)
    cfg.max_frames = cfg.frame_lens[cfg.rank % len_frames]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]
    cfg.sample_fps = cfg.sample_fps[cfg.rank % len_fps]
    
    if cfg.rank == 0:
        logging.info(f'Currnt worker with max_frames={cfg.max_frames}, batch_size={cfg.batch_size}, sample_fps={cfg.sample_fps}')

    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])
    # from ipdb import set_trace; set_trace()
    vit_trans = data.Compose([
        data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])) if cfg.resolution[0]>cfg.vit_resolution[0] else data.CenterCropWide(size=cfg.vit_resolution),
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])
    
    if cfg.max_frames == 1:
        cfg.sample_fps = 1
        dataset = DATASETS.build(cfg.img_dataset, transforms=train_trans, vit_transforms=vit_trans)
    else:
        dataset = DATASETS.build(cfg.vid_dataset, sample_fps=cfg.sample_fps, transforms=train_trans, vit_transforms=vit_trans, max_frames=cfg.max_frames)
    
    sampler = DistributedSampler(dataset, num_replicas=cfg.world_size, rank=cfg.rank) if (cfg.world_size > 1 and not cfg.debug) else None
    dataloader = DataLoader(
        dataset, 
        sampler=sampler,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        prefetch_factor=cfg.prefetch_factor)
    rank_iter = iter(dataloader) 
    
    # [Model] embedder
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
    model = MODEL.build(cfg.UNet, zero_y=zero_y_negative)
    model = model.to(gpu)

    teacher_unet = MODEL.build(cfg.UNet, zero_y=zero_y,)
    teacher_unet = teacher_unet.to(gpu)

    target_unet = MODEL.build(cfg.UNet, zero_y=zero_y, )
    target_unet = target_unet.to(gpu)

    resume_step = 1
    model, resume_step = PRETRAIN.build(cfg.Pretrain, model=model)
    torch.cuda.empty_cache()

    if cfg.use_ema:
        ema = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        ema = type(ema)([(k, ema[k].data.clone()) for k in list(ema.keys())[cfg.rank::cfg.world_size]])
    
    teacher_unet.load_state_dict(model.state_dict())
    teacher_unet.requires_grad_(False)
    target_unet.load_state_dict(model.state_dict())
    target_unet.train()
    target_unet.requires_grad_(False)

    if hasattr(cfg, "set_target_eval") and cfg.set_target_eval:
        target_unet.eval()
        teacher_unet.eval()

    # optimizer
    optimizer = optim.AdamW(params=model.parameters(),
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler(enabled=cfg.use_fp16)
    if cfg.use_fsdp:
        config = {}
        config['compute_dtype'] = torch.float32
        config['mixed_precision'] = True
        model = FSDP(model, **config)
    else:
        model = DistributedDataParallel(model, device_ids=[gpu]) if not cfg.debug else model.to(gpu)

    # scheduler
    scheduler = AnnealingLR(
        optimizer=optimizer,
        base_lr=cfg.lr,
        warmup_steps=cfg.warmup_steps,  # 10
        total_steps=cfg.num_steps,      # 200000
        decay_mode=cfg.decay_mode)      # 'cosine'
    
    # [Visual]
    viz_num = min(cfg.batch_size, 8)
    # visual_func = VISUAL.build(
    #     cfg.visual_train,
    #     cfg_global=cfg,
    #     viz_num=viz_num, 
    #     diffusion=diffusion, 
    #     autoencoder=autoencoder)
    
    ### Generators for various conditions
    # if 'depthmap' in cfg.video_compositions:
    #     from tools.annotator.depth import midas_v3
    #     midas = models.midas_v3(pretrained=True).eval().requires_grad_(False).to(
    #         memory_format=torch.channels_last).half().to(gpu)
    # if 'canny' in cfg.video_compositions:
    #     from tools.annotator.canny import CannyDetector
    #     canny_detector = CannyDetector()
    # if 'sketch' in cfg.video_compositions or 'single_sketch' in cfg.video_compositions:
    #     from tools.annotator.sketch import pidinet_bsd, sketch_simplification_gan
    #     pidinet = pidinet_bsd(pretrained=True, vanilla_cnn=True).eval().requires_grad_(False).to(gpu)
    #     cleaner = sketch_simplification_gan(pretrained=True).eval().requires_grad_(False).to(gpu)
    #     pidi_mean = torch.tensor(cfg.sketch_mean).view(1, -1, 1, 1).to(gpu)
    #     pidi_std = torch.tensor(cfg.sketch_std).view(1, -1, 1, 1).to(gpu)
    # if 'histogram' in cfg.video_compositions:
    #     # This code defines a color palette with 11 hues, 5 saturation levels, and 4 lightness levels. 
    #     # It then converts the palette colors from the CIELAB color space to the RGB color space and stores them in two tensors: lab_codebook and rgb_codebook. 
    #     # The codebook_size variable is set to the number of colors in the palette.
    #     # The code calculates a histogram filter using the lab_codebook tensor to represent color similarity between colors in the palette.
    #     # The filter is computed using the Euclidean distance between each color in the palette and all other colors in the palette, and then applying a Gaussian function with a standard deviation of cfg.hist_sigma.
    #     # Finally, the code normalizes the histogram filter so that each row of the resulting matrix sums to one.
    #     palette = Palette(num_hues=11, num_sat=5, num_light=4)
    #     lab_codebook = torch.from_numpy(palette.lab).float().to(gpu) # torch.Size([156, 3])
    #     rgb_codebook = torch.from_numpy(palette.rgb).float().to(gpu) # torch.Size([156, 3])
    #     codebook_size = len(lab_codebook)
    #     hist_filter = torch.exp(-torch.cdist(lab_codebook, lab_codebook).pow(2) / (2.0 * cfg.hist_sigma ** 2))
    #     hist_filter = hist_filter / hist_filter.sum(dim=1, keepdim=True)
    # else:
    #     palette = None

    
    

    betas = beta_schedule(schedule='linear_sd', num_timesteps=cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120, zero_terminal_snr=getattr(cfg, "zero_terminal_snr", False))

    # prediction_type_a = "epsilon"
    prediction_type_a = "v_prediction"
    if hasattr(cfg, "prediction_type_a"):
        prediction_type_a = cfg.prediction_type_a
    
    noise_scheduler = DDPMScheduler(beta_start=0.00085,beta_end=0.012,
                    beta_schedule= "scaled_linear",
                    clip_sample= False,
                    clip_sample_range= 1.0,
                    dynamic_thresholding_ratio= 0.995,
                    num_train_timesteps= 1000,
                    prediction_type= prediction_type_a,
                    sample_max_value= 1.0,
                    # set_alpha_to_one= False,
                    # skip_prk_steps= True,
                    steps_offset= 1,
                    thresholding= False,
                    timestep_spacing= "linspace", # "leading"
                    # trained_betas= Null,
                    variance_type= "fixed_small")
    if hasattr(cfg, "zero_terminal_snr") and cfg.zero_terminal_snr:
        noise_scheduler.betas = betas
        noise_scheduler.alphas = 1.0 - noise_scheduler.betas
        noise_scheduler.alphas_cumprod = torch.cumprod(noise_scheduler.alphas, dim=0)
    # The scheduler calculates the alpha and sigma schedule for us
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
    # set_trace()
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.numpy(),
        timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=cfg.ddim_timesteps,
    )
    diffusion_eval = LCMScheduler(prediction_type=prediction_type_a, beta_schedule= "scaled_linear",clip_sample=False, timestep_spacing="linspace", rescale_betas_zero_snr=getattr(cfg, "zero_terminal_snr", False))
    # diffusion_eval = noise_scheduler
    # diffusion_eval = DDIMScheduler(prediction_type="v_prediction",beta_schedule= "scaled_linear",clip_sample=False, timestep_spacing="linspace")  # worked
    num_inference_steps = 4
    if hasattr(cfg, "num_inference_steps"):
        num_inference_steps = cfg.num_inference_steps
    

    diffusion_eval.set_timesteps(num_inference_steps, device=model.device)
    # timesteps = diffusion_eval.timesteps

    # Also move the alpha and sigma noise schedules to accelerator.device.
    alpha_schedule = alpha_schedule.to(model.device)
    sigma_schedule = sigma_schedule.to(model.device)
    solver = solver.to(model.device)
    
    for step in range(resume_step, cfg.num_steps + 1): 
        model.train()
        
        try:
            batch = next(rank_iter)
        except StopIteration:
            rank_iter = iter(dataloader)
            batch = next(rank_iter)
        
        batch = to_device(batch, gpu, non_blocking=True)
        ref_frame, _, video_data, captions, video_key = batch
        batch_size, frames_num, _, _, _ = video_data.shape
        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')

        fps_tensor =  torch.tensor([cfg.sample_fps] * batch_size, dtype=torch.long, device=gpu)
        video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
        with torch.no_grad():
            decode_data = []
            for chunk_data in video_data_list:
                latent_z = autoencoder.encode_firsr_stage(chunk_data, cfg.scale_factor).detach()
                decode_data.append(latent_z) # [B, 4, 32, 56]
            video_data = torch.cat(decode_data,dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = batch_size) # [B, 4, 16, 32, 56]

        opti_timesteps = getattr(cfg, 'opti_timesteps', cfg.Diffusion.schedule_param.num_timesteps)
        t_round = torch.randint(0, opti_timesteps, (batch_size, ), dtype=torch.long, device=gpu) # 8

        # preprocess
        with torch.no_grad():
            _, _, y_words = clip_encoder(text=captions) # bs * 1 *1024 [B, 1, 1024]
            y_words_0 = y_words.clone()
            
        
        # forward
        model_kwargs = {'y': y_words,  'fps': fps_tensor}
        model_kwargs_zero = {'y': zero_y_negative.repeat(y_words.shape[0], 1, 1),  'fps': fps_tensor}
        if cfg.use_fsdp:
            loss = diffusion.loss(x0=video_data, 
                t=t_round, model=model, model_kwargs=model_kwargs,
                use_div_loss=cfg.use_div_loss) 
            loss = loss.mean()
        else:
            with amp.autocast(enabled=cfg.use_fp16):
                noise = torch.randn_like(video_data) # [24, 4, 16, 32, 56]
                bsz = video_data.shape[0]
                latents = video_data
                # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
                # topk = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps
                topk = noise_scheduler.config.num_train_timesteps  // cfg.ddim_timesteps
                index = torch.randint(0, cfg.ddim_timesteps, (bsz,), device=video_data.device).long()
                start_timesteps = solver.ddim_timesteps[index]
                timesteps = start_timesteps - topk
                timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

                # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
                c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
                c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
                c_skip, c_out = scalings_for_boundary_conditions(timesteps)
                c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]
                # set_trace()

                # 20.4.5. Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                # 20.4.5. Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
                noisy_model_input = noise_scheduler.add_noise(latents, noise, start_timesteps)
                

                # 20.4.6. Sample a random guidance scale w from U[w_min, w_max] and embed it
                # w = (args.w_max - args.w_min) * torch.rand((bsz,)) + args.w_min
                # w = (15.0 - 5.0) * torch.rand((bsz,)) + 5.0
                # w = (10.0 - 6.0) * torch.rand((bsz,)) + 6.0
                w = (0.0) * torch.rand((bsz,)) + 9.0
                if hasattr(cfg, "set_fixed_guidance") and cfg.set_fixed_guidance:
                    w = (0.0) * torch.rand((bsz,)) + cfg.set_fixed_guidance
                w_embedding = guidance_scale_embedding(w, embedding_dim=320) # args.unet_time_cond_proj_dim
                w = w.reshape(bsz, 1, 1, 1, 1)
                # Move to U-Net device and dtype
                w = w.to(device=latents.device, dtype=latents.dtype)
                w_embedding = w_embedding.to(device=latents.device, dtype=latents.dtype)


                noise_pred =  model(noisy_model_input, start_timesteps, t_w=w_embedding, **model_kwargs)

                pred_x_0 = predicted_origin(
                        noise_pred,
                        start_timesteps,
                        noisy_model_input,
                        prediction_type_a,
                        alpha_schedule,
                        sigma_schedule,
                    )
                # set_trace()
                pred_x_0_clone = pred_x_0.clone()
                model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

                
                # 20.4.10. Use the ODE solver to predict the kth step in the augmented PF-ODE trajectory after
                # noisy_latents with both the conditioning embedding c and unconditional embedding 0
                # Get teacher model prediction on noisy_latents and conditional embedding
                with torch.no_grad():
                    with torch.autocast("cuda"):
                        
                        cond_teacher_output = teacher_unet(
                            noisy_model_input,
                            start_timesteps,
                            **model_kwargs
                        )
                        cond_pred_x0 = predicted_origin(
                            cond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            prediction_type_a,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # Get teacher model prediction on noisy_latents and unconditional embedding
                        
                        uncond_teacher_output = teacher_unet(
                            noisy_model_input,
                            start_timesteps,
                            **model_kwargs_zero,
                        )
                        uncond_pred_x0 = predicted_origin(
                            uncond_teacher_output,
                            start_timesteps,
                            noisy_model_input,
                            prediction_type_a,
                            alpha_schedule,
                            sigma_schedule,
                        )

                        # 20.4.11. Perform "CFG" to get x_prev estimate (using the LCM paper's CFG formulation)
                        pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                        # pred_x0 = video_data # add
                        pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                        # pred_noise = cond_teacher_output # + w * (cond_teacher_output - uncond_teacher_output)
                        
                        # v_prediction to epsilon
                        if prediction_type_a == "v_prediction":
                            # sigmas_a = extract_into_tensor(sigma_schedule, timesteps, noisy_model_input.shape)
                            # alphas_a = extract_into_tensor(alpha_schedule, timesteps, noisy_model_input.shape)
                            sigmas_a = extract_into_tensor(sigma_schedule, start_timesteps, noisy_model_input.shape)
                            alphas_a = extract_into_tensor(alpha_schedule, start_timesteps, noisy_model_input.shape)
                            pred_noise = alphas_a * pred_noise + sigmas_a * noisy_model_input

                        x_prev = solver.ddim_step(pred_x0, pred_noise, index) # 这个有问题

                        # derive variables
                        # xt = pred_noise
                        # x0 = pred_x0
                        # stride = topk
                        # eta = 0.0
                        # eps = (_i(diffusion.sqrt_recip_alphas_cumprod, start_timesteps, xt) * xt - x0) / \
                        #     _i(diffusion.sqrt_recipm1_alphas_cumprod, start_timesteps, xt)
                        # alphas = _i(diffusion.alphas_cumprod, start_timesteps, xt)
                        # alphas_prev = _i(diffusion.alphas_cumprod, (start_timesteps - stride).clamp(0), xt)
                        # sigmas = eta * torch.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

                        # # random sample
                        # noise = torch.randn_like(xt)
                        # direction = torch.sqrt(1 - alphas_prev - sigmas ** 2) * eps
                        # mask = start_timesteps.ne(0).float().view(-1, *((1, ) * (xt.ndim - 1)))
                        # xt_1 = torch.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
                        # x_prev = xt_1

                # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
                with torch.no_grad():
                    # with torch.autocast("cuda", dtype=weight_dtype):
                    with torch.autocast("cuda"):
                        
                        target_noise_pred = target_unet(
                            x_prev.float(),
                            timesteps,
                            # t_w=w_embedding,
                            **model_kwargs,
                        )
                    pred_x_0 = predicted_origin(
                        target_noise_pred,
                        timesteps,
                        x_prev,
                        # noise_scheduler.config.prediction_type,
                        prediction_type_a,
                        alpha_schedule,
                        sigma_schedule,
                    )
                    target = c_skip * x_prev + c_out * pred_x_0 # original 
                    # target = c_skip * x_prev + c_out * pred_x0

                # 20.4.13. Calculate loss
                
                huber_c = 0.001
                # from ipdb import set_trace; set_trace()
                loss = torch.mean(
                    torch.sqrt((model_pred.float() - target.float()) ** 2 + huber_c**2) - huber_c
                )
                # loss = diffusion.loss(
                #         x0=video_data, 
                #         t=t_round, 
                #         model=model, 
                #         model_kwargs=model_kwargs, 
                #         use_div_loss=cfg.use_div_loss) # cfg.use_div_loss: False    loss: [80]
                # loss = loss.mean()
        
        # backward
        if cfg.use_fsdp:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 0.05)
            optimizer.step()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if not cfg.use_fsdp:
            scheduler.step()
        
        # ema update
        if cfg.use_ema:
            temp_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            for k, v in ema.items():
                v.copy_(temp_state_dict[k].lerp(v, cfg.ema_decay))

        update_ema(target_unet.parameters(), model.parameters(), 0.95)
        all_reduce(loss)
        loss = loss / cfg.world_size
        
        if cfg.rank == 0 and step % cfg.log_interval == 0: # cfg.log_interval: 100
            logging.info(f'Step: {step}/{cfg.num_steps} Loss: {loss.item():.3f} scale: {scaler.get_scale():.1f} LR: {scheduler.get_lr():.7f}')

        # Visualization
        # if step == resume_step or step == cfg.num_steps or step % cfg.viz_interval == 0:
        #     with torch.no_grad():
        #         try:
        #             visual_kwards = [
        #                 {
        #                     'y': y_words_0[:viz_num],
        #                     'fps': fps_tensor[:viz_num],
        #                 },
        #                 {
        #                     'y': zero_y_negative.repeat(y_words_0.size(0), 1, 1),
        #                     'fps': fps_tensor[:viz_num],
        #                 }
        #             ]
        #             input_kwards = {
        #                 'model': model, 'video_data': video_data[:viz_num], 'step': step, 
        #                 'ref_frame': ref_frame[:viz_num], 'captions': captions[:viz_num]}
        #             visual_func.run(visual_kwards=visual_kwards, **input_kwards)
        #         except Exception as e:
        #             logging.info(f'Save videos with exception {e}')
        
        # Save checkpoint
        if step == cfg.num_steps or step % cfg.save_ckp_interval == 0 and step >resume_step:
            os.makedirs(osp.join(cfg.log_dir, 'checkpoints'), exist_ok=True)
            if cfg.use_ema:
                local_ema_model_path = osp.join(cfg.log_dir, f'checkpoints/ema_{step:08d}_rank{cfg.rank:04d}.pth')
                save_dict = {
                    'state_dict': ema.module.state_dict() if hasattr(ema, 'module') else ema,
                    'step': step}
                torch.save(save_dict, local_ema_model_path)
                if cfg.rank == 0:
                    logging.info(f'Begin to Save ema model to {local_ema_model_path}')
            if cfg.rank == 0:
                local_model_path = osp.join(cfg.log_dir, f'checkpoints/non_ema_{step:08d}.pth')
                logging.info(f'Begin to Save model to {local_model_path}')
                save_dict = {
                    'state_dict': model.module.state_dict() if not cfg.debug else model.state_dict(),
                    'step': step}
                torch.save(save_dict, local_model_path)
                logging.info(f'Save model to {local_model_path}')
    
    if cfg.rank == 0:
        logging.info('Congratulations! The training is completed!')
    
    # synchronize to finish some processes
    if not cfg.debug:
        torch.cuda.synchronize()
        dist.barrier()

