import os
import sys
import torch
import imageio
import numpy as np
import os.path as osp
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-2]))
from PIL import Image, ImageDraw, ImageFont

from einops import rearrange

from tools import *
import utils.transforms as data
from utils.seed import setup_seed
from tools.modules.config import cfg
from utils.config import Config as pConfig
from utils.registry_class import ENGINE, DATASETS, AUTO_ENCODER


def test_enc_dec(gpu=0):
    setup_seed(0)
    cfg_update = pConfig(load=True)

    for k, v in cfg_update.cfg_dict.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    
    save_dir = os.path.join('workspace/test_data/autoencoder', cfg.auto_encoder['type'])
    os.system('rm -rf %s' % (save_dir))
    os.makedirs(save_dir, exist_ok=True)

    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])
    
    vit_trans = data.Compose([
        data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])) if cfg.resolution[0]>cfg.vit_resolution[0] else data.CenterCropWide(size=cfg.vit_resolution),
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    video_mean = torch.tensor(cfg.mean).view(1, -1, 1, 1) #n c f h w
    video_std = torch.tensor(cfg.std).view(1, -1, 1, 1) #n c f h w

    txt_size = cfg.resolution[1]
    nc = int(38 * (txt_size / 256))
    font = ImageFont.truetype('data/font/DejaVuSans.ttf', size=13)

    dataset = DATASETS.build(cfg.vid_dataset, sample_fps=4, transforms=train_trans, vit_transforms=vit_trans)
    print('There are %d videos' % (len(dataset)))

    autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
    autoencoder.eval() # freeze
    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.to(gpu)
    for idx, item in enumerate(dataset):
        local_path = os.path.join(save_dir, '%04d.mp4' % idx)
        # ref_frame, video_data, caption = item
        ref_frame, vit_frame, video_data = item[:3]
        video_data = video_data.to(gpu)

        image_list = []
        video_data_list = torch.chunk(video_data, video_data.shape[0]//cfg.chunk_size,dim=0)
        with torch.no_grad():
            decode_data = []
            for chunk_data in video_data_list:
                latent_z = autoencoder.encode_firsr_stage(chunk_data).detach()
                # latent_z = get_first_stage_encoding(encoder_posterior).detach()
                kwargs = {"timesteps": chunk_data.shape[0]}
                recons_data = autoencoder.decode(latent_z, **kwargs)

                vis_data = torch.cat([chunk_data, recons_data], dim=2).cpu()
                vis_data = vis_data.mul_(video_std).add_(video_mean)  # 8x3x16x256x384
                vis_data = vis_data.cpu()
                vis_data.clamp_(0, 1)
                vis_data = vis_data.permute(0, 2, 3, 1)
                vis_data = [(image.numpy() * 255).astype('uint8') for image in vis_data]
                image_list.extend(vis_data)
        
        num_image = len(image_list)
        frame_dir = os.path.join(save_dir, 'temp')
        os.makedirs(frame_dir, exist_ok=True)
        for idx in range(num_image):
            tpth = os.path.join(frame_dir, '%04d.png' % (idx+1))
            cv2.imwrite(tpth, image_list[idx][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate 8 -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
        os.system(cmd); os.system(f'rm -rf {frame_dir}')


if __name__ == '__main__':
    test_enc_dec()
