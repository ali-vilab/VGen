import os
import sys
import imageio
import numpy as np
import os.path as osp
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-2]))
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

import utils.transforms as data
from tools.modules.config import cfg
from utils.config import Config as pConfig
from utils.registry_class import ENGINE, DATASETS

from tools import *

def test_video_dataset():
    cfg_update = pConfig(load=True)

    for k, v in cfg_update.cfg_dict.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    exp_name = os.path.basename(cfg.cfg_file).split('.')[0]
    save_dir = os.path.join('workspace', 'test_data/datasets', cfg.vid_dataset['type'], exp_name)
    os.system('rm -rf %s' % (save_dir))
    os.makedirs(save_dir, exist_ok=True)

    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])
    vit_trans = T.Compose([
        data.CenterCropWide(cfg.vit_resolution),
        T.ToTensor(),
        T.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    video_mean = torch.tensor(cfg.mean).view(1, -1, 1, 1) #n c f h w
    video_std = torch.tensor(cfg.std).view(1, -1, 1, 1) #n c f h w

    img_mean = torch.tensor(cfg.mean).view(-1, 1, 1) # c f h w
    img_std = torch.tensor(cfg.std).view(-1, 1, 1) # c f h w

    vit_mean = torch.tensor(cfg.vit_mean).view(-1, 1, 1) # c f h w
    vit_std = torch.tensor(cfg.vit_std).view(-1, 1, 1) # c f h w

    txt_size = cfg.resolution[1]
    nc = int(38 * (txt_size / 256))
    font = ImageFont.truetype('data/font/DejaVuSans.ttf', size=13)

    dataset = DATASETS.build(cfg.vid_dataset, sample_fps=cfg.sample_fps[0], transforms=train_trans, vit_transforms=vit_trans)
    print('There are %d videos' % (len(dataset)))
    for idx, item in enumerate(dataset):
        ref_frame, vit_frame, video_data, caption, video_key = item

        video_data = video_data.mul_(video_std).add_(video_mean)
        video_data.clamp_(0, 1)
        video_data = video_data.permute(0, 2, 3, 1)
        video_data = [(image.numpy() * 255).astype('uint8') for image in video_data]

        # Single Image
        ref_frame = ref_frame.mul_(img_mean).add_(img_std)
        ref_frame.clamp_(0, 1)
        ref_frame = ref_frame.permute(1, 2, 0)
        ref_frame = (ref_frame.numpy() * 255).astype('uint8')

        # Text image
        txt_img = Image.new("RGB", (txt_size, txt_size), color="white") 
        draw = ImageDraw.Draw(txt_img)
        lines = "\n".join(caption[start:start + nc] for start in range(0, len(caption), nc))
        draw.text((0, 0), lines, fill="black", font=font)
        txt_img = np.array(txt_img)

        video_data = [np.concatenate([ref_frame, u, txt_img], axis=1) for u in video_data]
        spath = os.path.join(save_dir, '%04d.gif' % (idx))
        imageio.mimwrite(spath, video_data, fps =8)

        # if idx > 100: break


def test_vit_image(test_video_flag=True):
    cfg_update = pConfig(load=True)

    for k, v in cfg_update.cfg_dict.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    exp_name = os.path.basename(cfg.cfg_file).split('.')[0]
    save_dir = os.path.join('workspace', 'test_data/datasets', cfg.img_dataset['type'], exp_name)
    os.system('rm -rf %s' % (save_dir))
    os.makedirs(save_dir, exist_ok=True)

    train_trans = data.Compose([
        data.CenterCropWide(size=cfg.resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.mean, std=cfg.std)])
    vit_trans = data.Compose([
        data.CenterCropWide(cfg.resolution),
        data.Resize(cfg.vit_resolution),
        data.ToTensor(),
        data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)])

    img_mean = torch.tensor(cfg.mean).view(-1, 1, 1) # c f h w
    img_std = torch.tensor(cfg.std).view(-1, 1, 1) # c f h w
    
    vit_mean = torch.tensor(cfg.vit_mean).view(-1, 1, 1) # c f h w
    vit_std = torch.tensor(cfg.vit_std).view(-1, 1, 1) # c f h w

    txt_size = cfg.resolution[1]
    nc = int(38 * (txt_size / 256))
    font = ImageFont.truetype('artist/font/DejaVuSans.ttf', size=13)

    dataset = DATASETS.build(cfg.img_dataset, transforms=train_trans, vit_transforms=vit_trans)
    print('There are %d videos' % (len(dataset)))
    for idx, item in enumerate(dataset):
        ref_frame, vit_frame, video_data, caption, video_key = item
        video_data = video_data.mul_(img_std).add_(img_mean)
        video_data.clamp_(0, 1)
        video_data = video_data.permute(0, 2, 3, 1)
        video_data = [(image.numpy() * 255).astype('uint8') for image in video_data]

        # Single Image
        vit_frame = vit_frame.mul_(vit_std).add_(vit_mean)
        vit_frame.clamp_(0, 1)
        vit_frame = vit_frame.permute(1, 2, 0)
        vit_frame = (vit_frame.numpy() * 255).astype('uint8')

        zero_frame = np.zeros((cfg.resolution[1], cfg.resolution[1], 3), dtype=np.uint8)
        zero_frame[:vit_frame.shape[0], :vit_frame.shape[1], :] = vit_frame

        # Text image
        txt_img = Image.new("RGB", (txt_size, txt_size), color="white") 
        draw = ImageDraw.Draw(txt_img)
        lines = "\n".join(caption[start:start + nc] for start in range(0, len(caption), nc))
        draw.text((0, 0), lines, fill="black", font=font)
        txt_img = np.array(txt_img)

        video_data = [np.concatenate([zero_frame, u, txt_img], axis=1) for u in video_data]
        spath = os.path.join(save_dir, '%04d.gif' % (idx))
        imageio.mimwrite(spath, video_data, fps =8)

        # if idx > 100: break


if __name__ == '__main__':
    # test_video_dataset()
    test_vit_image()

