import os
import sys
import json
import torch
import imageio
import numpy as np
import os.path as osp
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-2]))
from thop import profile
from ptflops import get_model_complexity_info

import artist.data as data
from tools.modules.config import cfg
from tools.modules.unet.util import *
from utils.config import Config as pConfig
from utils.registry_class import ENGINE, MODEL


def save_temporal_key():
    cfg_update = pConfig(load=True)

    for k, v in cfg_update.cfg_dict.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    model = MODEL.build(cfg.UNet)

    temp_name = ''
    temp_key_list = []
    spth = 'workspace/module_list/UNetSD_I2V_vs_Text_temporal_key_list.json'
    for name, module in model.named_modules():    
        if isinstance(module, (TemporalTransformer, TemporalTransformer_attemask, TemporalAttentionBlock, TemporalAttentionMultiBlock, TemporalConvBlock_v2, TemporalConvBlock)):
            temp_name = name
            print(f'Model: {name}')
        elif isinstance(module, (ResidualBlock, ResBlock, SpatialTransformer, Upsample, Downsample)):
            temp_name = ''

        if hasattr(module, 'weight'):
            if temp_name != '' and (temp_name in name):
                temp_key_list.append(name)
                print(f'{name}')
        # print(name)
    
    save_module_list = []
    for k, p in model.named_parameters():
        for item in temp_key_list:
            if item in k:
                print(f'{item} --> {k}')
                save_module_list.append(k)

    print(int(sum(p.numel() for k, p in model.named_parameters()) / (1024 ** 2)), 'M parameters')
    
    # spth = 'workspace/module_list/{}'
    json.dump(save_module_list, open(spth, 'w'))
    a = 0


def save_spatial_key():
    cfg_update = pConfig(load=True)

    for k, v in cfg_update.cfg_dict.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    model = MODEL.build(cfg.UNet)
    temp_name = ''
    temp_key_list = []
    spth = 'workspace/module_list/UNetSD_I2V_HQ_P_spatial_key_list.json'
    for name, module in model.named_modules():    
        if isinstance(module, (ResidualBlock, ResBlock, SpatialTransformer, Upsample, Downsample)):
            temp_name = name
            print(f'Model: {name}')
        elif isinstance(module, (TemporalTransformer, TemporalTransformer_attemask, TemporalAttentionBlock, TemporalAttentionMultiBlock, TemporalConvBlock_v2, TemporalConvBlock)):
            temp_name = ''

        if hasattr(module, 'weight'):
            if temp_name != '' and (temp_name in name):
                temp_key_list.append(name)
                print(f'{name}')
        # print(name)
    
    save_module_list = []
    for k, p in model.named_parameters():
        for item in temp_key_list:
            if item in k:
                print(f'{item} --> {k}')
                save_module_list.append(k)

    print(int(sum(p.numel() for k, p in model.named_parameters()) / (1024 ** 2)), 'M parameters')
    
    # spth = 'workspace/module_list/{}'
    json.dump(save_module_list, open(spth, 'w'))
    a = 0


if __name__ == '__main__':
    # save_temporal_key()
    save_spatial_key()



# print([k for (k, _) in self.input_blocks.named_parameters()])


