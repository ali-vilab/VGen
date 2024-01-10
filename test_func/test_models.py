import os
import sys
import torch
import imageio
import numpy as np
import os.path as osp
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-2]))
from thop import profile
from ptflops import get_model_complexity_info

import artist.data as data
from tools.modules.config import cfg
from utils.config import Config as pConfig
from utils.registry_class import ENGINE, MODEL


def test_model():
    cfg_update = pConfig(load=True)

    for k, v in cfg_update.cfg_dict.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v

    model = MODEL.build(cfg.UNet)
    print(int(sum(p.numel() for k, p in model.named_parameters()) / (1024 ** 2)), 'M parameters')
    
    # state_dict = torch.load('cache/pretrain_model/jiuniu_0600000.pth', map_location='cpu')
    # model.load_state_dict(state_dict, strict=False)
    model = model.cuda()

    x = torch.Tensor(1, 4, 16, 32, 56).cuda()
    t = torch.Tensor(1).cuda()
    sims = torch.Tensor(1, 32).cuda()
    fps = torch.Tensor([8]).cuda()
    y = torch.Tensor(1, 1, 1024).cuda()
    image = torch.Tensor(1, 3, 256, 448).cuda()
    
    ret = model(x=x, t=t, y=y, ori_img=image, sims=sims, fps=fps)
    print('Out shape if {}'.format(ret.shape))

    # flops, params = profile(model=model, inputs=(x, t, y, image, sims, fps))
    # print('Model: {:.2f} GFLOPs and {:.2f}M parameters'.format(flops/1e9, params/1e6))

    def prepare_input(resolution):
        return dict(x=[x, t, y, image, sims, fps])

    flops, params = get_model_complexity_info(model, (1, 4, 16, 32, 56), 
        input_constructor = prepare_input,
        as_strings=True, print_per_layer_stat=True)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)

if __name__ == '__main__':
    test_model()
