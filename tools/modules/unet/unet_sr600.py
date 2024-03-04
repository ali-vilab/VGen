'''
/* 
*Copyright (c) 2021, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/
'''

import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from .util import *
from utils.registry_class import MODEL
import torch.fft as fft

USE_TEMPORAL_TRANSFORMER = True


def Fourier_filter(x, threshold, scale):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    x_filtered = x_filtered.type(dtype)
    return x_filtered


@MODEL.register_class()
class UNetSD_SR600(nn.Module):
    def __init__(self,
                 in_dim=7,
                 dim=512,
                 y_dim=512,
                 context_dim=512,
                 out_dim=6,
                 dim_mult=[1, 2, 3, 4],
                 num_heads=None,
                 head_dim=64,
                 num_res_blocks=3,
                 attn_scales=[1 / 2, 1 / 4, 1 / 8],
                 use_scale_shift_norm=True,
                 dropout=0.1,
                 temporal_attn_times=1,
                 temporal_attention = True,
                 use_checkpoint=False,
                 use_image_dataset=False,
                 use_sim_mask = False,
                 inpainting=True,
                 **kwargs):
        embed_dim = dim * 4
        num_heads=num_heads if num_heads else dim//32
        super(UNetSD_SR600, self).__init__()
        self.in_dim = in_dim # 4
        self.dim = dim # 320
        self.y_dim = y_dim # 768
        self.context_dim = context_dim # 1024
        self.embed_dim = embed_dim # 1280
        self.out_dim = out_dim # 4
        self.dim_mult = dim_mult # [1, 2, 4, 4]
        ### for temporal attention
        self.num_heads = num_heads # 8
        ### for spatial attention
        self.head_dim = head_dim # 64
        self.num_res_blocks = num_res_blocks # 2
        self.attn_scales = attn_scales # [1.0, 0.5, 0.25]
        self.use_scale_shift_norm = use_scale_shift_norm # True
        self.temporal_attn_times = temporal_attn_times # 1
        self.temporal_attention = temporal_attention # True
        self.use_checkpoint = use_checkpoint # True
        self.use_image_dataset = use_image_dataset # False
        self.use_sim_mask = use_sim_mask # False
        self.inpainting = inpainting # True

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult] # [320, 320, 640, 1280, 1280]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]] # [1280, 1280, 1280, 640, 320]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), # [320,1280]
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        
        # encoder
        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        ####need an initial temporal attention?
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(TemporalTransformer(dim, num_heads, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
            else:
                init_block.append(TemporalAttentionMultiBlock(dim, num_heads, head_dim, rotary_emb=self.rotary_emb, temporal_attn_times=temporal_attn_times, use_image_dataset=use_image_dataset))
        # elif temporal_conv:
        # init_block.append(InitTemporalConvBlock(dim,dropout=dropout,use_image_dataset=use_image_dataset))
        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList([ResBlock(in_dim, embed_dim, dropout, out_channels=out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,)])
                if scale in attn_scales:
                    # block.append(FlashAttentionBlock(out_dim, context_dim, num_heads, head_dim))
                    block.append(
                            SpatialTransformer(
                                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim,
                                disable_self_attn=False, use_linear=True
                            )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(TemporalTransformer(out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb = self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    # block = nn.ModuleList([ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm, 'downsample')])
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim, padding=(2, 1)
                    )
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    # block.append(TemporalConvBlock(out_dim,dropout=dropout,use_image_dataset=use_image_dataset))
                    self.input_blocks.append(downsample)
        
        self.middle_block = nn.ModuleList([
            ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,),
            SpatialTransformer(
                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim,
                disable_self_attn=False, use_linear=True
            )])        
        
        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                 TemporalTransformer( 
                            out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal,
                            multiply_zero=use_image_dataset,
                        )
                )
            else:
                self.middle_block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =  self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))        

        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlock(in_dim + shortcut_dims.pop(), embed_dim, dropout, out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset, )])
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=1024,
                            disable_self_attn=False, use_linear=True
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset
                                    )
                            )
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = UpsampleSR600(out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))
        
        # zero out the last layer params
        nn.init.zeros_(self.out[-1].weight)
    
    def forward(self, 
        x,
        t,
        y,
        x_lr=None,
        fps=None,
        video_mask=None,
        focus_present_mask = None,
        prob_focus_present = 0.,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        mask_last_frame_num = 0  # mask last frame num
        ):

        batch, x_c, x_f, x_h, x_w= x.shape
        device = x.device
        self.batch = batch

        #### image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device)) # [False, False]

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        else:
            time_rel_pos_bias = None
        
        # embeddings
        e = self.time_embed(sinusoidal_embedding(t, self.dim)) #+ self.y_embedding(y)
        context = y #self.context_embedding(y).view(-1, 4, self.context_dim)

        # repeat f times for spatial e and context
        e=e.repeat_interleave(repeats=x_f, dim=0)
        context=context.repeat_interleave(repeats=x_f, dim=0)

        # x = torch.cat([x, temp_x_lr], dim=1)
        # x = x + temp_x_lr
        ## always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        # encoder
        xs = []
        for idx, block in enumerate(self.input_blocks):
            x = self._forward_single(block, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)
            # print(f"encoder shape: {x.shape}")
        
        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,focus_present_mask, video_mask)
        # print(f"mid shape: {x.shape}")

        # decoder
        b_num = 0
        for block in self.output_blocks:
            # print(f"decoder shape: {x.shape}")
            if b_num == 0:
                temp_b, temp_c, _, _ = x.size()
                x[:,:temp_c//2] = x[:,:temp_c//2] * 1.1
                hs_ = xs.pop()
                hs_ = Fourier_filter(hs_, threshold=1, scale=0.6)
                x = torch.cat([x, hs_], dim=1)
            elif b_num == 1:
                temp_b, temp_c, _, _ = x.size()
                x[:,:temp_c//2] = x[:,:temp_c//2] * 1.2
                hs_ = xs.pop()
                hs_ = Fourier_filter(hs_, threshold=1, scale=0.4)
                x = torch.cat([x, hs_], dim=1)
            else:
                x = torch.cat([x, xs.pop()], dim=1)
            # x = torch.cat([x, xs.pop()], dim=1)
            b_num += 1
            x = self._forward_single(block, x, e, context, time_rel_pos_bias,focus_present_mask, video_mask, reference=xs[-1] if len(xs) > 0 else None)
        
        # head
        x = self.out(x) # [32, 4, 32, 32]

        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b = batch)
        return x
    
    def _forward_single(self, module, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=None):
        if isinstance(module, ResidualBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, reference)
        elif isinstance(module, ResBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = x.contiguous()
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, context)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, CrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, MemoryEfficientCrossAttention):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, FeedForward):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, context)
        elif isinstance(module, Upsample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x)
        elif isinstance(module, Downsample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x)
        elif isinstance(module, Resample):
            # module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = module(x, reference)
        elif isinstance(module, TemporalAttentionBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalAttentionMultiBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, InitTemporalConvBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, TemporalConvBlock):
            module = checkpoint_wrapper(module) if self.use_checkpoint else module
            x = rearrange(x, '(b f) c h w -> b c f h w', b = self.batch)
            x = module(x)
            x = rearrange(x, 'b c f h w -> (b f) c h w')
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block,  x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference)
        else:
            x = module(x)
        return x


if __name__ == '__main__':

    # [model] unet
    sd_model = UNetSDSR600(
        in_dim=4,
        dim=320,
        y_dim=1024,
        context_dim=1024,
        out_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_heads=8,
        head_dim=64,
        num_res_blocks=2,
        attn_scales=[1 / 1, 1 / 2, 1 / 4],
        use_scale_shift_norm=True,
        dropout=0.1,
        temporal_attn_times=1,
        temporal_attention = True,
        use_checkpoint=True,
        use_image_dataset=False,
        use_sim_mask = False,
        inpainting=True,
        training=False)
