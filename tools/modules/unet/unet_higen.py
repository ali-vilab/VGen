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
import xformers
import xformers.ops
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from fairscale.nn.checkpoint import checkpoint_wrapper

from .util import *
from .mha_flash import FlashAttentionBlock
from utils.registry_class import MODEL


USE_TEMPORAL_TRANSFORMER = True

class ResBlockWoImg(ResBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
    ):
        super().__init__(channels, emb_channels, dropout, out_channels, use_conv, use_scale_shift_norm, dims, up, down, use_temporal_conv, use_image_dataset)
        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2WoImg(self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset)
            # self.temopral_conv_2 = TemporalConvBlock(self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset)


class TemporalConvBlock_v2WoImg(TemporalConvBlock_v2):
    def __init__(self, in_dim, out_dim=None, dropout=0.0, use_image_dataset=False):
        super(TemporalConvBlock_v2WoImg, self).__init__(in_dim, out_dim, dropout, use_image_dataset)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if x.size(2) == 1:
            x = identity + 0.0 * x
        else:
            x = identity + x
        return x


class TemporalTransformerWoImg(TemporalTransformer):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True, only_self_att=True, multiply_zero=False):
        super().__init__(in_channels, n_heads, d_head,
                 depth, dropout, context_dim,
                 disable_self_attn, use_linear,
                 use_checkpoint, only_self_att, multiply_zero)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        x_in = x
        x = self.norm(x)

        if not self.use_linear:
            x = rearrange(x, 'b c f h w -> (b h w) c f').contiguous()
            x = self.proj_in(x)
        # [16384, 16, 320]
        if self.use_linear:
            x = rearrange(x, '(b f) c h w -> b (h w) f c', f=self.frames).contiguous()
            x = self.proj_in(x)

        if self.only_self_att:
            x = rearrange(x, 'bhw c f -> bhw f c').contiguous()
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            x = rearrange(x, '(b hw) f c -> b hw f c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) c f -> b hw f c', b=b).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # context[i] = repeat(context[i], '(b f) l con -> b (f r) l con', r=(h*w)//self.frames, f=self.frames).contiguous()
                context[i] = rearrange(context[i], '(b f) l con -> b f l con', f=self.frames).contiguous()
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = repeat(context[i][j], 'f l con -> (f r) l con', r=(h*w)//self.frames, f=self.frames).contiguous()
                    x[j] = block(x[j], context=context_i_j)
        
        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) f c -> b f c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            # x = rearrange(x, 'bhw f c -> bhw c f').contiguous()
            x = rearrange(x, 'b hw f c -> (b hw) c f').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c f -> b c f h w', b=b, h=h, w=w).contiguous()
        
        if x.size(2) == 1:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class TextContextCrossTransformerMultiLayer(nn.Module):
    def __init__(self, y_dim, depth, embed_dim, context_dim, num_tokens):
        super(TextContextCrossTransformerMultiLayer, self).__init__()
        self.context_transformer = nn.ModuleList(
            [BasicTransformerBlock(embed_dim, n_heads=8, d_head=embed_dim//8, dropout=0.0, context_dim=embed_dim,
                                   disable_self_attn=True, checkpoint=True)
                for d in range(depth)]
        )
        self.input_mapping = nn.Linear(y_dim, embed_dim)
        self.output_mapping = nn.Linear(embed_dim, context_dim)
        scale = embed_dim ** -0.5
        self.tokens = nn.Parameter(scale * torch.randn(1, num_tokens, embed_dim))

    def forward(self, x):
        x = self.input_mapping(x)
        out = self.tokens.repeat(x.size(0), 1, 1)
        for transformer in self.context_transformer:
            out = transformer(out, context=x)
        return self.output_mapping(out)


@MODEL.register_class()
class UNetSD_HiGen(nn.Module):
    def __init__(self,
            config=None,
            in_dim=4,
            dim=512,
            y_dim=512,
            context_dim=512,
            hist_dim = 156,
            dim_condition=4,
            out_dim=6,
            num_tokens=4,
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
            training=True,
            inpainting=True,
            use_fps_condition=False,
            p_all_zero=0.1,
            p_all_keep=0.1,
            zero_y=None,
            adapter_transformer_layers=1,
            context_embedding_depth=4,
            **kwargs):
        super(UNetSD_HiGen, self).__init__()
        
        embed_dim = dim * 4
        num_heads=num_heads if num_heads else dim//32
        self.zero_y = zero_y
        self.in_dim = in_dim
        self.dim = dim
        self.y_dim = y_dim
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.hist_dim = hist_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        ### for temporal attention
        self.num_heads = num_heads
        ### for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_sim_mask = use_sim_mask
        self.training=training
        self.inpainting = inpainting
        self.p_all_zero = p_all_zero
        self.p_all_keep = p_all_keep
        self.use_fps_condition = use_fps_condition

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(dim, embed_dim), # [320,1280]
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim))
        
        if self.use_fps_condition:
            self.fps_embedding = nn.Sequential(
                nn.Linear(dim, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim))
            nn.init.zeros_(self.fps_embedding[-1].weight)
            nn.init.zeros_(self.fps_embedding[-1].bias)
        
        self.context_embedding = TextContextCrossTransformerMultiLayer(y_dim, context_embedding_depth, embed_dim, context_dim, num_tokens=self.num_tokens)

        self.asim_embedding = nn.Sequential(
                    nn.Linear(32, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim))
        nn.init.zeros_(self.asim_embedding[-1].weight)
        nn.init.zeros_(self.asim_embedding[-1].bias)

        self.msim_embedding = nn.Sequential(
                    nn.Linear(dim, embed_dim),
                    nn.SiLU(),
                    nn.Linear(embed_dim, embed_dim))
        nn.init.zeros_(self.msim_embedding[-1].weight)
        nn.init.zeros_(self.msim_embedding[-1].bias)

        self.img_embedding = nn.Conv2d(self.in_dim, dim, 3, padding=1)
        nn.init.zeros_(self.img_embedding.weight)
        nn.init.zeros_(self.img_embedding.bias)

        if temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            self.rotary_emb = RotaryEmbedding(min(32, head_dim))
            self.time_rel_pos_bias = RelativePositionBias(heads = num_heads, max_distance = 32)

        # encoder
        self.input_blocks = nn.ModuleList()
        init_block = nn.ModuleList([nn.Conv2d(self.in_dim, dim, 3, padding=1)])
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(TemporalTransformerWoImg(dim, num_heads, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
            else:
                init_block.append(TemporalAttentionMultiBlock(dim, num_heads, head_dim, rotary_emb=self.rotary_emb, temporal_attn_times=temporal_attn_times, use_image_dataset=use_image_dataset))

        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.ModuleList([ResBlockWoImg(in_dim, embed_dim, dropout, out_channels=out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset)])
                if scale in attn_scales:
                    block.append(
                            SpatialTransformer(
                                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim,
                                disable_self_attn=False, use_linear=True
                            )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(TemporalTransformerWoImg(out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset))
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb = self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(
                        out_dim, True, dims=2, out_channels=out_dim
                    )
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)
        
        self.middle_block = nn.ModuleList([
            ResBlockWoImg(out_dim, embed_dim, dropout, use_scale_shift_norm=False, use_image_dataset=use_image_dataset,),
            SpatialTransformer(
                out_dim, out_dim // head_dim, head_dim, depth=1, context_dim=self.context_dim,
                disable_self_attn=False, use_linear=True
            )])        
        
        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                 TemporalTransformerWoImg( 
                            out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal,
                            multiply_zero=use_image_dataset,
                        )
                )
            else:
                self.middle_block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =  self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))        

        self.middle_block.append(ResBlockWoImg(out_dim, embed_dim, dropout, use_scale_shift_norm=False))

        # decoder
        self.output_blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.ModuleList([ResBlockWoImg(in_dim + shortcut_dims.pop(), embed_dim, dropout, out_dim, use_scale_shift_norm=False, use_image_dataset=use_image_dataset, )])
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
                                TemporalTransformerWoImg(
                                    out_dim, out_dim // head_dim, head_dim, depth=transformer_depth, context_dim=context_dim,
                                    disable_self_attn=disabled_sa, use_linear=use_linear_in_temporal, multiply_zero=use_image_dataset
                                    )
                            )
                        else:
                            block.append(TemporalAttentionMultiBlock(out_dim, num_heads, head_dim, rotary_emb =self.rotary_emb, use_image_dataset=use_image_dataset, use_sim_mask=use_sim_mask, temporal_attn_times=temporal_attn_times))
                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(out_dim, True, dims=2.0, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, self.out_dim, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)

    def get_motion_embedding(self, batch, f, motion_cond):
        if f > 1:
            if motion_cond.size(1) != f:
                motion_embedding = sinusoidal_embedding(motion_cond.flatten(0, 1), self.dim).view(batch, f-1, self.dim)
                motion_embedding = torch.nn.functional.interpolate(motion_embedding.transpose(1, 2), size=(f), mode='linear').transpose(1, 2)
            else:
                motion_embedding = sinusoidal_embedding(motion_cond.flatten(0, 1), self.dim).view(batch, f, self.dim)
            return self.msim_embedding(motion_embedding).flatten(0, 1)
        else:
            return self.msim_embedding(sinusoidal_embedding(motion_cond, self.dim))

    def get_appearance_embedding(self, batch, f, appearance_cond):
        return self.asim_embedding(appearance_cond).flatten(0, 1)

    def forward(self, 
        x,
        t,
        y = None,
        fps = None,
        masked = None,
        video_mask = None,
        spat_prior = None,
        motion_cond = None,
        appearance_cond = None,
        focus_present_mask = None,
        prob_focus_present = 0.,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
        mask_last_frame_num = 0,  # mask last frame num
        **kwargs):
        
        assert self.inpainting or masked is None, 'inpainting is not supported'

        batch, c, f, h, w= x.shape
        device = x.device
        self.batch = batch

        #### image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device = device))

        if self.temporal_attention and not USE_TEMPORAL_TRANSFORMER:
            time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device = x.device)
        else:
            time_rel_pos_bias = None
        
        # [Embeddings]
        if self.use_fps_condition and fps is not None:
            embeddings = self.time_embed(sinusoidal_embedding(t, self.dim)) + self.fps_embedding(sinusoidal_embedding(fps, self.dim))
        else:
            embeddings = self.time_embed(sinusoidal_embedding(t, self.dim))
        embeddings = embeddings.repeat_interleave(repeats=f, dim=0)
        embeddings = embeddings + self.get_motion_embedding(batch, f, motion_cond)
        embeddings = embeddings + self.get_appearance_embedding(batch, f, appearance_cond)

        # [Context]
        context = self.context_embedding(y)
        context = context.repeat_interleave(repeats=f, dim=0)
        
        x = rearrange(x, 'b c f h w -> (b f) c h w')
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, embeddings, context, spat_prior, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)
        
        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, embeddings, context, spat_prior, time_rel_pos_bias,focus_present_mask, video_mask)
        
        # decoder
        for block in self.output_blocks:
            x = torch.cat([x, xs.pop()], dim=1)
            x = self._forward_single(block, x, embeddings, context, spat_prior, time_rel_pos_bias,focus_present_mask, video_mask, reference=xs[-1] if len(xs) > 0 else None)
        
        # head
        x = self.out(x) # [32, 4, 32, 32]

        # reshape back to (b c f h w)
        x = rearrange(x, '(b f) c h w -> b c f h w', b = batch)
        return x
        
    
    def _forward_single(self, module, x, e, context, spat_prior, time_rel_pos_bias, focus_present_mask, video_mask, reference=None):
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
        elif isinstance(module, TemporalTransformer_attemask):
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
        elif isinstance(module, nn.Conv2d) and x.size(1) == self.in_dim:
            x = module(x)
            f = x.size(0) // self.batch
            x = x + self.img_embedding(spat_prior).repeat_interleave(repeats=f, dim=0)
        elif isinstance(module, nn.ModuleList):
            for block in module:
                x = self._forward_single(block,  x, e, context, spat_prior, time_rel_pos_bias, focus_present_mask, video_mask, reference)
        else:
            x = module(x)
        return x
