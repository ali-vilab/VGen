import os
import torch
import logging
import open_clip
import numpy as np
import torch.nn as nn
import torchvision.transforms as T

from utils.registry_class import EMBEDDER


@EMBEDDER.register_class()
class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, pretrained, arch="ViT-H-14", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


@EMBEDDER.register_class()
class FrozenOpenCLIPVisualEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, pretrained, vit_resolution=(224, 224), arch="ViT-H-14", device="cuda", max_length=77,
                 freeze=True, layer="last"):
        super().__init__()
        assert layer in self.LAYERS
        model, _, preprocess = open_clip.create_model_and_transforms(
                arch, device=torch.device('cpu'), pretrained=pretrained)
        # Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        del model.transformer 
        self.model = model
        data_white = np.ones((vit_resolution[0], vit_resolution[1], 3), dtype=np.uint8)*255
        self.white_image = preprocess(T.ToPILImage()(data_white)).unsqueeze(0)

        self.device = device
        self.max_length = max_length # 77
        if freeze:
            self.freeze()
        self.layer = layer # 'penultimate'
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self): # model.encode_image(torch.randn(2,3,224,224))
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, image):
        # tokens = open_clip.tokenize(text)
        z = self.model.encode_image(image.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)

        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)



@EMBEDDER.register_class()
class FrozenOpenCLIPTtxtVisualEmbedder(nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = [
        #"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, pretrained, arch="ViT-H-14", device="cuda", max_length=77,
                 freeze=True, layer="last", **kwargs):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    # def forward(self, text):
    #     tokens = open_clip.tokenize(text)
    #     z = self.encode_with_transformer(tokens.to(self.device))
    #     return z
    
    def forward(self, image=None, text=None):
        # xi = self.encode_image(image) if image is not None else None
        xi = self.model.encode_image(image.to(self.device)) if image is not None else None
        # tokens = open_clip.tokenize(text, truncate=True)
        tokens = open_clip.tokenize(text)
        xt, x = self.encode_with_transformer(tokens.to(self.device))
        return xi, xt, x

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        xt = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
        return xt, x
    
    # def encode_with_transformer(self, text):
    #     x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
    #     x = x + self.model.positional_embedding
    #     x = x.permute(1, 0, 2)  # NLD -> LND
    #     x = self.model.transformer(x)
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.model.ln_final(x)
    #     xt = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.model.text_projection
    #     # text embedding, token embedding
    #     return xt, x

    def encode_image(self, image):
        return self.model.visual(image)

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        
        return self(text)

