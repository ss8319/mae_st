# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
from mae_st.util.logging import master_print as print
from mae_st.util.video_vit import Attention, Block, PatchEmbed


## This file contains the FINE-TUNING model: one Class (the MAE-ST encoder + a classifier head)
## and 3 builder functions at the bottom that only set the model size (base / large / huge).
## vs models_mae.py: no masking, no decoder. Parameter names match the MAE encoder's,
## so load_state_dict(strict=False) in main_finetune.py copies the encoder and skips the decoder.

class VisionTransformer(nn.Module):
    """Vision Transformer with support for global average pooling"""
    ## __init__ builds the parts: patch embedding, CLS + positional embeddings,
    ## Transformer blocks, and the classifier (dropout + linear head)
    def __init__(
        self,
        num_frames,                 # frames per input clip (recipe: 16)
        t_patch_size,               # frames per video cube in time (recipe: 2)
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=400,            # Kinetics-400
        embed_dim=768,              # overridden by the builder (ViT-L: 1024)
        depth=12,                   # overridden by the builder (ViT-L: 24)
        num_heads=12,               # overridden by the builder (ViT-L: 16)
        mlp_ratio=4.0,
        no_qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,              # unused
        attn_drop_rate=0.0,         # unused
        drop_path_rate=0.0,         # stochastic depth max rate (recipe: 0.2)
        norm_layer=nn.LayerNorm,
        dropout=0.5,                # dropout before the head (recipe: 0.3)
        sep_pos_embed=False,        # main_finetune.py sets this to True
        cls_embed=False,            # main_finetune.py sets this to True
        **kwargs,
    ):
        super().__init__()
        print(locals())

        self.sep_pos_embed = sep_pos_embed
        # --------------------------------------------------------------------------
        # MAE encoder specifics (same as the encoder half of models_mae.py)
        # Conv3d: video -> one token per 2 x 16 x 16 cube
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size    # token grid (T, H, W), recipe: (8, 14, 14)
        self.input_size = input_size
        self.cls_embed = cls_embed

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            # spatial (where in the frame) + temporal (which time step) + CLS
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            # unused in the recipe (sep_pos_embed is on)
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim), requires_grad=True
            )  # fixed or not?

        # stochastic depth: drop-path rate rises linearly from 0 (first block) to drop_path_rate (last block)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    attn_func=partial(
                        Attention,
                        input_size=self.patch_embed.input_size,
                    ),
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # NEW vs pretraining: the classifier
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embed_dim, num_classes)   # 1024 -> 400 class scores

        # re-initialised to std 2e-5 in main_finetune.py after loading the checkpoint
        torch.nn.init.normal_(self.head.weight, std=0.02)

    ## parameters excluded from weight decay (CLS + positional embeddings)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_class",
        }

    ## forward: video clip -> class scores
    ## Shapes below: recipe values (16 frames, 2 x 16 x 16 cubes), ViT-L
    def forward(self, x):
        # Step 1: patch_embed; cut the clip into cubes, one 1024-number token each
        x = self.patch_embed(x)     # [N, 3, 16, 224, 224] -> [N, 8, 196, 1024]
        N, T, L, C = x.shape  # T: temporal; L: spatial

        x = x.view([N, T * L, C])   # flatten into 1 sequence: [N, 1568, 1024]

        # Step 2: Prepend CLS token (NO masking here: all 1568 tokens are kept)
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)   # [N, 1569, 1024]

        # Step 3: Build + ADD positional embeddings for ALL tokens (no gather, nothing was dropped)
        if self.sep_pos_embed:
            # spatial (196, tiled x8) + temporal (8, each x196) -> [1, 1568, 1024]
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                # CLS position in front -> [1, 1569, 1024]
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            pos_embed = self.pos_embed[:, :, :]
        x = x + pos_embed           # broadcasts [1, 1569, 1024] over the batch

        # never actually runs in this repo (Attention has no requires_t_shape)
        requires_t_shape = (
            len(self.blocks) > 0  # support empty decoder
            and hasattr(self.blocks[0].attn, "requires_t_shape")
            and self.blocks[0].attn.requires_t_shape
        )
        if requires_t_shape:
            x = x.view([N, T, L, C])

        # Step 4: 24 Transformer blocks on all 1569 tokens (shape unchanged)
        for blk in self.blocks:
            x = blk(x)

        if requires_t_shape:
            x = x.view([N, T * L, C])

        # Step 5: Classifier
        # average the 1568 patch tokens (CLS is dropped, NOT used for classification)
        # NOTE: x[:, 1:] assumes CLS exists; with cls_embed=False it would drop a real patch token
        x = x[:, 1:, :].mean(dim=1)  # global pool: [N, 1569, 1024] -> [N, 1024]
        x = self.norm(x)
        # x = self.fc_norm(x)
        x = self.dropout(x)
        x = self.head(x)            # [N, 1024] -> [N, 400]

        return x                    # class scores (logits)


## Builders: same class, different size. main_finetune.py picks one via --model (recipe: vit_large_patch16)

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# NOTE: name says patch14 but patch_size is 16
def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
