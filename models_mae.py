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
from mae_st.util import video_vit
from mae_st.util.logging import master_print as print


## This file contains one Class that defines the MAE ViT model

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""
    ## __init__ is the constructor that initializes the model
    ## it builds all the parts: patch embedding, positional encodings, Transformer blocks, and the decoder
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        num_frames=16,
        t_patch_size=4,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        pred_t_dim=8,
        **kwargs,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed: # CLS token are matrix of value 0
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            # 3 types of positional embedding
            # spatial pos embedding: where in the frame does this token belong to?
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            # temporal pos embedding: when is this token? aka which time step
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Cuts the input video into patches of pixel values 
        Pure reshaping and not learn weights
        imgs: (N, 3, T, H, W)
        x: (N, L, u * patch_size**2 * 3), L = (T // u) * (H // p) * (W // p)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.t_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 3))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, u * patch_size**2 * 3)
        imgs: (N, 3, T, H, W)
        Uses self.patch_info, so patchify must have been called first.
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample (per-clip) shuffling.
        High level idea:
        Give every token a random number, then sort by it. That shuffles the tokens into a random queue.
        Keep the front 10% of the queue (156 of 1568) and remove the rest. The encoder sees only those.
        Remember where each kept token came from (ids_keep), so it gets the right position embedding.
        Remember how to undo the shuffle (ids_restore), so the decoder can put everything back in grid order 
        and the loss is computed on the hidden slots.

        x: [N, L, D], sequence

        Input:
        x is the video after it has been embedded into a sequence of tokens
        N is batch size: how many video clips are in the batch
        L is the length of the sequence: how many tokens are in the sequence 
        D is the dimension of the tokens: the embedding dimension

        Output: 

        """
        N, L, D = x.shape  # batch, length, dim

        # Step 1: Calculate len_keep is the number of tokens that survive the masking
        # mask_ratio is the fraction to hide: 0.9 in the recipe (code default 0.75).
        # 1 - mask_ratio is the fraction to keep: 0.1.
        len_keep = int(L * (1 - mask_ratio))

        #Step 2: For each clip, generate random noise

        # One random number per token -> a different mask for every clip
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1) of dimension (N,L)
        #               A     B     C     D     E     F
        #    clip 0:   [0.00, 0.11, 0.29, 0.03, 0.47, 0.06]
        #    clip 1:   [0.77, 0.74, 0.59, 0.89, 0.45, 0.80]

        # argsort: returns the indices that would sort an array (place in ascending order)
        # Intuitively: ids_shuffle is a random queue of the token numbers.
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove

        #          spot: 0  1 | 2  3  4  5
        # clip 0:       [0, 3,| 5, 1, 2, 4]   → A D | F B C E    keep A, D
        # clip 1:       [4, 2,| 1, 0, 5, 3]   → E C | B A F D    keep E, C
        #               keep  | hide

        # Intuitively: ids_restore is each token's place in that queue.
        # ids_restore reverses ids_shuffle
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        #               A  B  C  D  E  F
        #clip 0:       [0, 3, 4, 1, 5, 2]    
        #clip 1:       [3, 2, 1, 5, 0, 4]    

        # Step 3: Keep the first len_keep tokens
        # keep the first subset as unmasked tokens
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Masked tokens are DELETED, not zeroed -> encoder runs on a shorter sequence
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # x_masked  shape [2, 2, 3]
        #
        # clip 0:  [[  0,   1,   2],     ← A
        #           [ 30,  31,  32]]     ← D
        # clip 1:  [[140, 141, 142],     ← E
        #           [120, 121, 122]]     ← C

        # Step 4: Generate the binary mask: 0 is keep, 1 is remove(mask)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        # IMPORTANT to set mask in original token order
        # Mask is back in original token order, so the loss knows which patches were hidden
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # x_masked: the tokens that survived. Only these go to the encoder.
        # mask: a checklist of which tokens were hidden, so the loss knows what to score.
        # ids_restore: the undo key the decoder uses to put every token back in its original place.
        # ids_keep: which original positions were kept, so each surviving token gets the right position embedding.
        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):


        # Step 1: patch_embed
        # patch_embed cuts the clip into non-overlapping 2 × 16 × 16 cubes and 
        # turns each into a 1024-number token, using one Conv3d.
        x = self.patch_embed(x)

        # read and return the size of the embeddings
        N, T, L, C = x.shape  
        # N is the number of video clips in the batch on this GPU.
        # T is the no of embeddings in the time dimension after Conv3D,
        # L is the no of embedding positions in the space dimensions after the Conv3D
        # C is the size of the embedding

        x = x.reshape(N, T * L, C) # flatten into 1 sequence
   

        # Step 2: MAE Random masking; applied to embeddings (not pixels)
        # masking: length -> length * (1 - mask_ratio)
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C)
        
        # Step 3: Append cls token to masked tokens
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # Step 4: Give each unmasked token a position embedding including CLS token, 
        # so it knows where in the video it came from, and add that vector to the token.
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            # gather(input, dim, index) means "along dim, take the elements at the positions listed in index."
            # masking dropped and reordered tokens, so gather drops and reorders the positional embeddings the same way
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed: # CLS token gets its own positional embedding
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        # add the positional embedding
        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        # run unmasked tokens through the transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # pass through a LayerNorm 
        x = self.norm(x)

        if self.cls_embed:
            # remove cls token becuase the decoder does not need it
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        # x is of size [N, 157, 1024]; 157 embeddings of dimension 1024; 157 correspond to the number of unmasked cubes 
        # remove the CLS token to get [N, 156, 1024]

        # x is the encoder's output for the 156 visible (unmasked) tokens ([N, 156, 1024], going to the decoder), 
        # mask marks which of the 1568 tokens were hidden (for the loss), 
        # and ids_restore is the undo key the decoder uses to put tokens back in grid order.
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # High level: Fill the hidden slots with placeholders, put everything back in grid order, 
        # and predict the pixels of every patch.
        
        # Shapes below: recipe values (90% mask -> 156 kept), ViT-L encoder, 512-wide decoder

        # ---- 1. Grid size of the whole video before masking (the decoder rebuilds all of it) ----
        # x is embedding of size [N, 156, 1024]  
        N = x.shape[0]                            # batch size
        T = self.patch_embed.t_grid_size          # 8 time steps
        H = W = self.patch_embed.grid_size        # 14 x 14  -> 1568 slots in total

        # ---- 2. Shrink x to the decoder's width ----
        # decoder_embed is a SINGLE linear layer; embeddings dimension is halved
        x = self.decoder_embed(x)                 # [N, 156, 1024] -> [N, 156, 512]
        C = x.shape[-1]                           # 512

        # ---- 3. Fill the 1412 hidden slots with copies of ONE learned mask token ----
        # repeat(): repeat elements of an array along a specified axis
        # mask_token: one learned [1, 1, 512] vector (created as zeros, re-initialised normal std 0.02 in initialize_weights)
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)  # [1, 1, 512] -> [N, 1412, 512]
        
        # concatenate the unmasked embeddings and the mask tokens (placeholders for the hidden slots)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # [N, 156 + 1412 = 1568, 512], still shuffled
        x_ = x_.view([N, T * H * W, C])           # no-op

        # ---- 4. Un-shuffle: every token back to its original grid slot (index widened to 512 to move whole tokens) ----
        # Inputs: 
        # x_ is [N, 1568, 512]: 156 real tokens + 1412 mask tokens, in shuffled order
        # ids_restore [N, 1568]: the un-shuffle key
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])            # no-op

        # ---- 5. Add the decoder's CLS in front ----
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token                          # [1, 1, 512]
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)   # one per clip: [N, 1, 512]
            x = torch.cat((decoder_cls_tokens, x), dim=1)                       # -> [N, 1569, 512]

        # ---- 6. Add the positional embedding for ALL 1568 slots ----
        if self.sep_pos_embed:
            # spatial (196, tiled x8) + temporal (8, each x196) -> [1, 1568, 512] 
            # both are learned (trunc-normal std 0.02 in initialize_weights)
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                # CLS position in front -> [1, 1569, 512]
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            # unused in pretraining (sep_pos_embed is on)
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        # never actually runs in this repo
        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # ---- 7. 4 blocks (recipe) on all 1569 tokens: mask tokens gather info from the real ones ----
        for blk in self.decoder_blocks:
            x = blk(x)                            # shape unchanged
        x = self.decoder_norm(x)                  # final LayerNorm

        # ---- 8. Decoder turns each token into pixels ----
        # Transformation that happens [N, 1569, 512]  →  [N, 1569, 768]
        # 512 is the decoder's embedding size while 768 is the predicted pixels for 1 patch
        # 1 patch: 16 x 16 pixels x 3 colors = 768 actual pixel values
        x = self.decoder_pred(x) # linear layer

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        # ---- 9. Drop CLS ----
        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]                       # -> [N, 1568, 768]
        else:
            x = x[:, :, :]                        # no-op

        return x                                  # pred: one pixel guess per patch, in grid order

    def forward_loss(self, imgs, pred, mask):
        # Score how well the decoder predicted the hidden patches (pixels)
        # Use the MSE loss between predicted and real pixels on hidden patches ONLY
        """
        imgs: the real 5-D video,  [N, 3, T, H, W] 
        pred: the predicted pixels for every patch (not yet reshaped into frames), [N, t*h*w, u*p*p*3]
        mask: which patches were hidden, [N, t*h*w], 0 is keep, 1 is remove,
        """
        # index_select returns a new tensor that indexes an input tensor
        # along a specific dimension using integer indices
        _imgs = torch.index_select( 
            imgs, # the real 5-D video
            2,  # dim 2 = frames
            # linspace returns a one-dimensional tensor of steps equally spaced points
            # between start and end inclusive
            # Example: linspace(0, 15, 8) → 8 frame numbers
            torch.linspace( 
                0,
                imgs.shape[2] - 1,
                self.pred_t_dim,
            )
            .long()  # round down to frame numbers: 0, 2, 4, 6, 8, 10, 12, 15
            .to(imgs.device),
        )
        # _imgs is the real input video cut down to the 8 frames the loss scores against

        # patchify cuts the real video into pixel patches, [N, 3, 8, 224, 224] → [N, 1568, 768],
        # so it lines up patch-for-patch with the decoder's prediction.
        # pixel patch or target in this case is the REAL pixel values 
        target = self.patchify(_imgs)

        if self.norm_pix_loss: # normalize each patch (on in the recipe)
            mean = target.mean(dim=-1, keepdim=True)  # each patch's mean over its 768 values: [N, 1568, 1]
            var = target.var(dim=-1, keepdim=True)    # each patch's variance: [N, 1568, 1]
            target = (target - mean) / (var + 1.0e-6) ** 0.5  # each patch -> mean 0, variance 1
            
        # Calculate the MSE Loss
        loss = (pred - target) ** 2 # error squaring operation: [N, 1568, 768]
        loss = loss.mean(dim=-1)  # calculate mean loss per patch: [N, 1568]
        mask = mask.view(loss.shape)  # no-op: already [N, 1568]

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75): 
        # encoder → decoder → loss
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, u*p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
