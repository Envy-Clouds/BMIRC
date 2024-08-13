from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .models_utils import (Block, trunc_normal_, PositionalEncoding)


class LinearOutputAdapter(nn.Module):
    """
    Linear output adapter.

    :param num_classes: Number of classes
    :param dim_tokens_enc: Dimension of tokens from the encoder
    :param use_mean_pooling: When set to True, uses mean pooling before linear classification head.
        Otherwise, use last token (usually the global token)
    :param norm_layer: Normalization layer
    :param init_scale: Initialization scale for linear classification head
    """

    def __init__(self,
                 num_classes: int,
                 dim_tokens_enc: Optional[int] = None,
                 use_mean_pooling: bool = True,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 init_scale: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dim_tokens_enc = dim_tokens_enc
        self.use_mean_pooling = use_mean_pooling
        self.norm_layer = norm_layer
        self.init_scale = init_scale

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc: int = 768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        self.dim_tokens_enc = dim_tokens_enc

        self.norm = self.norm_layer(self.dim_tokens_enc)
        self.head = nn.Linear(dim_tokens_enc, self.num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        self.head.weight.data.mul_(self.init_scale)
        self.head.bias.data.mul_(self.init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.init(dim_tokens_enc=self.dim_tokens_enc)

    def forward(self,
                encoder_tokens: torch.Tensor,
                **kwargs):

        if self.use_mean_pooling:
            x = encoder_tokens.mean(1)
        else:
            x = encoder_tokens[:, 0]

        x = self.head(self.norm(x))
        return x


class MyOutputAdapter(nn.Module):
    """
    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens_enc: Dimension of tokens coming from encoder. Can be set using init method.
    :param dim_tokens: Dimension of decoder tokens
    :param depth: Number of additional (full self-attention) transformer layers after initial cross attention and MLP
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    :param mlp_ratio: MLP hidden dim ratio
    :param num_heads: Number of attention heads
    :param qkv_bias: Set to True to enable bias
    :param drop_rate: Probability of dropping attention layer outputs
    :param attn_drop_rate: Probability of dropping attention matrix elements
    :param drop_path_rate: DropPath drop rate
    :param norm_layer: Type of normalization layer
    """

    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int, int]],
                 dim_tokens_enc: Optional[int] = 192,
                 dim_tokens: int = 96,
                 depth: int = 4,
                 learnable_pos_emb: int = True,
                 image_size: Union[int, Tuple[int]] = None,
                 mlp_ratio: int = 4.0,
                 num_heads: int = 3,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
                 ):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = patch_size_full
        self.dim_tokens_enc = dim_tokens_enc
        self.dim_tokens = dim_tokens
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = image_size

        # Actual patch height and width, taking into account stride of input
        # self.P_H = 1
        self.P_W = self.patch_size_full

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.dim_tokens))

        self.mask_token1 = nn.Parameter(torch.zeros(1, 1, self.dim_tokens), requires_grad=False)

        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        w_posemb = self.image_size // self.P_W
        if not self.learnable_pos_emb:
            self.pos_emb = PositionalEncoding(w_posemb, self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=False)
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        self.decoder_norm = norm_layer(self.dim_tokens)

        # Optional full self-attention transformer layers
        if depth > 0:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
            self.decoder_transformer = nn.Sequential(*[
                Block(dim=self.dim_tokens, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                      attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth)
            ])
        else:
            self.decoder_transformer = nn.Identity()

        self.dim_patch = self.num_channels * self.P_W
        self.out_proj = nn.Linear(self.dim_tokens, self.dim_patch)

        self.init(dim_tokens_enc=dim_tokens_enc)

        proj_layers = [
            torch.nn.Linear(self.dim_tokens_enc, self.dim_tokens)
            for _ in range(3)
        ]
        self.proj_layers = torch.nn.ModuleList(proj_layers)

        inter_norm_layers = [
            norm_layer(self.dim_tokens_enc)
            for _ in range(3)
        ]
        self.inter_norm_layers = torch.nn.ModuleList(inter_norm_layers)

        gate_layers = [
            torch.nn.Linear(self.dim_tokens*2, 1)
            for _ in range(3)
        ]
        self.gate_layers = torch.nn.ModuleList(gate_layers)

        self.act_f = torch.nn.Sigmoid()


    def init(self, dim_tokens_enc: int = 192):
        '''
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        '''
        self.dim_tokens_enc = dim_tokens_enc

        # Projection of encoder tokens to the patch dimension
        self.proj_context = nn.Linear(self.dim_tokens_enc, self.dim_tokens)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb', 'mask_token'}

    def get_context(self, context_tokens, input_info, ids_restore, flag=True):
        B = context_tokens.shape[0]

        # Add mask tokens
        if flag:
            mask_tokens = repeat(self.mask_token, '() () d -> b n d', b=B,
                                 n=input_info['num_task_tokens'] - context_tokens.shape[1])
        else:
            mask_tokens = repeat(self.mask_token1, '() () d -> b n d', b=B,
                                 n=input_info['num_task_tokens'] - context_tokens.shape[1])

        context_with_mask = torch.cat([context_tokens, mask_tokens], dim=1)

        # Unshuffle context_with_mask
        context_with_mask = torch.gather(context_with_mask, dim=1,
                                         index=ids_restore.unsqueeze(-1).repeat(1, 1, context_with_mask.shape[2]))

        if flag:
            pos_emb = rearrange(self.pos_emb, 'b d nw -> b nw d')
            context_with_mask += pos_emb

        return context_with_mask

    def forward(self,
                encoder_tokens: torch.Tensor,
                input_info: dict,
                ids_restore: torch.Tensor,
                inter_input: list
                ):
        """
        Forward pass taking output tokens from encoder and optionally a subset of them corresponding
        to this output adapter's task (needs an additional mask describing position of these tokens in the queries).

        :param encoder_tokens: Output of encoder
        :param input_info: Dictionary with information about the input modalities
        :param ids_keep: IDs of unmasked tokens (tokens given to the encoder)
        :param ids_restore: IDs to unshuffle tokens
        """
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        W = input_info['image_size']
        N_W = W // self.P_W

        decoder_tokens = self.proj_context(encoder_tokens)

        x = self.get_context(decoder_tokens, input_info, ids_restore)

        # Optional transformer layers if depth > 0
        for i, layer in enumerate(self.decoder_transformer):
            if i == 0:
                x = layer(x)
            else:
                inter = self.proj_layers[-i](self.inter_norm_layers[-i](inter_input[-i]))
                inter = self.get_context(inter, input_info, ids_restore, flag=False)
                inter_weight = self.act_f(self.gate_layers[-i](torch.cat((inter, x), dim=2)))
                x = layer(inter_weight*inter+(1-inter_weight)*x)

        x = self.decoder_norm(x)

        # Project each token to (C * P_H * P_W)
        x = self.out_proj(x)

        # Reshape sequence of patches into image
        x = rearrange(
            x, 'b nw (c pw) -> b c (nw pw)',
            nw=N_W, pw=self.P_W, c=self.num_channels
        )

        return x
