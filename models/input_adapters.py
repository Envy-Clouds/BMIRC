from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .models_utils import trunc_normal_, PositionalEncoding


class PatchedInputAdapter(nn.Module):
    """Adapter for spatial inputs, like images or feature maps.
    Creates tokens from patches over the image.

    :param num_channels: Number of input channels of the image/feature map
    :param stride_level: Stride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param dim_tokens: Dimension of output tokens. Can be set using init method.
    :param sincos_pos_emb: Set to True (default) to use fixed 2D sin-cos positional embeddings
    :param learnable_pos_emb: Set to True to learn positional embeddings instead
    :param image_size: Default image size. Used to initialize size of positional embeddings.
    """

    def __init__(self,
                 num_channels: int,
                 stride_level: int,
                 patch_size_full: Union[int, Tuple[int, int]],
                 dim_tokens: Optional[int] = None,
                 sincos_pos_emb: bool = False,
                 learnable_pos_emb: bool = True,
                 image_size: Union[int, Tuple[int]] = None):

        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size_full = patch_size_full
        self.dim_tokens = dim_tokens
        self.sincos_pos_emb = sincos_pos_emb
        self.learnable_pos_emb = learnable_pos_emb
        self.image_size = image_size
        self.num_patches = self.image_size // patch_size_full

        # Actual patch height and width, taking into account stride of input
        self.P_W = self.patch_size_full

        if self.dim_tokens is not None:
            self.init(dim_tokens=dim_tokens)

    def init(self, dim_tokens: int = 768):
        """
        Initialize parts of encoder that are dependent on dimension of tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens: Dimension of tokens
        """
        self.dim_tokens = dim_tokens

        # Task embedding identifying from which task a given token comes from
        # Fixed-size positional embeddings. Can be interpolated to different input sizes
        # h_posemb = 1
        w_posemb = self.image_size // self.P_W
        if self.sincos_pos_emb:
            self.pos_emb = PositionalEncoding(w_posemb, self.dim_tokens)
            self.pos_emb = nn.Parameter(self.pos_emb, requires_grad=self.learnable_pos_emb)
        else:
            # self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, h_posemb, w_posemb))
            self.pos_emb = nn.Parameter(torch.zeros(1, self.dim_tokens, w_posemb))
            trunc_normal_(self.pos_emb, std=0.02)

        self.proj = nn.Conv1d(
            in_channels=self.num_channels, out_channels=self.dim_tokens,
            kernel_size=self.P_W, stride=self.P_W
        )


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_emb'}

    def forward(self, x):
        """
        Forward pass through input adapter, transforming image to sequence of tokens.
        Adds task and positional encodings.

        :param x: Input image tensor
        """
        # B, C, H, W = x.shape
        B, C, W = x.shape
        assert self.dim_tokens is not None, 'Need to call init(dim_tokens) function first'

        assert W % self.P_W == 0, f'Image sizes {W} must be divisible by patch sizes {self.P_W}'
        N_W = W // self.P_W  # Number of patches in width

        x_patch = rearrange(self.proj(x), 'b d nw -> b nw d')

        # Create positional embedding
        x_pos_emb = F.interpolate(self.pos_emb, size=N_W, mode='linear', align_corners=False)
        x_pos_emb = rearrange(x_pos_emb, 'b d nw -> b nw d')

        # Add patches and positional embeddings
        x = x_patch + x_pos_emb

        return x
