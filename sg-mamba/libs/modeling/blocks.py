from functools import partial
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn.functional import linear, softmax, dropout
from .weight_init import trunc_normal_
from mamba_ssm.modules.mamba_simple import Mamba as ViM
from mamba_ssm.modules.mamba_new import Mamba as DBM
from mamba_ssm.modules.mamba_v2m import Mamba as V2M
from mamba_ssm.modules.mamba_mulscale import Mamba as MDBM
from mamba_ssm.modules.mamba_vision_mulscale import Mamba as MViM
# from mamba_ssm.modules.mamba_v2m_mulscale import Mamba as MV2M # Deleted
from mamba_ssm.modules.cross_mamba import Mamba as CrossMamba
Tensor = torch.Tensor
from typing import Optional, Tuple
import warnings


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T//self.stride,
                mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


# attention / transformers
class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the input embedding
        n_head,          # number of heads in multi-head self-attention
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0   # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # calculate query, key, values for all heads in batch
        # (B, nh * hs, T)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ (v * mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class MaskedMHCCA(nn.Module):
    """
    Multi Head Conv Cross Attention with mask (PostNorm)

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output/query features
        kv_dim,          # dimension of the key/value features
        n_head,          # number of heads in multi-head self-attention
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        assert kv_dim % n_head == 0
        self.n_embd = n_embd
        self.kv_dim = kv_dim
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.kv_dim, self.kv_dim, kernel_size,
            stride=stride, padding=padding, groups=self.kv_dim, bias=False
        )
        self.key_norm = LayerNorm(self.kv_dim)
        self.value_conv = MaskedConv1D(
            self.kv_dim, self.kv_dim, kernel_size,
            stride=stride, padding=padding, groups=self.kv_dim, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.kv_dim)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.key = nn.Conv1d(self.kv_dim, self.n_embd, 1)
        self.value = nn.Conv1d(self.kv_dim, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, query, key, value, q_mask, kv_mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = query.size()
        # B, D, T = key.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(query, q_mask)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(key, kv_mask)
        v, _ = self.value_conv(value, kv_mask)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask

class MaskedPoolingMHCA(nn.Module):
    """
    MViT-style Multi Head Conv Attention with mask (pooling attention)
    Reference: http://arxiv.org/abs/2104.11227
    """
    def __init__(
        self,
        n_embd,           # dimension of the output features
        n_head,           # number of heads in multi-head self-attention
        n_qx_stride=1,    # dowsampling stride for query and input
        n_kv_stride=1,    # downsampling stride for key and value
        attn_pdrop=0.0,   # dropout rate for the attention map
        proj_pdrop=0.0,   # dropout rate for projection op
        pool_qx_size=2,   # pooling size of the query and input
        pool_kv_size=2,   # pooling size of the key and value
        pool_method='avg' # pooling method (avg or max)
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # pooling
        if pool_method.lower() == 'avg':
            self.pool_qx = AvgPooler(pool_qx_size, stride=pool_qx_size)
            self.pool_kv = AvgPooler(pool_kv_size, stride=pool_kv_size)
        elif pool_method.lower() == 'max':
            self.pool_qx = MaxPooler(pool_qx_size, stride=pool_qx_size)
            self.pool_kv = MaxPooler(pool_kv_size, stride=pool_kv_size)
        elif pool_method.lower() == 'conv':
            kernel_size = pool_qx_size + 1
            padding = kernel_size // 2
            self.pool_qx = MaskedConv1D(self.n_embd, self.n_embd, kernel_size, stride=pool_qx_size, padding=padding, groups=self.n_embd, bias=False)
            self.pool_kv = MaskedConv1D(self.n_embd, self.n_embd, kernel_size, stride=pool_kv_size, padding=padding, groups=self.n_embd, bias=False)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # pooling
        q, qx_mask = self.pool_qx(q, qx_mask)
        k, kv_mask = self.pool_kv(k, kv_mask)
        v, _       = self.pool_kv(v, kv_mask)
        pooled_x,_ = self.pool_qx(x, mask)
        
        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype) + pooled_x
        return out, qx_mask
    

class MaskedPoolingMHCAv2(MaskedPoolingMHCA):
    '''
    Improved MViT-style Pooling Multi Head Conv Attention with mask 
    Reference: https://arxiv.org/pdf/2112.01526v2
    '''
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # self.num_patches = 
        # self.rel_pe = nn.Parameter(
        #         torch.zeros(1, self.n_head, self., self.n_channels)) # TODO: check the dimension
        # trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)
    
    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # pooling
        q, qx_mask = self.pool_qx(q, qx_mask)
        k, kv_mask = self.pool_kv(k, kv_mask)
        v, _       = self.pool_kv(v, kv_mask)
        pooled_x,_ = self.pool_qx(x, mask)
        
        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # print('k ', k.shape, 'pe ', self.rel_pe.shape)
        # breakpoint()
        # #　> New: add relative positional encoding to pooled_k
        # k = k + self.rel_pe

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))

        # > New: add a skip connection
        out = out + q

        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype) + pooled_x
        return out, qx_mask

class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        window_size,     # size of the local attention window
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
        use_rel_pe=False # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap  = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        # layernorm
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False       # if to add rel position encoding to attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # specify the attention module
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe  # only valid for local attention
            )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask

class PostNormCrossTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kv_embd,               # dimension of the key and value features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False       # if to add rel position encoding to attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # specify the attention module
        if mha_win_size > 1:
            # self.attn = LocalMaskedMHCA(
            #     n_embd,
            #     n_head,
            #     window_size=mha_win_size,
            #     n_qx_stride=n_ds_strides[0],
            #     n_kv_stride=n_ds_strides[1],
            #     attn_pdrop=attn_pdrop,
            #     proj_pdrop=proj_pdrop,
            #     use_rel_pe=use_rel_pe  # only valid for local attention
            # )
            raise NotImplementedError
        else:
            self.attn = MaskedMHCCA(
                n_embd,
                kv_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, query, key, value, query_mask, kv_mask, pos_embd=None):
        # post-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(query, key, value, query_mask, kv_mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(query) * out_mask_float + self.drop_path_attn(out)
        out = self.ln1(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(out) * out_mask_float)
        out = self.ln2(out)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask

class PreNormCrossTransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kv_embd,               # dimension of the key and value features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False,      # if to add rel position encoding to attention
        gamma_num=0,           # if to use gamma in the output (0: no gamma, 1: gamma in attn, 2: gamma in mlp & attn)
        init_value=1e-1,       # initial value for the gamma
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.ln3 = LayerNorm(kv_embd)

        # specify the attention module
        if mha_win_size > 1:
            # self.attn = LocalMaskedMHCA(
            #     n_embd,
            #     n_head,
            #     window_size=mha_win_size,
            #     n_qx_stride=n_ds_strides[0],
            #     n_kv_stride=n_ds_strides[1],
            #     attn_pdrop=attn_pdrop,
            #     proj_pdrop=proj_pdrop,
            #     use_rel_pe=use_rel_pe  # only valid for local attention
            # )
            raise NotImplementedError
        else:
            self.attn = MaskedMHCCA(
                n_embd,
                kv_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()
        
        # gamma
        if gamma_num >= 1:
            self.gamma1 = nn.Parameter(torch.ones([1, n_embd, 1]) * init_value, requires_grad=True)
        else:
            self.gamma1 = 1.0
        if gamma_num >= 2:
            self.gamma2 = nn.Parameter(torch.ones([1, n_embd, 1]) * init_value, requires_grad=True)
        else:
            self.gamma2 = 1.0

    def forward(self, query, key, value, query_mask, kv_mask, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(query), self.ln3(key), self.ln3(value), query_mask, kv_mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(query) * out_mask_float + self.drop_path_attn(out * self.gamma1)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float * self.gamma2) 
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask

class PreNormDINOTransformerBlock(PreNormCrossTransformerBlock):
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kv_embd,               # dimension of the key and value features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False,      # if to add rel position encoding to attention
        init_value=1e-1,      # initial value for the scale
    ):
        super().__init__(
            n_embd, kv_embd, n_head, n_ds_strides, n_out, n_hidden,
            act_layer, attn_pdrop, proj_pdrop, path_pdrop, mha_win_size, use_rel_pe
        )
        self.gamma = nn.Parameter(torch.ones([1, n_embd, 1]) * init_value, requires_grad=True)
        self.ln2 = None
        self.mlp = None
        self.drop_path_mlp = None
        

    def forward(self, query, key, value, query_mask, kv_mask, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(query), self.ln3(key), self.ln3(value), query_mask, kv_mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(query) * out_mask_float + self.drop_path_attn(out * self.gamma)
        return out, out_mask

class MaskCrossMamba(nn.Module):
    '''
    A wrapper for the cross mamba attention module with masking
    '''
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kv_embd,               # dimension of the key and value features
        kernel_size=4,         # conv kernel size
        channel_agg=False,    # whether to use channel aggregation 
        ca_type='ca',         # type of channel aggregation
    ):
        super().__init__()
        self.cross_mamba = CrossMamba(n_embd, kv_embd, expand=1, d_conv=kernel_size, 
                                      use_fast_path=True, channel_agg=channel_agg, ca_implement_type=ca_type)
    def forward(self, query, value, query_mask, kv_mask):
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)
        out = self.cross_mamba(query, value).transpose(1, 2) 
        out_mask = query_mask & kv_mask # NOTE: kv_mask is the same as query_mask, the implementation is wrong probably
        out = out* out_mask.to(out.dtype)
        return out, out_mask


class PreNormCrossMambaBlock(nn.Module):
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kv_embd,               # dimension of the key and value features
        n_out=None,            # output dimension, if None, set to input dim
        init_value=1e-1,       # initial value for the scale
        path_pdrop=0.0,        # drop path rate
        channel_agg=False,     # whether to use channel aggregation
        ca_type='ca',          # type of channel aggregation
    ):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(kv_embd)
        self.gamma = nn.Parameter(torch.ones([1, n_embd, 1]) * init_value, requires_grad=True)
        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
        self.cross_mamba = MaskCrossMamba(n_embd, kv_embd, channel_agg=channel_agg, ca_type=ca_type)

    def forward(self, query, value, query_mask, kv_mask, pos_embd=None):
        out, out_mask = self.cross_mamba(self.ln1(query), self.ln2(value), query_mask, kv_mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.drop_path_attn(out * self.gamma) + query * out_mask_float
        return out, out_mask

class MulScaleTransformerBlock(TransformerBlock):
    def __init__(
        self,
        n_embd,
        n_head,
        n_ds_strides=(1, 1),
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
        mha_win_size=-1,
        use_rel_pe=False,
        scale_factor=2,
        pool_method='avg',
    ):
        super().__init__(
            n_embd, n_head, n_ds_strides, n_out, n_hidden,
            act_layer, attn_pdrop, proj_pdrop, path_pdrop,
            mha_win_size, use_rel_pe
        )
        # specify the attention module
        if mha_win_size > 1:
            raise NotImplementedError
        else:
            self.attn = MaskedPoolingMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                pool_qx_size=scale_factor,
                pool_kv_size=scale_factor,
                pool_method=pool_method
            )
        
    def forward(self, x, mask, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask

class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kernel_size=3,         # conv kernel size
        n_ds_stride=1,         # downsampling stride for the current layer
        expansion_factor=2,    # expansion factor of feat dims
        n_out=None,            # output dimension, if None, set to input dim
        act_layer=nn.ReLU,     # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

        # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = n_embd * expansion_factor
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
    

class MaxPooler(nn.Module):
    def __init__(
            self,
            kernel_size,
            stride,
            padding: int = 0,):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(
            kernel_size, stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x, mask, **kwargs):

        # out, out_mask = self.channel_att(x, mask)

        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=x.size(-1) // self.stride, mode='nearest')
        else:
            # masking out the features
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.to(x.dtype)

        return out, out_mask.bool()

class AvgPooler(MaxPooler):
    def __init__(
            self,
            kernel_size,
            stride,
            padding=0,):
        super().__init__(kernel_size, stride, padding)
        self.ds_pooling = nn.AvgPool1d(
            kernel_size, stride=stride, padding=padding)

class MaskMambaBlock(nn.Module):
    def __init__(
        self,        
        n_embd,                # dimension of the input features
        kernel_size=4,         # conv kernel size
        n_ds_stride=1,         # downsampling stride for the current layer
        drop_path_rate=0.3,         # drop path rate
        use_mamba_type="dbm",
        pool_method='max',          # pooling method
    ) -> None:
        super().__init__()
        if use_mamba_type == 'dbm':
            self.mamba = DBM(n_embd, d_conv=kernel_size, use_fast_path=True, expand=1)
        elif use_mamba_type == "vim":
            # vim
            self.mamba = ViM(n_embd, d_conv=kernel_size, bimamba_type="v2", use_fast_path=True)
        elif use_mamba_type == "v2m":
            # v2m
            self.mamba = V2M(n_embd, d_conv=kernel_size, bimamba_type="2d", use_fast_path=True)
        else:
            raise NotImplementedError
        if n_ds_stride > 1:
            if pool_method == 'avg':
                self.downsample = AvgPooler(kernel_size=n_ds_stride, stride=n_ds_stride, padding=0)
            elif pool_method == 'max':
                self.downsample = MaxPooler(kernel_size=n_ds_stride, stride=n_ds_stride, padding=0)
            else:
                raise NotImplementedError
        else:
            self.downsample = None
            
        
        self.norm = nn.LayerNorm(n_embd)
                
        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_ = x.transpose(1,2)
        x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)
        x = x_ * mask.to(x.dtype)

        x  = res + self.drop_path(x)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return  x, mask

class MaskMultiScaleMambaBlock(nn.Module):
    '''
    Mamba block with MViT style pooling
    '''
    def __init__(
        self,        
        n_embd,                     # dimension of the input features
        kernel_size=4,              # conv kernel size
        n_ds_stride=1,              # downsampling stride for the current layer
        drop_path_rate=0.3,         # drop path rate
        pool_method:str='avg',      # pooling method       
        use_mamba_type="mdbm",
        gamma=0.0,
    ) -> None:
        super().__init__()
        assert pool_method in ['avg', 'max', 'conv', 'none', '2d-avg', '2d-max', '2d-conv', '2d_avg', '2d_max', '2d_conv']
        if n_ds_stride == 1:
            print("Warning: n_ds_stride is 1, no downsampling will be performed.")
        self.stride = n_ds_stride
        pool_method = pool_method.lower()
        if pool_method.startswith('2d'):
            assert math.sqrt(n_ds_stride) == int(math.sqrt(n_ds_stride)), "n_ds_stride must be a square number for 2d pooling"
        if use_mamba_type == 'mdbm':
            self.mamba = MDBM(n_embd, d_conv=kernel_size, pool_method=pool_method, 
                              pool_size=n_ds_stride, use_fast_path=True, expand=1)
        elif use_mamba_type == 'mvim':
            self.mamba = MViM(n_embd, d_conv=kernel_size, bimamba_type="v2", pool_method=pool_method,
                              pool_size=n_ds_stride, use_fast_path=True)
        # elif use_mamba_type == 'mv2m':
        #     self.mamba = MV2M(n_embd, d_conv=kernel_size, bimamba_type="2d", pool_method=pool_method,
        #                       pool_size=n_ds_stride, use_fast_path=True)
        else:
            raise NotImplementedError
              
        self.norm = nn.LayerNorm(n_embd)
        if pool_method == 'avg':
            self.pooler = AvgPooler(kernel_size=n_ds_stride, stride=n_ds_stride, padding=0)
        elif pool_method == 'max':
            self.pooler = MaxPooler(kernel_size=n_ds_stride, stride=n_ds_stride, padding=0)
        elif pool_method == 'conv':
            kernel_size = n_ds_stride + 1
            padding = (n_ds_stride + 1) // 2
            self.pooler = MaskedConv1D(n_embd, n_embd, kernel_size=kernel_size, stride=n_ds_stride, padding=padding, groups=n_embd)
        elif pool_method == 'none' or pool_method is None:
            self.pooler = nn.Identity()
        elif pool_method == '2d-avg' or pool_method == '2d_avg':
            stride = int(math.sqrt(n_ds_stride))
            stride = (stride, stride)
            pool = nn.AvgPool2d(stride, stride, padding=0)
            self.pooler = partial(self._2d_pooling, pool=pool)
        elif pool_method == '2d-max' or pool_method == '2d_max':
            stride = int(math.sqrt(n_ds_stride))
            stride = (stride, stride)
            pool = nn.MaxPool2d(stride, stride, padding=0)
            self.pooler = partial(self._2d_pooling, pool=pool)
        elif pool_method == '2d-conv' or pool_method == '2d_conv': # TODO: need to test and verify (considering different length of input)
            n_ds_stride = int(math.sqrt(n_ds_stride))
            padding = (n_ds_stride + 1) // 2

            kernel_size = (n_ds_stride + 1, n_ds_stride + 1)
            padding = (padding, padding)
            n_ds_stride = (n_ds_stride, n_ds_stride)
            pool = nn.Conv2d(n_embd, n_embd, kernel_size=kernel_size, stride=n_ds_stride, padding=padding, groups=n_embd)
            self.pooler = partial(self._2d_pooling, pool=pool)
        else:
            raise NotImplementedError
        
        if gamma > 0:
            self.gamma = nn.Parameter(torch.ones([1, n_embd, 1]) * gamma, requires_grad=True)
        else:
            self.gamma = 1

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res, mask = self.pooler(x, mask)
        x_ = x.transpose(1,2)
        x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)

        x = x_ * mask.to(x.dtype)

        x  = res + self.drop_path(x * self.gamma)

        return  x, mask

    def _2d_pooling(self, x, mask, pool):
        B, C, N = x.size()
        stride = int(math.sqrt(self.stride))
        h = w = int(np.sqrt(N))
        x = x.view(B, C, h, w)
        mask = mask.view(B, 1, h, w)

        x = pool(x)
        # downsample the mask using nearest neighbor
        mask = F.interpolate(
            mask.to(x.dtype), size=(h // stride, w // stride), mode='nearest')

        x = x.view(B, C, -1)
        mask = mask.view(B, 1, -1)

        mask = mask.bool()

        return x, mask
    
class LocalGlobalTemporalEncoder(nn.Module):
    def __init__(self, input_dim, dropout, temporal_scale=128, window_size=41,use_vswg=False):
        super(LocalGlobalTemporalEncoder, self).__init__()
        dim_feedforward = 256       # bottleneck like (512 -> 256 -> 512)
        self.self_atten = GlobalLocalAttention(input_dim, num_heads=8, dropout=0.11, temporal_scale=temporal_scale, window_size=window_size,use_vswg=use_vswg)
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, mask):  # TODO: add mask
        # mask [B, 1, T]
        src = features.permute(2, 0, 1) # A2NET: b, 512, 128; actionmamba: b, 512, 192
        # src: [T, B, C]
        q = k = src
        src2 = self.self_atten(q, k, src, mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src2 = src2 * mask.permute(2, 0, 1).to(src2.dtype)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = src.permute(1, 2, 0)
        return src, mask


class GlobalLocalAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout, temporal_scale, window_size=9,use_vswg=False):
        super(GlobalLocalAttention, self).__init__()
        self.num_heads = num_heads
        self.temporal_scale = temporal_scale
        self.use_slide=use_vswg
        self.scale_attention = MultiheadAttention(input_dim,
                                                  num_heads=self.num_heads,
                                                  use_slide=use_vswg,
                                                  dropout=dropout)
        self.mask_matrix = nn.Parameter(self._mask_matrix(window_size).float(), requires_grad=False)
        if use_vswg:
            self.attenLinear1=nn.Linear(512,1)
            self.attenLinear2=nn.Linear(512,1)
            nn.init.kaiming_normal_(self.attenLinear1.weight,mode='fan_in')
            nn.init.kaiming_normal_(self.attenLinear2.weight,mode='fan_in')
        self.window_size=window_size
        self.local_len=self.num_heads//2
        

    def _mask_matrix(self, window_size):
        m = torch.ones((1, self.num_heads,
                        self.temporal_scale,
                        self.temporal_scale), dtype=torch.bool)
        w_len = window_size
        local_len = self.num_heads//2
        #local_len = 0
        for i in range(local_len):
            for j in range(self.temporal_scale):
                for k in range(w_len):
                    m[0, i, j, min(max(j - w_len // 2 + k, 0), self.temporal_scale - 1)] = False

        for i in range(local_len, self.num_heads):
            m[0, i] = False
        return m
    
    def getwin_pro1(self,heads_query):
        avg_head_query=nn.functional.avg_pool2d(heads_query,kernel_size=(3,1),stride=1,padding=(1,0))
        relu_head_query=nn.functional.leaky_relu(avg_head_query)
        windows_pro=self.attenLinear1(relu_head_query.view(-1,self.temporal_scale,512)).view(-1,4,self.temporal_scale,1)
        windows_pro=torch.nn.functional.sigmoid(windows_pro)*((self.window_size+1)//2)
        windows_pro=windows_pro.repeat(1,1,1,(self.window_size+1)//2)
        T=0.0001
        pos_emb=torch.from_numpy(np.arange(0,(self.window_size+1)//2)).to(windows_pro.device).view(1,1,1,-1)
        pos_emb=pos_emb.repeat(windows_pro.shape[0],windows_pro.shape[1],windows_pro.shape[2],1)
        gumble_pro=torch.nn.functional.sigmoid((windows_pro-(self.window_size-1)//2+pos_emb)/T)
        return gumble_pro

    def getwin_pro2(self,heads_query, feat_mask):
        # feat_mask: b, 1, t, heads_query: b, 4, t, c
        avg_head_query=nn.functional.avg_pool2d(heads_query,kernel_size=(3,1),stride=1,padding=(1,0))
        avg_head_query=avg_head_query * feat_mask.unsqueeze(-1).float()
        relu_head_query=nn.functional.leaky_relu(avg_head_query)
        windows_pro=self.attenLinear2(relu_head_query.view(-1,self.temporal_scale,512)).view(-1,4,self.temporal_scale,1)
        # windows_pro: b, 4, t, 1
        windows_pro= windows_pro.masked_fill(feat_mask.unsqueeze(-1)==0, float("-inf"))
        windows_pro=torch.nn.functional.sigmoid(windows_pro)*((self.window_size+1)//2)
        windows_pro=windows_pro.repeat(1,1,1,(self.window_size+1)//2)
        T=0.0001
        pos_emb=torch.from_numpy(np.arange(0,(self.window_size+1)//2)).to(windows_pro.device).view(1,1,1,-1)
        pos_emb=pos_emb.repeat(windows_pro.shape[0],windows_pro.shape[1],windows_pro.shape[2],1)
        # pos_emb: b, 4, t, 1
        feat_mask=feat_mask.to(pos_emb.device)
        residual=((windows_pro-pos_emb)/T).masked_fill(feat_mask.unsqueeze(-1)==0, float("-inf"))
        gumble_pro=torch.nn.functional.sigmoid(residual)
        return gumble_pro # b, 4, t, 5

    def forward(self, query, key, value, feat_mask):
        # feat_mask: b, 1, t; query: t, b, 512
        b = query.size(1) # batch size
        mask_mask = self.mask_matrix.bool().repeat(b, 1, 1, 1) # b, 8, t, t
        mask=torch.zeros((b,self.num_heads,self.temporal_scale,self.temporal_scale)).float().to(query.device) # b, 8, t, t
        if self.use_slide:
            heads_query=query.permute(1,0,2).contiguous().unsqueeze(1).repeat(1,self.num_heads//2,1,1) # b, t, c => b, 1, t, c => b, 4, t, c
            gumble_pro2=self.getwin_pro2(heads_query, feat_mask)
            pro_num=gumble_pro2.shape[3] # 5
            gumble_pro_all=torch.zeros(gumble_pro2.shape[:-1]+(pro_num*2-1,)).to(query.device) # b, 4, t, 9
            for i in range(pro_num-1):
                gumble_pro_all[:,:,:,i]=gumble_pro2[:,:,:,pro_num-1-i]
            gumble_pro_all[:,:,:,pro_num-1:pro_num*2-1]=gumble_pro2
            for i in range(self.local_len):
                for j in range(self.temporal_scale):
                    start_ind=max(0,j - self.window_size // 2)
                    end_ind=min(j + self.window_size // 2,self.temporal_scale-1)
                    mask[:, i, j, start_ind:end_ind+1] =gumble_pro_all[:,i,j,start_ind-j+self.window_size // 2:end_ind-j+self.window_size//2+1] 
            mask[:,self.local_len:,:,:]=1
        # feat_mask: b, 1, t => b, 1, t, t
        feat_mask = feat_mask.unsqueeze(2).repeat(1, 1, self.temporal_scale, 1).bool()
        mask *= feat_mask.float().to(mask.device)
        mask_mask *= feat_mask.bool().to(mask_mask.device)
        r, w = self.scale_attention(query, key, value, key_padding_mask=mask_mask,slide_mask=mask)
        return r, w
    

class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __annotations__ = {
        'bias_k': torch._jit_internal.Optional[torch.Tensor],
        'bias_v': torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'in_proj_weight']

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,use_slide=False):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.use_slide=use_slide
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None,slide_mask=None):
        # type: (Tensor, Tensor, Tensor, Optional[Tensor], bool, Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                slide_mask=slide_mask,use_slide=self.use_slide)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                slide_mask=slide_mask,use_slide=self.use_slide)

def multi_head_attention_forward(query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None,                   # type: Optional[Tensor]
                                 slide_mask=None,                 # type: Optional[Tensor]
                                 use_slide=None                   # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == num_heads
        assert key_padding_mask.size(2) == tgt_len
        assert key_padding_mask.size(3) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if not use_slide:
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask,
                float('-inf'),
            )
        else:
            attn_output_weights = attn_output_weights
            attn_output_weights=attn_output_weights.masked_fill(
                key_padding_mask.detach(),
                float('-inf'),
            )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)
    if use_slide:
        attn_output_weights=attn_output_weights*slide_mask.view(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """
    def __init__(
        self,
        dim_in=3,
        dim_out=768,
        kernel=(1, 16, 16),
        stride=(1, 4, 4),
        padding=(1, 7, 7),
        conv_2d=False,
    ):
        super().__init__()
        if conv_2d:
            conv = nn.Conv2d
        else:
            conv = nn.Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

    def forward(self, x, keep_spatial=False):
        x = self.proj(x)

        if keep_spatial: 
            return x, x.shape
        # B C (T) H W -> B (T)HW C
        return (
            x.flatten(2).transpose(1, 2),
            x.shape,
        )  

class SGPBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            downsample_type='max',
            init_conv_vars=1  # init gaussian variance for the weight
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = n_ds_stride

        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)
        convw = self.convw(out)
        convkw = self.convkw(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (convw + convkw) * psi + out

        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()

    
if __name__ == '__main__':
    block = MaskMultiScaleMambaBlock(
        n_embd=256,
        n_ds_stride=4,
        pool_method='max',
        use_mamba_type='mv2m',
    )
    x = torch.rand(1, 256, 64)
    mask = torch.ones(1, 1, 64)

    # all to cuda
    x = x.cuda()
    mask = mask.cuda()
    block = block.cuda()

    y, mask = block(x, mask)
    print(y.shape, mask.shape)