import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import tqdm
import numpy as np
import pdb
import math

CLIPMIN = 1e-5




def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric="minmax",
        dynamic=False,
        dynamic_method="per_cluster",
        group_size=None,
        shape=None,
        lwc=False,
    ):
        """
        support cluster quantize
        dynamic_method support per_token and per_cluster
        """
        super().__init__()
        self.symmetric = symmetric
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc

        init_value = 4.             # inti value of learnable weight clipping
        if lwc:
            if group_size:
                raise RuntimeError("New version of grouping does not support lwc yet.")
                dim1 = int(shape[0]*math.ceil(shape[1]/group_size))
                self.deficiency = shape[-1]%group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    assert self.symmetric   # support for mlc-llm symmetric quantization
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1,1))*init_value)
        self.sigmoid = nn.Sigmoid()

        self.enable = True
        self.group_size = group_size

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1

    def fake_quant(self, x, scale, round_zero_point):
        if self.group_size:
            x_org_shape = x.shape
            x = self.group_tensor(x)

        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = self.degroup_tensor(x_dequant, x_org_shape)
        return x_dequant

    def keep_high_prec(self, x, x_quant):
        mask = torch.zeros_like(x_quant, dtype=torch.bool)
        if len(mask.shape) == 3:
            mask[:, :, self.high_prec_channels] = True
        else:
            raise RuntimeError(f"Only support 3D tensor now, got shape {mask.shape}")
        x_out =  torch.where(mask, x, x_quant)
        return x_out


    def forward_normal(self, x: torch.Tensor):
        """
        Original forward function for Omniquant
        We use it for normal value quantization.
        """
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits-1).round_().div_(2**self.n_bits-1)

        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)  # x remains the same shape
        else:
            assert self.scale is not None
            assert self.round_zero_point is not None

        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)

        return x_dequant
    
    def forward(self, x: torch.Tensor):
        """
        Add reordering and outlier masking to the forward function
        """
        # reorder input tensor
        if hasattr(self, 'reorder_index') and self.reorder_index is not None:
            assert x.device == self.reorder_index.device, f"{x.device} vs {self.reorder_index.device}"
            x = torch.index_select(x, dim=-1, index=self.reorder_index)

        if hasattr(self, 'outlier_mask') and self.outlier_mask is not None:
            assert self.metric != 'fix0to1', "Not support fix0to1 with high_prec_channels"
            assert x.device == self.outlier_mask.device, f"{x.device} vs {self.outlier_mask.device}"
            if len(x.shape) == 3:
                # linear activation
                assert len(self.outlier_mask.shape) == 1, f"{self.outlier_mask.shape}"
                mask = self.outlier_mask.view(1, 1, -1)
            elif len(x.shape) == 4:
                # matmul activation
                raise NotImplementedError
                bsz, num_head, seq_len, head_dim = mask.shape
                mask = mask.transpose(1, 2).view(bsz, seq_len, -1)
                mask[:, :, self.high_prec_channels] = True
                mask = mask.view(bsz, seq_len, num_head, head_dim).transpose(1, 2)
            else:
                raise RuntimeError(f"Only support 3D & 4D tensor now, got shape {x.shape}")
            x_normal = x * (~mask)
            x_outlier = x * mask
            x_dequant = self.forward_normal(x_normal) + x_outlier
        else:
            x_dequant = self.forward_normal(x)

        # reorder the tensor back!
        if hasattr(self, 'reorder_index') and self.reorder_index is not None:
            x_dequant = torch.index_select(x_dequant, dim=-1, index=self.reorder_index_inv)

        has_nan = torch.isnan(x_dequant).any()
        if has_nan:
            print(f"min scale: {self.scale.min()}, max scale: {self.scale.max()}")
            print(f"min round zero point: {self.round_zero_point.min()}, max round zero point: {self.round_zero_point.max()}")
            print(f"num Nan: {torch.isnan(x_dequant).sum()}")
            raise RuntimeError("NaN detected in quantized tensor.")
        has_inf = torch.isinf(x_dequant).any()
        if has_inf:
            raise RuntimeError("Inf detected in quantized tensor.")
        return x_dequant

    def group_tensor(self, x):
        """
        Group the tensor at last dimension, retain all remaining dimensions
        """
        assert self.group_size is not None
        
        # derive grouped shape
        num_group = math.ceil(x.shape[-1] / self.group_size)
        deficiency = num_group * self.group_size - x.shape[-1]
        x_grouped_shape = list(x.shape[:-1]) + [num_group, self.group_size]

        if deficiency == 0:
            x_grouped = x.reshape(*x_grouped_shape)
        else:
            pad_zeros_shape = list(x.shape)[:-1] + [deficiency]
            pad_zeros = torch.zeros(pad_zeros_shape, dtype=x.dtype,device=x.device)
            x_grouped = torch.cat((x,pad_zeros),dim=1).reshape(*x_grouped_shape)
        return x_grouped

    def degroup_tensor(self, x_grouped, x_org_shape):

        num_group = math.ceil(x_org_shape[-1] / self.group_size)
        deficiency = num_group * self.group_size - x_org_shape[-1]
        x_degrouped = x_grouped.reshape(*x_grouped.shape[:-2], -1)

        if deficiency > 0:
            x_degrouped = x_degrouped[...,:-self.deficiency]
        return x_degrouped

    def per_token_dynamic_calibration(self, x):
        # Modified version keeps the shape of x, while getting scale and zero_points
        if self.group_size:
            x_ = self.group_tensor(x)
        else:
            x_ = x
        reduce_shape = [-1]
        xmin = x_.amin(reduce_shape, keepdim=True)
        xmax =  x_.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor)*xmax
            xmin = self.sigmoid(self.lowbound_factor)*xmin
        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point