# Activation-Outlier-aWare Quantization

import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.omniquant import get_named_linears
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc
from copy import deepcopy

def get_outlier_channel_index(
    act_scale,
    outlier_ratio,
    outlier_threshold,
    outlier_metric,
    logger = None,

) -> torch.Tensor:
    """
    Select outlier channels from activation scale.
    Returns: 1d tensor of outlier channel indices
    """
    assert len(act_scale.shape) == 1, "Only support 1D tensor now"

    if outlier_metric == 'scale':
        num_outlier_channels = math.ceil(act_scale.shape[0] * outlier_ratio)
        _, outlier_channels = torch.topk(act_scale, num_outlier_channels)
        if len(outlier_channels) == 0:
            return None
        else:
            return outlier_channels
    elif outlier_metric == 'threshold':
        act_scale_mid = torch.median(act_scale).item()
        outlier_channels = torch.where(act_scale > outlier_threshold * act_scale_mid)[0]
        if len(outlier_channels) == 0:
            return None
        else:
            return outlier_channels
    else:
        raise ValueError(f"Unsupported outlier metric {outlier_metric}")    


def get_scale_zero_point(xmin, xmax, abits, is_attention=False, efficient_groupwise=False):
    """
    Get scale and zero point with min-max at reduction dim
    """
    xrange = torch.max(xmax - xmin, dim=-1, keepdim=True)[0]
    scale = xrange / (2 ** abits - 1)
    scale = scale.clamp(min=1e-3, max=1e4)  # 1e-5 is not enough for OPT
    round_zero_point = (-xmin / scale).clamp(min=-1e4, max=1e4).round()

    if is_attention:
        scale = scale.reshape(1, -1, 1, 1)  # bsz, head, seq, hid
        num_attention_head = scale.shape[1]
        round_zero_point = round_zero_point.reshape(1, num_attention_head, 1, -1)

    if efficient_groupwise and scale.numel() > 1:
        # amortize cost of accumulation with shifting
        max_scale = torch.max(scale)  # align with max scale
        scale_exp = torch.log2(scale / max_scale).round().clamp(min=-16)
        scale = max_scale * (2 ** scale_exp)

    return scale, round_zero_point

def set_quantizer_scale_round_zero_point(quantizer, scale, round_zero_point):
    del quantizer.scale
    del quantizer.round_zero_point
    quantizer.register_buffer('scale', scale)
    quantizer.register_buffer('round_zero_point', round_zero_point)

def aowquant(
    lm,
    args,
    dataloader,
    act_stats,
    logger=None,
):
    logger.info("Starting ...")

    # move embedding layer and first layer to target device
    model = lm.model
    dev = lm.device
    use_cache = model.config.use_cache
    model.config.use_cache = False
    is_llama = False
    if "llama" in args.net.lower():
        is_llama = True
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
        DecoderLayer = QuantLlamaDecoderLayer
        pairs = {
            "qkv_proj": "qkvproj",
            "q_proj":"qkvproj",
            "k_proj":"qkvproj",
            "v_proj":"qkvproj",
            "o_proj":"oproj",
            "up_proj":"fc1",
            "gate_proj": "fc1",
            "down_proj": "fc2",
        }
        layer_name_prefix = "model.layers"
        a_dtype = torch.bfloat16
    elif "opt" in args.net.lower():
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        DecoderLayer = QuantOPTDecoderLayer
        pairs = {
            "q_proj":"qkvproj",
            "k_proj":"qkvproj",
            "v_proj":"qkvproj",
            "out_proj":"oproj",
            "fc1":"fc1",
            "fc2":"fc2",
        }
        layer_name_prefix = "model.decoder.layers"
        a_dtype = torch.float16
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")

    # quantize every layer, and set high precision activation channels
    for i in range(len(layers)):
        if args.debug and i > 0:
            break

        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)
        qlayer.org_device = layer.org_device

        # init smooth parameters
        qlayer.set_quant_state(weight_quant=False, act_quant=False)

        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                # set enabling state of activation quantzier
                linear_category = pairs[name.split(".")[-1]]
                module.use_act_quant = getattr(args, f"aow_quant_act_{linear_category}")

                if not module.use_act_quant:
                    continue
                logger.info(f"Set activation quantizer of {name} to {module.use_act_quant}")

                if is_llama and linear_category == 'qkvproj':
                    all_stats = act_stats[f"{layer_name_prefix}.{i}.input_layernorm"]
                    act_scale = (all_stats['output']['max'] - all_stats['output']['min']).to(device=dev, dtype=a_dtype).clamp(1e-5)
                else:
                    all_stats = act_stats[f"{layer_name_prefix}.{i}.{name}"]
                    act_scale = (all_stats['input']['max'] - all_stats['input']['min']).to(device=dev, dtype=a_dtype).clamp(1e-5)

                # parse outlier ratio
                _act_outlier_ratio = getattr(args, f"act_outlier_ratio_{linear_category}")
                if _act_outlier_ratio is None:
                    _act_outlier_ratio = args.act_outlier_ratio

                # select outlier channels， and generate outlier mask before reordering
                outlier_index = get_outlier_channel_index(
                    act_scale,
                    outlier_ratio = _act_outlier_ratio,
                    outlier_threshold = args.act_outlier_threshold,
                    outlier_metric = args.act_outlier_metric,
                    logger = logger,
                )

                if outlier_index is not None:
                    outlier_mask = torch.nn.functional.one_hot(outlier_index, num_classes=act_scale.shape[0]).sum(dim=0).bool()
                else:
                    outlier_mask = torch.zeros_like(act_scale, dtype=torch.bool)
                outlier_mask = outlier_mask.to(device=dev)

                # set scale and round zero point of normal values
                if args.a_dynamic_method == 'none':
                    xmax = all_stats['input']['max'].to(device=dev, dtype=a_dtype) * ~outlier_mask
                    xmin = all_stats['input']['min'].to(device=dev, dtype=a_dtype) * ~outlier_mask
                    scale, round_zero_point = get_scale_zero_point(xmin, xmax, args.abits)
                    set_quantizer_scale_round_zero_point(module.act_quantizer, scale, round_zero_point)

                # Add outlier mask
                module.act_quantizer.register_buffer('outlier_mask', outlier_mask)

            elif isinstance(module, QuantMatMul):
                all_stats = act_stats[f"{layer_name_prefix}.{i}.{name}"]
                
                def get_xmin_xmax(tensor_name):
                    xmax = all_stats[tensor_name]['max'].to(device=dev, dtype=a_dtype)
                    xmin = all_stats[tensor_name]['min'].to(device=dev, dtype=a_dtype)
                    # we only allow per head quantization
                    attention_head_size = lm.model.config.hidden_size // lm.model.config.num_attention_heads
                    xmax = xmax.view(-1, attention_head_size)
                    xmin = xmin.view(-1, attention_head_size)
                    return xmin, xmax

                if "qkt_matmul" in name:
                    module.use_x1_quant = getattr(args, "aow_quant_act_q")
                    module.use_x2_quant = getattr(args, "aow_quant_act_k")

                    if args.aow_quant_act_q:
                        logger.info(f"Quantize activation (q) of {name}")

                        # prepare static scale and zero point
                        if args.a_dynamic_method == 'none':
                            xmin, xmax = get_xmin_xmax('q')
                            scale, round_zero_point = get_scale_zero_point(xmin, xmax, args.abits, is_attention=True)
                            set_quantizer_scale_round_zero_point(module.x1_quantizer, scale, round_zero_point)

                    if args.aow_quant_act_k:
                        logger.info(f"Quantize activation (k) of {name}")

                        # prepare static scale and zero point
                        if args.a_dynamic_method == 'none':
                            xmin, xmax = get_xmin_xmax('k')
                            scale, round_zero_point = get_scale_zero_point(xmin, xmax, args.abits, is_attention=True)
                            set_quantizer_scale_round_zero_point(module.x2_quantizer, scale, round_zero_point)

                elif "pv_matmul" in name:
                    module.use_x2_quant = getattr(args, "aow_quant_act_v")

                    if args.aow_quant_act_v:
                        logger.info(f"Quantize activation (v) of {name}")

                        # prepare static scale and zero point
                        if args.a_dynamic_method == 'none':
                            xmin, xmax = get_xmin_xmax('v')
                            scale, round_zero_point = get_scale_zero_point(xmin, xmax, args.abits, is_attention=True)
                            set_quantizer_scale_round_zero_point(module.x2_quantizer, scale, round_zero_point)

        # qlayer.register_scales_and_zeros()
        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    gc.collect()
    return model