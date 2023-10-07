# Activation-Outlier-aWare Quantization

import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.omniquant import get_named_linears
from quantize.quantizer import pad_zeros
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc

def get_outlier_channel_index(
    act_scale,
    outlier_ratio,
    group_size = None,
    outlier_metric = 'scale',
    logger = None,

) -> torch.Tensor:
    """
    Select outlier channels from activation scale.
    Returns: 1d tensor of outlier channel indices if group_size is None, 
             or 2d tensor of outlier channel indices given group size
    """
    assert outlier_metric == 'scale', "Only support scale now"
    assert outlier_ratio > 0 and outlier_ratio < 1, "Outlier ratio should be in (0, 1)"
    assert len(act_scale.shape) == 1, "Only support 1D tensor now"

    if group_size is None:
        num_outlier_channels = int(act_scale.shape[0] * outlier_ratio)
        _, outlier_channels = torch.topk(act_scale, num_outlier_channels)
    else:
        num_groups = math.ceil(act_scale.shape[0] / group_size)
        deficiency = num_groups * group_size - act_scale.shape[0]
        num_outlier_channels_per_group = math.ceil(group_size * outlier_ratio)
        logger.info(f"Outlier channels per group: {num_outlier_channels_per_group}")

        # pad zero and group act_state
        if deficiency == 0:
            act_scale_grouped = act_scale.reshape(num_groups, group_size)
        else:
            pad_zeros = torch.zeros([deficiency], dtype=act_scale.dtype, device=act_scale.device)
            act_scale_grouped = torch.cat([act_scale, pad_zeros], dim=-1).reshape(num_groups, group_size)
        
        # select topk for each group
        _, outlier_channels = torch.topk(act_scale_grouped, num_outlier_channels_per_group, dim=-1)
    
    return outlier_channels


def get_reorder_channel_index(
    act_scale,
    outlier_mask = None,
    reorder_metric = 'scale',
    logger = None,
):
    """
    Reorder normal channels of activation scale.
    Returns: 1d tensor of reordered normal channel indices
    """
    assert reorder_metric == 'scale', "Only support scale now"
    assert len(act_scale.shape) == 1, "Only support 1D tensor now"
    if outlier_mask is None:
        outlier_mask = torch.zeros(act_scale.shape[0], dtype=torch.bool)
        logger.info("No outlier mask provided, select all channels as normal channels")

    normal_act_scale = torch.where(outlier_mask, torch.zeros_like(act_scale), act_scale)
    _, reorder_indices = torch.sort(normal_act_scale, descending=True)

    num_outlier_channels = outlier_mask.sum()
    reorder_indices = reorder_indices[:-num_outlier_channels]

    return reorder_indices


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
            "q_proj":"qkvproj",
            "k_proj":"qkvproj",
            "v_proj":"qkvproj",
            "o_proj":"oproj",
            "up_proj":"fc1",
            "gate_proj": "fc1",
            "down_proj": "fc2",
        }
        layer_name_prefix = "model.layers"
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
    elif "falcon" in args.net.lower():
        layers = model.transformer.h
        model.transformer.word_embeddings.to(dev)
        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)
        DecoderLayer = QuantFalconDecoderLayer
        layer_name_prefix = "model.transformer.h"
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon now")
    
    if args.deactive_amp:
        dtype = torch.float
    else:
        dtype = torch.float16

    # quantize every layer, and set high precision activation channels
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(lm.model.config, layer, args)
        qlayer = qlayer.to(dev)

        # init smooth parameters
        qlayer.set_quant_state(weight_quant=False, act_quant=False)

        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear):
                # set enabling state of activation quantzier
                linear_category = pairs[name.split(".")[-1]]
                module.use_act_quant = getattr(args, f"aow_quant_act_{linear_category}")
                # logger.info(f"Set activation quantizer of {name} to {module.use_act_quant}")

                if not module.use_act_quant:
                    continue

                all_stats = act_stats[f"{layer_name_prefix}.{i}.{name}"]
                act_scale = all_stats['abs_input']['max'].to(device=dev, dtype=dtype).clamp(1e-5)

                # enlarge outlier ratio for alignment in grouping
                if args.act_group_size:
                    num_outlier_per_group = math.ceil(args.act_group_size * args.outlier_ratio)
                    new_outlier_ratio = num_outlier_per_group / args.act_group_size
                    logger.info(f"Outlier ratio enlarged fron {args.outlier_ratio} to {new_outlier_ratio} for alignment in grouping")
                    args.outlier_ratio = new_outlier_ratio

                # select outlier channels
                # if use reordering, grouping is ignored
                outlier_index = get_outlier_channel_index(
                    act_scale,
                    args.outlier_ratio,
                    group_size = None if args.act_reorder else args.act_group_size,
                    outlier_metric = args.outlier_metric,
                    logger = logger,
                )

                # this is original outlier mask
                # if use reordering, this should be adjusted accordingly
                outlier_mask = torch.nn.functional.one_hot(outlier_index, num_classes=act_scale.shape[0]).bool()
                
                if args.act_reorder:
                    # reorder normal channels
                    normal_index = get_reorder_channel_index(
                        act_scale,
                        outlier_mask = outlier_mask,
                        reorder_metric = args.reorder_metric,
                        logger = logger,
                    )
                else:
                    # Notice that in this implementation, we reorder outlier channels to the front
                    # even if act_reordering is not enabled.
                    # However, this will not affect the final quantization result.
                    # If we use grouping, outlier channels remain in the original group;
                    # Let alone not using grouping.
                    normal_index = torch.masked_select(torch.arange(act_scale.shape[0]), ~outlier_mask)

                # concat reorder index
                if args.act_group_size:
                    num_groups = math.ceil(act_scale.shape[0] / args.act_group_size)

                    num_outlier_per_group = int(args.act_group_size * args.outlier_ratio)
                    num_normal_per_group = args.act_group_size - num_outlier_per_group

                    if num_outlier_per_group:
                        # spread outliers in each group
                        normal_index = pad_zeros(normal_index, num_normal_per_group)
                        normal_index = normal_index.view(-1, num_normal_per_group)
                        outlier_index = outlier_index.view(-1, num_outlier_per_group)
                        reorder_index = torch.cat([outlier_index, normal_index], dim=-1).view(-1)[:act_scale.shape[0]]
                        # put outlier index to the front is in favor for padding

                        assert (reorder_index == 0).sum() == 1, "Only 1 zero should appear in reorder index"
                
                        # adjust outlier map
                        final_outlier_mask = torch.cat([
                            torch.ones(num_groups, num_outlier_per_group),
                            torch.zeros(num_groups, num_normal_per_group)],
                            dim=-1).view(-1)[:act_scale.shape[0]]
                    else:
                        # No grouping is used
                        # directly concat normal and outlier index
                        reorder_index = torch.cat([outlier_index, normal_index], dim=-1).view(-1)

                        # adjust outlier map
                        final_outlier_mask = torch.cat([
                            torch.ones(len(outlier_index)),
                            torch.zeros(len(normal_index))],
                            dim=-1).view(-1)
                        
                    module.act_quantizer.reorder_index = reorder_index
                    module.act_quantizer.outlier_mask = final_outlier_mask

            elif isinstance(module, QuantMatMul):
                if "qkt_matmul" in name:
                    module.use_x1_quant = getattr(args, "aow_quant_act_q")
                    module.use_x2_quant = getattr(args, "aow_quant_act_k")
                elif "pv_matmul" in name:
                    module.use_x2_quant = getattr(args, "aow_quant_act_v")

        qlayer.register_scales_and_zeros()
        qlayer.half()
        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    gc.collect()
    return model