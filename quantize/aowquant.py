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

def get_outlier_channel_index(
    act_scale,
    act_outlier_ratio,
    group_size = None,
    outlier_metric = 'scale',
    logger = None,

) -> torch.Tensor:
    """
    Select outlier channels from activation scale.
    Returns: 1d tensor of outlier channel indices
    """
    assert outlier_metric == 'scale', "Only support scale now"
    assert act_outlier_ratio > 0 and act_outlier_ratio < 1, "Outlier ratio should be in (0, 1)"
    assert len(act_scale.shape) == 1, "Only support 1D tensor now"

    if group_size is None:
        num_outlier_channels = math.ceil(act_scale.shape[0] * act_outlier_ratio)
        _, outlier_channels = torch.topk(act_scale, num_outlier_channels)
    else:
        num_groups = math.ceil(act_scale.shape[0] / group_size)
        deficiency = num_groups * group_size - act_scale.shape[0]
        num_outlier_channels_per_group = math.ceil(group_size * act_outlier_ratio)
        logger.info(f"Outlier channels per group: {num_outlier_channels_per_group}")

        # pad zero and group act_state
        if deficiency == 0:
            act_scale_grouped = act_scale.reshape(num_groups, group_size)
        else:
            pad_zeros = torch.zeros([deficiency], dtype=act_scale.dtype, device=act_scale.device)
            act_scale_grouped = torch.cat([act_scale, pad_zeros], dim=-1).reshape(num_groups, group_size)
        
        # select topk for each group
        _, outlier_channels = torch.topk(act_scale_grouped, num_outlier_channels_per_group, dim=-1)

        # flatten outlier channels
        onehot = torch.nn.functional.one_hot(outlier_channels, num_classes=group_size).view(-1).bool()
        if deficiency:
            onehot = onehot[:-deficiency]
        outlier_channels = torch.masked_select(torch.arange(act_scale.shape[0], device=act_scale.device), onehot)
    
    assert len(outlier_channels.shape) == 1, "Outlier channels should be 1D tensor"
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
        outlier_mask = torch.zeros(act_scale.shape[0], dtype=torch.bool, device=act_scale.device)
        logger.info("No outlier mask provided, select all channels as normal channels")

    normal_act_scale = torch.where(outlier_mask, 
                                   torch.zeros_like(act_scale, dtype=act_scale.dtype, device=act_scale.device),
                                   act_scale)
    _, reorder_indices = torch.sort(normal_act_scale, descending=True)

    num_outlier_channels = outlier_mask.sum().item()
    if num_outlier_channels:
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
        if args.debug and i > 0:
            break

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
                    num_outlier_per_group = math.ceil(args.act_group_size * args.act_outlier_ratio)
                    new_outlier_ratio = num_outlier_per_group / args.act_group_size
                    if new_outlier_ratio > args.act_outlier_ratio:
                        logger.info(f"Outlier ratio enlarged from {args.act_outlier_ratio} to {new_outlier_ratio} for alignment in grouping")
                        args.act_outlier_ratio = new_outlier_ratio

                # select outlier channels， and generate outlier mask before reordering
                if args.act_outlier_ratio > 0:
                    outlier_index = get_outlier_channel_index(
                        act_scale,
                        args.act_outlier_ratio,
                        # if use reordering, grouping is ignored
                        group_size = None if args.act_reorder else args.act_group_size,
                        outlier_metric = args.outlier_metric,
                        logger = logger,
                    )

                    outlier_mask = torch.nn.functional.one_hot(outlier_index, num_classes=act_scale.shape[0]).sum(dim=0).bool()
                
                else:
                    outlier_index = None
                    outlier_mask = torch.zeros(act_scale.shape[0], dtype=torch.bool)

                outlier_mask = outlier_mask.to(device=dev)

                # select normal channels (probably reordered)
                if args.act_reorder:
                    # reorder normal channels
                    normal_index = get_reorder_channel_index(
                        act_scale,
                        outlier_mask = outlier_mask,
                        reorder_metric = args.reorder_metric,
                        logger = logger,
                    )
                else:
                    # just pick out normal channels
                    normal_index = torch.masked_select(torch.arange(act_scale.shape[0], device=act_scale.device), ~outlier_mask)

                # concat outlier index if necessary
                if outlier_index is None:
                    reorder_index = normal_index
                    final_outlier_mask = torch.zeros(act_scale.shape[0], dtype=torch.bool)
                else:
                    if args.act_group_size:
                        num_groups = math.ceil(act_scale.shape[0] / args.act_group_size)

                        num_outlier_per_group = math.ceil(args.act_group_size * args.act_outlier_ratio)
                        num_normal_per_group = args.act_group_size - num_outlier_per_group

                        # spread outliers in each group
                        normal_deficiency = num_groups * num_normal_per_group - normal_index.shape[0]
                        assert normal_deficiency < num_normal_per_group
                        if normal_deficiency:
                            normal_index = torch.cat([normal_index, 
                                                      torch.ones(normal_deficiency, dtype=normal_index.dtype, device=normal_index.device) * -1], 
                                                      dim=-1)
                        normal_index = normal_index.view(-1, num_normal_per_group)
                        outlier_index = outlier_index.view(-1, num_outlier_per_group)
                        assert normal_index.shape[0] == num_groups
                        assert outlier_index.shape[0] == num_groups

                        reorder_index = torch.cat([outlier_index, normal_index], dim=-1).view(-1)[:act_scale.shape[0]]
                        # put outlier index to the front is in favor for padding

                        assert (reorder_index == -1).sum() == 0
                
                        # adjust outlier map
                        final_outlier_mask = torch.cat([
                            torch.ones(num_groups, num_outlier_per_group, dtype=torch.bool),
                            torch.zeros(num_groups, num_normal_per_group, dtype=torch.bool)],
                            dim=-1).view(-1)[:act_scale.shape[0]]
                            
                    else:
                        reorder_index = torch.cat([outlier_index, normal_index], dim=-1).view(-1)
                        final_outlier_mask = torch.cat([
                            torch.ones(len(outlier_index), dtype=torch.bool),
                            torch.zeros(len(normal_index), dtype=torch.bool)],
                            dim=-1).view(-1)
                        
                final_outlier_mask = final_outlier_mask.to(device=dev)

                # check reorder index is a valid permuatation
                check_reorder_index = torch.nn.functional.one_hot(reorder_index, num_classes=act_scale.shape[0]).int().sum(dim=0)
                assert torch.all(check_reorder_index == 1)

                # generate inverted reorder index
                _, reorder_index_inv = torch.sort(reorder_index, descending=False)

                # check whether normal channels and outlier channels are separated
                if args.act_outlier_ratio > 0:
                    act_scale_reorder = torch.index_select(act_scale, dim=-1, index=reorder_index)
                    act_scale_outlier = act_scale_reorder * final_outlier_mask
                    act_scale_normal = act_scale_reorder * (~final_outlier_mask)
                    act_scale_outlier = act_scale_outlier[act_scale_outlier > 0]
                    act_scale_normal = act_scale_normal[act_scale_normal > 0]

                    max_act_scale_outlier = torch.max(act_scale_outlier).item()
                    max_act_scale_normal = torch.max(act_scale_normal).item()
                    logger.info(f"outlier max vs. normal max: {max_act_scale_outlier} vs. {max_act_scale_normal}")
                    
                    mean_act_scale_outlier = torch.mean(act_scale_outlier).item()
                    mean_act_scale_normal = torch.mean(act_scale_normal).item()
                    logger.info(f"outlier mean vs. normal mean: {mean_act_scale_outlier} vs. {mean_act_scale_normal}")

                # We first reorder, then apply outlier mask, after that grouping
                module.act_quantizer.register_buffer('reorder_index', reorder_index)
                module.act_quantizer.register_buffer('reorder_index_inv', reorder_index_inv)
                module.act_quantizer.register_buffer('outlier_mask', final_outlier_mask)

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