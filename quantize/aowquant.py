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
        onehot = torch.nn.functional.one_hot(outlier_channels, num_classes=group_size).sum(-2).view(-1).bool()
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


def get_unified_postlayernorm_outlier_index(act_stats, outlier_ratio, group_size, logger=None):
    """
    Use unified outlier mask for post-layernorm activations
    """
    post_layernorm_stats = {k: v['output'] for k, v in act_stats.items() if 'norm' in k}
    attention_list = []

    for v in post_layernorm_stats.values():
        output_scale = v['max'] - v['min']
        attention = torch.softmax(output_scale / output_scale.mean(), dim=-1)
        attention_list.append(attention)

    all_attention = torch.stack(attention_list, dim=0).mean(dim=0)

    unified_outlier_index = get_outlier_channel_index(
        all_attention,
        outlier_ratio,
        group_size = group_size,
        outlier_metric = 'scale',
        logger = logger,
    )
    return unified_outlier_index


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
    
    # unified postlayernorm outlier index
    unified_postlayernorm_outlier_index = get_unified_postlayernorm_outlier_index(
        act_stats,
        args.act_outlier_ratio,
        args.act_group_size,
        logger=logger,
    ).to(dev)

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

                if not module.use_act_quant:
                    continue
                logger.info(f"Set activation quantizer of {name} to {module.use_act_quant}")

                all_stats = act_stats[f"{layer_name_prefix}.{i}.{name}"]
                # act_scale = all_stats['abs_input']['max'].to(device=dev, dtype=a_dtype).clamp(1e-5)
                act_scale = (all_stats['input']['max'] - all_stats['input']['min']).to(device=dev, dtype=a_dtype).clamp(1e-5)

                # not using act reordering for oproj
                if linear_category == 'oproj':
                    use_act_reorder = False
                    select_outlier_within_group = True
                elif linear_category == 'fc2':
                    use_act_reorder = args.act_reorder
                    select_outlier_within_group = False
                else:
                    use_act_reorder = not args.act_unified_postlayernorm_outlier
                    select_outlier_within_group = False

                # enlarge outlier ratio for alignment in grouping
                if args.act_group_size:
                    num_outlier_per_group = math.ceil(args.act_group_size * args.act_outlier_ratio)
                    new_outlier_ratio = num_outlier_per_group / args.act_group_size
                    if new_outlier_ratio > args.act_outlier_ratio:
                        logger.info(f"Outlier ratio enlarged from {args.act_outlier_ratio} to {new_outlier_ratio} for alignment in grouping")
                        args.act_outlier_ratio = new_outlier_ratio

                # select outlier channels， and generate outlier mask before reordering
                if args.act_outlier_ratio > 0:
                    if args.act_unified_postlayernorm_outlier and linear_category in ('fc1', 'qkvproj'):
                        outlier_index = deepcopy(unified_postlayernorm_outlier_index)
                    else:
                        outlier_index = get_outlier_channel_index(
                            act_scale,
                            args.act_outlier_ratio,
                            # if use reordering, grouping is ignored
                            group_size = None if select_outlier_within_group else 128,  # we hardcode this for convenience
                            outlier_metric = args.outlier_metric,
                            logger = logger,
                        )

                    outlier_mask = torch.nn.functional.one_hot(outlier_index, num_classes=act_scale.shape[0]).sum(dim=0).bool()
                
                else:
                    outlier_index = None
                    outlier_mask = torch.zeros(act_scale.shape[0], dtype=torch.bool)

                outlier_mask = outlier_mask.to(device=dev)

                # select normal channels (probably reordered)
                if use_act_reorder:
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

                # # check if reordered arange matches both indexes
                # check_outlier_index = torch.masked_select(reorder_index, final_outlier_mask)
                # # check every element in check outlier index is in outlier index
                # for ol in check_outlier_index:
                #     assert ol in outlier_index

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

                # set scale and round zero point of normal values
                if args.a_dynamic_method == 'none':
                    xmax = all_stats['input']['max'].to(device=dev, dtype=a_dtype)
                    xmin = all_stats['input']['min'].to(device=dev, dtype=a_dtype)
                    xmax = torch.index_select(xmax, dim=-1, index=reorder_index) * ~final_outlier_mask
                    xmin = torch.index_select(xmin, dim=-1, index=reorder_index) * ~final_outlier_mask
                    if args.act_group_size:
                        # we simply assume grouping has no deficiency!
                        xmax = xmax.view(-1, args.act_group_size)
                        xmin = xmin.view(-1, args.act_group_size)
                    scale, round_zero_point = get_scale_zero_point(xmin, xmax, args.abits, efficient_groupwise=args.act_group_efficient_accumulation)
                    if args.act_group_size is None:
                        logger.info(f"Scale: {scale.item()}")
                    set_quantizer_scale_round_zero_point(module.act_quantizer, scale, round_zero_point)

                # We first reorder, then apply outlier mask, after that grouping
                module.act_quantizer.register_buffer('reorder_index', reorder_index)
                module.act_quantizer.register_buffer('reorder_index_inv', reorder_index_inv)
                module.act_quantizer.register_buffer('outlier_mask', final_outlier_mask)

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