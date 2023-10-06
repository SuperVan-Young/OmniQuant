# Activation-Outlier-aWare Quantization

import torch
import torch.nn as nn
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatmul
from quantize.omniquant import get_named_linears
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc

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

                # set high precision channels
                all_stats = act_stats[f"{layer_name_prefix}.{i}.{name}"]
                act_scale = all_stats['abs_input']['max'].to(device=dev, dtype=dtype).clamp(1e-5)
                num_high_prec_channels = int(act_scale.shape[0] * args.high_prec_ratio)
                _, high_prec_channels = torch.topk(act_scale, num_high_prec_channels)
                module.act_quantizer.high_prec_channels = [int(i) for i in high_prec_channels]

            elif isinstance(module, QuantMatmul):
                if "qkt_matmul" in name:
                    module.use_x1_quant = getattr(args, "aow-quant-act-q")
                    module.use_x2_quant = getattr(args, "aow-quant-act-k")
                elif "pv_matmul" in name:
                    module.use_x2_quant = getattr(args, "aow-quant-act-v")

        qlayer.register_scales_and_zeros()
        qlayer.half()
        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

    gc.collect()
    return model