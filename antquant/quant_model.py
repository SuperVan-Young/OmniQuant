import torch
import torch.nn as nn
from quant_modules import TensorQuantizer, Conv2dQuantizer, LinearQuantizer, Conv1dQuantizer
from quant_utils import quant_args
from transformers import pytorch_utils


def quantize_model(model, quantizer_args):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # print(f"model: {model}")
    # print(f"model type {type(model)}")

    # quantize layers
    if type(model) == nn.Conv2d:
        quant_mod = Conv2dQuantizer(**quant_args)
        quant_mod.set_param(model)
        del model
        return quant_mod
    elif type(model) == nn.Linear:
        quant_mod = LinearQuantizer(**quantizer_args)
        quant_mod.set_param(model)
        del model
        return quant_mod
    elif type(model) == pytorch_utils.Conv1D:
        quant_mod = Conv1dQuantizer(**quant_args)
        quant_mod.set_param(model)
        del model
        return quant_mod
    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, quantizer_args))
        return nn.Sequential(*mods)
    elif type(model) == nn.ModuleList:
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, quantizer_args))
        return nn.Sequential(*mods)
    elif isinstance(model, nn.Sequential):
        mods = []
        for n, m in model.named_children():
            mods.append(quantize_model(m, quantizer_args))
        return nn.Sequential(*mods)
    else:
        # recursively use the quantized module to replace the single-precision module
        # q_model = copy.deepcopy(model)
        q_model = model
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and attr != 'base_model' and attr!= 'lm_head':
                setattr(q_model, attr, quantize_model(mod, quantizer_args))
        return q_model