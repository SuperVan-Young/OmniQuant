import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
import argparse
import torch.nn as nn

from datasets import load_dataset
import functools
from tqdm import tqdm
from datautils import get_loaders
# try:
#     from llava.model import *   # required for llava
# except ImportError:
#     print("If want to quantize llave models, you should manually install llava from https://github.com/haotian-liu/LLaVA")

# import pdb
from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from models.int_falcon_layer import QuantFalconDecoderLayer
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.omni_norm import OmniLayerNorm, OmniLlamaRMSNorm
from parallel_utils import add_forward_hooks

def get_act_scales(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_scales

def get_act_shifts(model, dataloader, num_samples=128):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        comming_min = torch.min(tensor, dim=0)[0].float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *((comming_max+comming_min)/2)
        else:
            act_shifts[name] = (comming_max+comming_min)/2

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))


    for h in hooks:
        h.remove()

    return act_shifts

def wrap_up_model(model, args):
    if 'llama' in args.net.lower():
        layers = model.model.layers
        DecoderLayer = QuantLlamaDecoderLayer
    elif 'opt' in args.net.lower():
        layers = model.model.decoder.layers
        DecoderLayer = QuantOPTDecoderLayer
    elif 'falcon' in args.net.lower():
        layers = model.transformer.h
        DecoderLayer = QuantFalconDecoderLayer
    else:
        raise NotImplementedError

    # to deploy layers across all gpus
    layer_gpu_map = {}

    for i in range(len(layers)):
        layer = layers[i]
        layer_device = next(layer.parameters()).device.index
        qlayer = DecoderLayer(model.config, layer, args)
        qlayer.set_quant_state(weight_quant=False, act_quant=False)
        layers[i] = qlayer
        layer_gpu_map[qlayer] = layer_device
        del layer

    add_forward_hooks(layer_gpu_map)

    return model


def get_act_stats(model, dataloader, num_samples=128):
    """
    Profile all stats for all layers in the model.
    Layers include linear, layernorm, matmul.
    Stats contain: min, max, absmin, absmax, mean, absmean
    Both input and output activation are profiled.
    """
    model.eval()
    device = next(model.parameters()).device
    act_stats = {}  # layer name -> category -> stat list
    # category: input, abs_input, output, abs_output
    # stat list: min, max, mean, std

    def get_tensor_stat(tensor) -> dict:
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        stats = {}
        stats['min'] = torch.min(tensor, dim=0)[0].float().cpu()
        stats['max'] = torch.max(tensor, dim=0)[0].float().cpu()
        stats['mean'] = torch.mean(tensor, dim=0).float().cpu()
        stats['std'] = torch.std(tensor, dim=0).float().cpu()
        return stats

    def update_stats(name, category, comming_stats):
        act_stats.setdefault(name, {})
        act_stats[name].setdefault(category, {
            'min': None,
            'max': None,
            'mean': None,
            'std': None,
        })
        stats = act_stats[name][category]
        
        # update min
        if stats['min'] is None:
            stats['min'] = comming_stats['min']
        else:
            stats['min'] = torch.min(stats['min'], comming_stats['min'])

        # update max similar to min
        if stats['max'] is None:
            stats['max'] = comming_stats['max']
        else:
            stats['max'] = torch.max(stats['max'], comming_stats['max'])
        
        # running mean
        if stats['mean'] is None:
            stats['mean'] = comming_stats['mean']
        else:
            stats['mean'] = 0.99*stats['mean'] + 0.01*comming_stats['mean']

        # running std
        if stats['std'] is None:
            stats['std'] = comming_stats['std']
        else:
            stats['std'] = 0.99*stats['std'] + 0.01*comming_stats['std']

    def stat_linear_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        input_stats = get_tensor_stat(x)
        abs_input_stats = get_tensor_stat(x.abs())
        update_stats(name, 'input', input_stats)
        update_stats(name, 'abs_input', abs_input_stats)

        output_stats = get_tensor_stat(y)
        abs_output_stats = get_tensor_stat(y.abs())
        update_stats(name, 'output', output_stats)
        update_stats(name, 'abs_output', abs_output_stats)

    def stat_qktmatmul_hook(m, x, y, name):
        if isinstance(x, tuple):
            q = x[0]
            k = x[1]
            bsz, num_heads, q_len, head_dim = q.shape
            q = q.transpose(1, 2).reshape(-1, num_heads * head_dim)
            k = k.transpose(2, 3).transpose(1, 2).reshape(-1, num_heads * head_dim)
        q_stats = get_tensor_stat(q)
        abs_q_stats = get_tensor_stat(q.abs())
        update_stats(name, 'q', q_stats)
        update_stats(name, 'abs_q', abs_q_stats)

        k_stats = get_tensor_stat(k)
        abs_k_stats = get_tensor_stat(k.abs())
        update_stats(name, 'k', k_stats)
        update_stats(name, 'abs_k', abs_k_stats)

    def stat_pvmatmul_hook(m, x, y, name):
        if isinstance(x, tuple):
            v = x[1]
            bsz, num_heads, q_len, head_dim = v.shape
            v = v.transpose(1, 2).reshape(-1, num_heads * head_dim)

        v_stats = get_tensor_stat(v)
        abs_v_stats = get_tensor_stat(v.abs())
        update_stats(name, 'v', v_stats)
        update_stats(name, 'abs_v', abs_v_stats)
    
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (QuantLinear, OmniLlamaRMSNorm, OmniLayerNorm)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_linear_hook, name=name))
            )
            print(f"Register hook for {name} ({type(m)})")
        elif isinstance(m, QuantMatMul):
            if "qkt" in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_qktmatmul_hook, name=name))
                )
                print(f"Register hook for {name} ({type(m)})")
            elif "pv" in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_pvmatmul_hook, name=name))
                )
                print(f"Register hook for {name} ({type(m)})")

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_stats


def get_outlier_stats(model, dataloader, num_samples=128, threshold=3):
    """
    Profile outlier stats for all layers in the model.
    First collect tensor-level mean and std, use 3 sigma rule to identify outliers
    Then collect channel-level outlier status, including:
    - outlier ratio
    - outlier min / max
    - outlier L2 norm
    - channel L2 norm (just like Wanda's metric for pruning)
    """

    model.eval()
    device = next(model.parameters()).device
    outlier_stats = {}  # layer name -> category -> stat list
    # category: input, output

    # first round: get tensor mean and std
    def get_tensor_stat(tensor) -> dict:
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        stats = {}
        stats['tensor_min'] = torch.min(tensor).float().cpu()
        stats['tensor_max'] = torch.max(tensor).float().cpu()
        stats['tensor_mean'] = torch.mean(tensor).float().cpu()
        stats['tensor_std'] = torch.std(tensor).float().cpu()
        return stats
    
    def update_tensor_stats(name, category, comming_stats):
        outlier_stats.setdefault(name, {})
        outlier_stats[name].setdefault(category, {
            # whole tensor
            'tensor_mean': [],
            'tensor_std': [],
            'tensor_min': [],
            'tensor_max': [],

            # outlier stats per channel
            'outlier_ratio': [],
            'outlier_min': [],
            'outlier_max': [],
            'outlier_l1_norm': [],
            'outlier_l2_norm': [],

            # all l2 norm per channel
            'all_l2_norm': [],

            # quantization mse per tensor
            'normal_mse_int4': [],
            'normal_mse_int8': [],
            'outlier_mse_int4': [],
            'outlier_mse_int8': [],
            'outlier_mse_e1m2': [],
            'outlier_mse_e2m1': [],
        })
        stats = outlier_stats[name][category]
        
        stats['tensor_min'].append(comming_stats['tensor_min'])
        stats['tensor_max'].append(comming_stats['tensor_max'])
        stats['tensor_mean'].append(comming_stats['tensor_mean'])
        stats['tensor_std'].append(comming_stats['tensor_std'])

    def stat_linear_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        input_stats = get_tensor_stat(x)
        update_tensor_stats(name, 'input', input_stats)

        output_stats = get_tensor_stat(y)
        update_tensor_stats(name, 'output', output_stats)

    def stat_qktmatmul_hook(m, x, y, name):
        if isinstance(x, tuple):
            q = x[0]
            k = x[1]
            bsz, num_heads, q_len, head_dim = q.shape
            q = q.transpose(1, 2).reshape(-1, num_heads * head_dim)
            k = k.transpose(2, 3).transpose(1, 2).reshape(-1, num_heads * head_dim)
        q_stats = get_tensor_stat(q)
        update_tensor_stats(name, 'q', q_stats)

        k_stats = get_tensor_stat(k)
        update_tensor_stats(name, 'k', k_stats)

    def stat_pvmatmul_hook(m, x, y, name):
        if isinstance(x, tuple):
            v = x[1]
            bsz, num_heads, q_len, head_dim = v.shape
            v = v.transpose(1, 2).reshape(-1, num_heads * head_dim)

        v_stats = get_tensor_stat(v)
        update_tensor_stats(name, 'v', v_stats)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (QuantLinear, OmniLlamaRMSNorm, OmniLayerNorm)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_linear_hook, name=name))
            )
            print(f"Register hook for {name} ({type(m)})")
        elif isinstance(m, QuantMatMul):
            if "qkt" in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_qktmatmul_hook, name=name))
                )
                print(f"Register hook for {name} ({type(m)})")
            elif "pv" in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_pvmatmul_hook, name=name))
                )
                print(f"Register hook for {name} ({type(m)})")

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    # get average mean and std
    for name, stats in outlier_stats.items():
        for category, stat_list in stats.items():
            stat_list['tensor_min'] = torch.stack(stat_list['tensor_min']).min().item()
            stat_list['tensor_max'] = torch.stack(stat_list['tensor_max']).max().item()
            stat_list['tensor_mean'] = torch.stack(stat_list['tensor_mean']).mean().item()
            stat_list['tensor_std'] = torch.stack(stat_list['tensor_std']).mean().item()

    # second round: get outlier stats

    # quantize outliers with different datatypes
    # Here we use unified scaling factor for normal and outlier values
    # to see the required precision for outliers
    def fakequant_tensor(x, xabsmax, quant_grid: torch.tensor):
        # assume symmetric quantization with integer encoding
        if isinstance(quant_grid, str):
            assert 'int' in quant_grid
            n_bits = int(quant_grid[3:])
            scale = xabsmax / (2 ** (n_bits - 1) - 1)
            qmax = 2 ** (n_bits - 1) - 1
            qmin = -qmax - 1
            x_q = torch.round(x / scale)
            x_dequant = x_q * scale
        else:
            quant_grid = quant_grid.to(x.device)
            max_quant_val = max(quant_grid)
            scale = xabsmax / max_quant_val

            x_q = x / scale
            diff = torch.abs(x_q.unsqueeze(-1) - quant_grid.unsqueeze(0))
            index = diff.argmin(dim=-1)
            x_q = quant_grid[index]

            x_dequant = x_q * scale

        mse = ((x - x_dequant) ** 2).mean().cpu()
        # check if mse is nan
        if torch.isnan(mse):
            mse = torch.tensor(0.0)

        return x_dequant, mse
    
    def get_float_grid(exp_bit = 2, mant_bit = 1, signed = True):
        values = [0]

        for e in range(2 ** exp_bit):
            for m in range(2 ** mant_bit):
                if e == 0 and m == 0:
                    continue
                v = 2 ** e * (2 **mant_bit + m)
                values.append(v)
                if signed:
                    values.append(-v)
        values = torch.tensor(values)
        values, _ = torch.sort(values)
        return values


    def get_outlier_stat(tensor, tensor_stat, threshold=3) -> dict:
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).float().detach()

        tensor_mean = tensor_stat['tensor_mean']
        tensor_std = tensor_stat['tensor_std']
        tensor_min = tensor_stat['tensor_min']
        tensor_max = tensor_stat['tensor_max']

        outlier_mask = ((tensor - tensor_mean) / tensor_std).abs() > threshold
        num_outlier_per_channel = outlier_mask.sum(dim=0).float()

        stats = {}
        stats['outlier_ratio'] = outlier_mask.float().mean(dim=0).cpu()
        stats['outlier_min'] = torch.min(tensor * outlier_mask, dim=0)[0].cpu()
        stats['outlier_max'] = torch.max(tensor * outlier_mask, dim=0)[0].cpu()
        stats['all_l2_norm'] = torch.mean(tensor ** 2, dim=0).cpu()

        # compute norm of non-zero elements for each channel
        square_sum = tensor.masked_select(outlier_mask).pow(2).sum(dim=0)
        stats['outlier_l2_norm'] = square_sum / num_outlier_per_channel
        stats['outlier_l2_norm'][num_outlier_per_channel == 0] = 0  # ensure not to divide by zero
        stats['outlier_l2_norm'] = stats['outlier_l2_norm'].cpu()

        abs_sum = tensor.masked_select(outlier_mask).abs().sum(dim=0)
        stats['outlier_l1_norm'] = abs_sum / num_outlier_per_channel
        stats['outlier_l1_norm'][num_outlier_per_channel == 0] = 0
        stats['outlier_l1_norm'] = stats['outlier_l1_norm'].cpu()

        # calculate MSE for quantization
        x_normal = tensor[~outlier_mask]
        x_outlier = tensor[outlier_mask]
        x_normal_absmax = (torch.tensor([-1, 1]) * threshold * tensor_std + tensor_mean).abs().max()
        x_outlier_absmax = torch.tensor([tensor_min, tensor_max]).abs().max()

        quant_grid_e1m2 = get_float_grid(1, 2, signed=True)
        quant_grid_e2m1 = get_float_grid(2, 1, signed=True)

        # only quantize normal values, use range between threshold
        _, stats[f'normal_mse_int4'] = fakequant_tensor(x_normal, x_normal_absmax, quant_grid='int4')
        _, stats[f'normal_mse_int8'] = fakequant_tensor(x_normal, x_normal_absmax, quant_grid='int8')

        # only quantize outlier values, use full range
        _, stats[f'outlier_mse_int4'] = fakequant_tensor(x_outlier, x_outlier_absmax, quant_grid='int4')
        _, stats[f'outlier_mse_int8'] = fakequant_tensor(x_outlier, x_outlier_absmax, quant_grid='int8')
        _, stats[f'outlier_mse_e1m2'] = fakequant_tensor(x_outlier, x_outlier_absmax, quant_grid=quant_grid_e1m2)
        _, stats[f'outlier_mse_e2m1'] = fakequant_tensor(x_outlier, x_outlier_absmax, quant_grid=quant_grid_e2m1)

        return stats
    
    def update_outlier_stats(name, category, comming_stats):
        stats = outlier_stats[name][category]
        for k, v in comming_stats.items():
            stats[k].append(v)

    def stat_outlier_linear_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        input_stats = get_outlier_stat(x, outlier_stats[name]['input'], threshold=threshold)
        update_outlier_stats(name, 'input', input_stats)

        output_stats = get_outlier_stat(y, outlier_stats[name]['output'], threshold=threshold)
        update_outlier_stats(name, 'output', output_stats)

    def stat_outlier_qktmatmul_hook(m, x, y, name):
        if isinstance(x, tuple):
            q = x[0]
            k = x[1]
            bsz, num_heads, q_len, head_dim = q.shape
            q = q.transpose(1, 2).reshape(-1, num_heads * head_dim)
            k = k.transpose(2, 3).transpose(1, 2).reshape(-1, num_heads * head_dim)
        q_stats = get_outlier_stat(q, outlier_stats[name]['q'], threshold=threshold)
        update_outlier_stats(name, 'q', q_stats)

        k_stats = get_outlier_stat(k, outlier_stats[name]['k'], threshold=threshold)
        update_outlier_stats(name, 'k', k_stats)

    def stat_outlier_pvmatmul_hook(m, x, y, name):
        if isinstance(x, tuple):
            v = x[1]
            bsz, num_heads, q_len, head_dim = v.shape
            v = v.transpose(1, 2).reshape(-1, num_heads * head_dim)

        v_stats = get_outlier_stat(v, outlier_stats[name]['v'], threshold=threshold)
        update_outlier_stats(name, 'v', v_stats)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (QuantLinear, OmniLlamaRMSNorm, OmniLayerNorm)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_outlier_linear_hook, name=name))
            )
            print(f"Register hook for {name} ({type(m)})")
        elif isinstance(m, QuantMatMul):
            if "qkt" in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_outlier_qktmatmul_hook, name=name))
                )
                print(f"Register hook for {name} ({type(m)})")
            elif "pv" in name:
                hooks.append(
                    m.register_forward_hook(
                        functools.partial(stat_outlier_pvmatmul_hook, name=name))
                )
                print(f"Register hook for {name} ({type(m)})")

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    # get average outlier stats
    for name, stats in outlier_stats.items():
        for category, stat_list in stats.items():
            stat_list['outlier_min'] = torch.stack(stat_list['outlier_min']).min(dim=0)
            stat_list['outlier_max'] = torch.stack(stat_list['outlier_max']).max(dim=0)
            stat_list['all_l2_norm'] = torch.stack(stat_list['all_l2_norm']).mean(dim=0)
            stat_list['outlier_l1_norm'] = torch.stack(stat_list['outlier_l1_norm']).mean(dim=0)
            stat_list['outlier_l2_norm'] = torch.stack(stat_list['outlier_l2_norm']).mean(dim=0)
            stat_list['outlier_ratio'] = torch.stack(stat_list['outlier_ratio']).mean(dim=0)

            for mse_name, mse_list in stat_list.items():
                if 'mse' not in mse_name: continue
                stat_list[mse_name] = torch.stack(mse_list).mean()

    return outlier_stats


def build_model_and_tokenizer(model_name):
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/home/zhangchen/hugginface/llama-7b-hf-transformers-4.29', help='model name')
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',
                        help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/',
                        help='where to save the act shifts')
    parser.add_argument("--calib_dataset",type=str,default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix","pile"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=128)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--act-stats-output-path", type=str, default="./act_stats/", 
                        help="Where to save all act stats")
    parser.add_argument("--profile-all-stats", action="store_true", default=False, help="Profile all stats")
    parser.add_argument("--outlier-stats-output-path", type=str, default="./outlier_stats/",)
    parser.add_argument("--profile-outlier-stats", action="store_true", default=False, help="Profile outlier stats")
    parser.add_argument("--outlier-threshold", type=int, default=3, help="threshold for outlier detection")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model)
    dataloader, _ = get_loaders(
    args.calib_dataset,
    nsamples=args.num_samples,
    seed=args.seed,
    model=args.model,
    seqlen=args.seq_len,
    )
    
    args.net = args.model.split('/')[-1]
    print(f"Profiling model: {args.net}")

    if args.profile_all_stats:
        args.weight_quant_params = {}
        args.act_quant_params = {}
        args.q_quant_params = {}
        args.k_quant_params = {}
        args.p_quant_params = {}
        args.v_quant_params = {}
        wrap_up_model(model, args)
        act_stats = get_act_stats(model, dataloader, args.num_samples)
        save_path = os.path.join(args.act_stats_output_path,f'{args.net}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(act_stats, save_path)
    if args.profile_outlier_stats:
        args.weight_quant_params = {}
        args.act_quant_params = {}
        args.q_quant_params = {}
        args.k_quant_params = {}
        args.p_quant_params = {}
        args.v_quant_params = {}
        wrap_up_model(model, args)
        outlier_stats = get_outlier_stats(model, dataloader, args.num_samples, threshold=args.outlier_threshold)
        save_path = os.path.join(args.outlier_stats_output_path,f'{args.net}_outlier.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(outlier_stats, save_path)
    if not args.profile_all_stats and not args.profile_outlier_stats:
        act_scales = get_act_scales(model, dataloader,args.num_samples)
        save_path = os.path.join(args.scales_output_path,f'{args.net}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(act_scales, save_path)

        act_shifts = get_act_shifts(model, dataloader,args.num_samples)
        save_path = os.path.join(args.shifts_output_path,f'{args.net}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(act_shifts, save_path)

if __name__ == '__main__':
    main()
