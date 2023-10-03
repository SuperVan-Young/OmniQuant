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

    def get_tensor_stat(tensor) -> tuple:
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        stats = {}
        stats['min'] = torch.min(tensor, dim=0)[0].float().cpu()
        stats['max'] = torch.max(tensor, dim=0)[0].float().cpu()
        stats['mean'] = torch.mean(tensor, dim=0).float().cpu()
        stats['std'] = torch.std(tensor, dim=0).float().cpu()
        return stats

    def update_min_max(name, category, comming_stats):
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
        update_min_max(name, 'input', input_stats)
        update_min_max(name, 'abs_input', abs_input_stats)

        output_stats = get_tensor_stat(y)
        abs_output_stats = get_tensor_stat(y.abs())
        update_min_max(name, 'output', output_stats)
        update_min_max(name, 'abs_output', abs_output_stats)
    
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_linear_hook, name=name))
            )

    for i in tqdm(range(num_samples)):
        model(dataloader[i][0].to(device))

    for h in hooks:
        h.remove()

    return act_stats



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
    if args.profile_all_stats:
        act_stats = get_act_stats(model, dataloader, args.num_samples)
        save_path = os.path.join(args.act_stats_output_path,f'{args.net}.pt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(act_stats, save_path)
    else:
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
