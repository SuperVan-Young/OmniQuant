# Generate experiment scripts

import os
import math
import argparse
from copy import deepcopy
from itertools import product

from experiments.configs import *
from experiments.scripts import get_multi_model_script, get_multi_config_script
from experiments.server import get_server_config
from experiments.model import get_model_list

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--server", type=str, choices=['V100', 'A6000', 'A100'], default='V100', help="server type")
parser.add_argument("--model_list", choices=['tiny', 'small', 'mediam', 'large', 'all', 'opt_all', 'llama_all'], default='tiny', type=str, help="model list")
parser.add_argument("--a_dynamic_method", type=str, default="none", choices=["per_token", 'none'])

args = parser.parse_args()

model_list = get_model_list(args.model_list)
extra_experiment_configs = {
    "a_dynamic_method": args.a_dynamic_method,
}

def write_script(script, script_path):
    with open(script_path, "w") as f:
        f.write(script)

def main():
    write_script(get_multi_model_script(
        model_name_list=get_model_list('tiny'),
        server_config=get_server_config(args.server),
        experiment_config_dict=get_efficient_grouping_experiment_configs(**extra_experiment_configs),
        top_output_dir='./output/efficient_grouping/'), 
        './scripts/demo/efficient_grouping.sh')
    
    write_script(get_multi_model_script(
        model_name_list=get_model_list('tiny'),
        server_config=get_server_config(args.server),
        experiment_config_dict=get_unified_postlayernorm_outlier_experiment_configs(**extra_experiment_configs),
        top_output_dir='./output/unified_postlayernorm_outlier/'), 
        './scripts/demo/unified_postlayernorm_outlier.sh')
    
    fc2_tuning_configs = get_outlier_experiment_configs(
        act_reorder=None,
        act_group_size=128,
        **extra_experiment_configs)
    fc2_tuning_configs = {k: v for k, v in fc2_tuning_configs.items() if "aow_quant_act_fc2" in v.keys()}
    write_script(get_multi_model_script(
        model_name_list=['llama-7b-meta'],
        server_config=get_server_config(args.server),
        experiment_config_dict=fc2_tuning_configs,
        top_output_dir='./output/llama_fc2_tuning/'), 
        './scripts/demo/llama_fc2_tuning.sh')
    
    write_script(get_multi_model_script(
        model_name_list=get_model_list('tiny'),
        server_config=get_server_config(args.server),
        experiment_config_dict=get_outlier_bits_experiment_configs(**extra_experiment_configs),
        top_output_dir='./output/outlier_bits/'), 
        './scripts/demo/outlier_bits.sh')
    
    write_script(get_multi_model_script(
        model_name_list=get_model_list('tiny'),
        server_config=get_server_config(args.server),
        experiment_config_dict=get_full_model_static_experiment_configs(),
        top_output_dir='./output/full_model_static/'), 
        './scripts/demo/full_model_static.sh')
    
    # final experiments
    os.makedirs('./scripts/final', exist_ok=True)
    for model_list_type in ['tiny', 'small', 'medium', 'large']:
        write_script(get_multi_model_script(
            model_name_list=get_model_list(model_list_type),
            server_config=get_server_config(args.server),
            experiment_config_dict=get_full_model_experiment_configs(),
            top_output_dir='./output/full_model/'), 
            f'./scripts/final/{model_list_type}.sh')
    

if __name__ == "__main__":
    main()