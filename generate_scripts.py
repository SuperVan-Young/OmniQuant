# Generate experiment scripts

import os
import math
import argparse
from copy import deepcopy

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--server", type=str, choices=['V100', 'A6000', 'A100'], default='V100', help="server type")
parser.add_argument("--model_list", choices=['demo', 'small', 'mediam', 'large', 'all', 'opt_all', 'llama_all'], default='demo', type=str, help="model list")
parser.add_argument("--no-large", action='store_true', help="exclude large models")
parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token", 'none'])

args = parser.parse_args()

EXPERIMENT_SERVER = args.server

if EXPERIMENT_SERVER == "V100":
    GPU_LIST = [f"{i}" for i in range(2,8)]
    GPU_MEMORY = 32
    MODEL_DIR = "/home/xuechenhao/hugginface"
elif EXPERIMENT_SERVER == "A6000":
    GPU_LIST = [f"{i}" for i in range(4)]
    GPU_MEMORY = 48
    MODEL_DIR = "/home/zhangchen/hugginface"
elif EXPERIMENT_SERVER == "A100":
    GPU_LIST = [f"{i}" for i in range(2)]
    GPU_MEMORY = 80
    MODEL_DIR = "/home/xuechenhao/hugginface"
else:
    raise NotImplementedError

DEMO_MODEL_LIST = [
    # Now that we don't modify TransformerLayer, we can use one type of CasualLM
    "opt-6.7b",
    "llama-7b-meta",
]

SMALL_MODEL_LIST = [
    "llama-7b-meta",
    "opt-6.7b",
    "llama-13b-meta",
    "opt-13b",
]

MEDIAM_MODEL_LIST = [
    "opt-30b",
    "llama-30b-meta",
]

LARGE_MODEL_LIST = [
    "opt-66b",
    "llama-65b-meta",
]

ALL_MODEL_LIST = [
    # TODO: small model have groupsize problem
    # "opt-1.3b",
    # "opt-2.7b",
    "opt-6.7b",
    "llama-7b-meta",
    "opt-13b",
    "llama-13b-meta",
    "opt-30b",
    "llama-30b-meta",
    "opt-66b",
    "llama-65b-meta",
]

OPT_ALL_MODEL_LIST = [
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
]

LLAMA_ALL_MODEL_LIST = [
    "llama-7b-meta",
    "llama-13b-meta",
    "llama-30b-meta",
    "llama-65b-meta",
]

if args.model_list == 'demo':
    MODEL_LIST = DEMO_MODEL_LIST
elif args.model_list == 'small':
    MODEL_LIST = SMALL_MODEL_LIST
elif args.model_list == 'mediam':
    MODEL_LIST = MEDIAM_MODEL_LIST
elif args.model_list == 'large':
    MODEL_LIST = LARGE_MODEL_LIST
elif args.model_list == 'all':
    MODEL_LIST = ALL_MODEL_LIST
elif args.model_list == 'opt_all':
    MODEL_LIST = OPT_ALL_MODEL_LIST
elif args.model_list == 'llama_all':
    MODEL_LIST = LLAMA_ALL_MODEL_LIST
else:
    raise NotImplementedError

if args.no_large:
    model_size = {model: float(model.split('-')[1].replace('b', '')) for model in MODEL_LIST}
    for model in MODEL_LIST:
        if model_size[model] > 60:
            MODEL_LIST.remove(model)
    print(MODEL_LIST)

CONFIG_DICT = {
    ###############################################
    # Baseline Experiments
    ###############################################

    # # FULL PRECISION
    # "W16A16": {
    #     "wbits": 16,
    #     "abits": 16,
    # },

    # # qkvproj W16A4 & W16A8
    # "qkvproj_W16A4": {
    #     "wbits": 16,
    #     "abits": 4,
    #     "aow-quant-act-qkvproj": None,
    # },
    # "qkvproj_W16A8": {
    #     "wbits": 16,
    #     "abits": 8,
    #     "aow-quant-act-qkvproj": None,
    # },


    # # fc1 W16A4 & W16A8
    # "fc1_W16A4": {
    #     "wbits": 16,
    #     "abits": 4,
    #     "aow-quant-act-fc1": None,
    # },
    # "fc1_W16A8": {
    #     "wbits": 16,
    #     "abits": 8,
    #     "aow-quant-act-fc1": None,
    # },

    # # oproj W16A4 & W16A8
    # "oproj_W16A4": {
    #     "wbits": 16,
    #     "abits": 4,
    #     "aow-quant-act-oproj": None,
    # },
    # "oproj_W16A8": {
    #     "wbits": 16,
    #     "abits": 8,
    #     "aow-quant-act-oproj": None,
    # },

    # # fc2 W16A4 & W16A8
    # "fc2_W16A4": {
    #     "wbits": 16,
    #     "abits": 4,
    #     "aow-quant-act-fc2": None,
    # },
    # "fc2_W16A8": {
    #     "wbits": 16,
    #     "abits": 8,
    #     "aow-quant-act-fc2": None,
    # },

    # q W16A4 & W16A8
    "q_W16A4": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-q": None,
    },
    "q_W16A8": {
        "wbits": 16,
        "abits": 8,
        "aow-quant-act-q": None,
    },

    # k W16A4 & W16A8
    "k_W16A4": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-k": None,
    },
    "k_W16A8": {
        "wbits": 16,
        "abits": 8,
        "aow-quant-act-k": None,
    },

    # v W16A4 & W16A8
    "v_W16A4": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-v": None,
    },
    "v_W16A8": {
        "wbits": 16,
        "abits": 8,
        "aow-quant-act-v": None,
    },

    ###############################################
    # Modifications
    ###############################################

    # qkvproj: outlier
    "qkvproj_W16A4_ol1p128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-qkvproj": None,
        "act-outlier-ratio": 0.0078125,  # 1 / 128
    },

    # fc1: outlier
    "fc1_W16A4_ol1p128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc1": None,
        "act-outlier-ratio": 0.0078125,  # 1 / 128
    },

    # fc2: reorder + groupwise + outlier

    # groupwise
    "fc2_W16A4_g128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc2": None,
        "act-group-size": 128,
    },

    # outlier
    "fc2_W16A4_ol1p128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc2": None,
        "act-outlier-ratio": 0.0078125,  # 1 / 128
    },

    # groupwise + outlier
    "fc2_W16A4_g128_ol1p128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc2": None,
        "act-group-size": 128,
        "act-outlier-ratio": 0.0078125,  # 1 / 128
    },

    # reorder + groupwise
    "fc2_W16A4_g128_r": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc2": None,
        "act-group-size": 128,
        "act-reorder": None,
    },

    # reorder + groupwise + outlier
    "fc2_W16A4_g128_ol1p128_r": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc2": None,
        "act-group-size": 128,
        "act-outlier-ratio": 0.0078125,  # 1 / 128
        "act-reorder": None,
    },

    # oproj: groupwise + outlier

    # groupwise
    "oproj_W16A4_g128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-oproj": None,
        "act-group-size": 128,
    },

    # outlier
    "oproj_W16A4_ol1p128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-oproj": None,
        "act-outlier-ratio": 0.0078125,  # 1 / 128
    },

    # groupwise + outlier
    "oproj_W16A4_g128_ol1p128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-oproj": None,
        "act-group-size": 128,
        "act-outlier-ratio": 0.0078125,  # 1 / 128
    },
    
}

def gen_single_experiment_script(
    model_name,
    available_gpus,
    extra_configs: dict = {},
    output_subdir = None,
) -> str:
    """
    Generate script for running a single experiment on one model.
    """
    assert "wbits" in extra_configs
    assert "abits" in extra_configs
    
    scripts = ""

    if output_subdir is None:
        output_dir = f"$OUTPUT_DIR/{model_name}"
    else:
        output_dir = f"$OUTPUT_DIR/{output_subdir}"
        scripts += f"mkdir -p {output_dir}"

    scripts += f"""
CUDA_VISIBLE_DEVICES=\"{available_gpus}\" python main.py \\
--eval_ppl --epoch 0 --quant-method aowquant \\
--model $MODEL_DIR/{model_name} \\
--output_dir {output_dir} \\
"""
    if len(available_gpus.split(",")) > 1:
        scripts += "--multigpu \\\n"
    if args.a_dynamic_method == "none":
        scripts += "--a_dynamic_method none \\\n"
    for config_name, config_val in extra_configs.items():
        if config_val is None:
            scripts += f"--{config_name} \\\n"
        elif isinstance(config_val, (list, tuple)):
            scripts += f"--{config_name} {' '.join([str(val) for val in config_val])} \\\n"
        else:
            scripts += f"--{config_name} {config_val} \\\n"
    scripts += "&\n"
    return scripts

def allocate_gpu(model_list) -> dict:
    """
    Allocate GPU for each model.
    """

    def parse_model_size(model_name):
        if "opt" in model_name:
            return float(model_name.split("-")[1].replace("b", ""))
        elif "llama" in model_name:
            return float(model_name.split("-")[1].replace("b", ""))
        else:
            raise NotImplementedError
        
    def _allocate_gpu(model_name):
        weight_size_in_GB = parse_model_size(model_name) * 2.5  # spare some room for runtime memory
        num_gpu = math.ceil(weight_size_in_GB / GPU_MEMORY)
        assert num_gpu <= len(GPU_LIST), f"Model {model_name} requires {num_gpu} GPUs, but only {len(GPU_LIST)} GPUs are available."
        return num_gpu

    return {model: _allocate_gpu(model) for model in model_list}

def pack_experiments(
    model_list,
    model_dir,
    output_dir,
    script_path,
    extra_configs,
):
    """
    Pack experiments on different models into a single script.
    """

    # Allocate GPU for each model
    model_gpu_dict = allocate_gpu(model_list)

    # Generate scripts
    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n\n")

        f.write("# This script is generated by generate_scripts.py.\n")
        f.write("# This script is exptected to be run on server %s.\n\n" % EXPERIMENT_SERVER)

        f.write(f"MODEL_DIR={model_dir}\n")
        f.write(f"OUTPUT_DIR={output_dir}\n")

        used_gpu = 0

        for model_name in model_list:
            num_gpus = model_gpu_dict[model_name]
            if used_gpu + num_gpus > len(GPU_LIST):
                used_gpu = 0
                f.write("wait\n\n")
            available_gpus = ",".join(GPU_LIST[used_gpu:used_gpu+num_gpus])
            f.write(gen_single_experiment_script(
                model_name,
                available_gpus,
                extra_configs,
            ))
            # prevent OOM by manually wait
            used_gpu += model_gpu_dict[model_name]

        f.write("wait\n\n")

def gen_all_scripts(
    model_list=DEMO_MODEL_LIST,
):
    """
    Generate scripts for running all experiments.
    """
    script_dir = f"./scripts/aow_{EXPERIMENT_SERVER}"
    os.makedirs(script_dir, exist_ok=True)

    for config_name, config_val in CONFIG_DICT.items():
        pack_experiments(
            model_list=model_list,
            model_dir=MODEL_DIR,
            output_dir=f"./output/{config_name}",
            script_path=f"{script_dir}/{config_name}.sh",
            extra_configs=config_val,
        )
    
    # Generate an entry script
    with open(os.path.join(script_dir, 'run_all.sh'), "w") as f:
        f.write("#!/bin/bash\n\n")
        for config_name in CONFIG_DICT.keys():
            f.write(f"bash {script_dir}/{config_name}.sh\n")

    # debugging scripts and entry
    for model in DEMO_MODEL_LIST:
        gen_debugging_script(model)

    with open(os.path.join(script_dir, 'run_debugging.sh'), "w") as f:
        f.write("#!/bin/bash\n\n")
        for model in DEMO_MODEL_LIST:
            f.write(f"bash {script_dir}/debugging_{model}.sh\n")

def gen_debugging_script(
    model_name = 'opt-6.7b',
):
    """
    Generate a model script to run all experiments for debugging
    """
    script_dir = f"./scripts/aow_{EXPERIMENT_SERVER}"
    os.makedirs(script_dir, exist_ok=True)

    # allocate GPU for the model
    num_gpus = allocate_gpu([model_name])[model_name]

    # Iterate through experiment configurations and generate scripts
    with open(os.path.join(script_dir, f'debugging_{model_name}.sh'), "w") as f:
        f.write("#!/bin/bash\n\n")

        f.write("# This script is generated by generate_scripts.py.\n")
        f.write("# This script is exptected to be run on server %s.\n\n" % EXPERIMENT_SERVER)

        f.write(f"MODEL_DIR={MODEL_DIR}\n")

        if args.a_dynamic_method == 'none':
            f.write(f"OUTPUT_DIR=./output_static_demo\n\n")
        else:
            f.write(f"OUTPUT_DIR=./output_demo\n\n")

        used_gpu = 0

        for config_name, configs in CONFIG_DICT.items():
            # if 'W16A8' in config_name or 'W16A16' in config_name:
            #     continue
            # if len(config_name.split('_')) <= 2:
            #     continue  # baseline experiments

            # use few eval dataset for debugging
            configs = deepcopy(configs)
            configs['eval-ppl-dataset'] = 'wikitext2'
            # configs['debug'] = None  # only quantize one layer for debugging

            if used_gpu + num_gpus > len(GPU_LIST):
                used_gpu = 0
                f.write("wait\n\n")
            available_gpus = ",".join(GPU_LIST[used_gpu:used_gpu+num_gpus])
            f.write(f"\n# {config_name}\n")
            f.write(gen_single_experiment_script(
                model_name,
                available_gpus,
                configs,
                output_subdir=f"{config_name}/{model_name}"
            ))
            # prevent OOM by manually wait
            used_gpu += num_gpus

        f.write("wait\n\n")

    
if __name__ == "__main__":
    # gen_all_scripts(SMALL_MODEL_LIST + MEDIAM_MODEL_LIST)
    gen_all_scripts(MODEL_LIST)