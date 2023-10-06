# Generate experiment scripts

import os
import math

os.makedirs("./scripts/aow", exist_ok=True)

EXPERIMENT_SERVER = "V100"

if EXPERIMENT_SERVER == "V100":
    GPU_LIST = [f"{i}" for i in range(8)]
    GPU_MEMORY = 32
    MODEL_DIR = "/home/xuechenhao/hugginface"
elif EXPERIMENT_SERVER == "A6000":
    GPU_LIST = [f"{i}" for i in range(4)]
    GPU_MEMORY = 48
    MODEL_DIR = "/home/zhangchen/hugginface"
else:
    raise NotImplementedError

DEMO_MODEL_LIST = [
    "llama-7b-hf-transformers-4.29",
    "opt-6.7b",
]

SMALL_MODEL_LIST = [
    "llama-7b-hf-transformers-4.29",
    "opt-6.7b",
    "llama-13b-hf-transformers-4.29",
    "opt-13b",
]

LARGE_MODEL_LIST = [
    "opt-30b",
    "llama-30b-hf-transformers-4.29",
    "opt-66b",
    "llama-66b-hf-transformers-4.29",
]

FULL_MODEL_LIST = [
    # TODO: small model have groupsize problem
    # "opt-1.3b",
    # "opt-2.7b",
    "opt-6.7b",
    "llama-7b-hf-transformers-4.29",
    "opt-13b",
    "llama-13b-hf-transformers-4.29",
    "opt-30b",
    "llama-30b-hf-transformers-4.29",
    "opt-66b",
    "llama-66b-hf-transformers-4.29",
]

CONFIG_DICT = {
    # Baseline Experiments

    # FULL PRECISION
    "W16A16": {
        "wbits": 16,
        "abits": 16,
    },

    # qkvproj W16A4 & W16A8
    "qkvproj_W16A4": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-qkvproj": None,
    },
    "qkvproj_W16A8": {
        "wbits": 16,
        "abits": 8,
        "aow-quant-act-qkvproj": None,
    },

    # fc1 W16A4 & W16A8
    "fc1_W16A4": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc1": None,
    },
    "fc1_W16A8": {
        "wbits": 16,
        "abits": 8,
        "aow-quant-act-fc1": None,
    },

    # oproj W16A4 & W16A8
    "oproj_W16A4": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-oproj": None,
    },
    "oproj_W16A8": {
        "wbits": 16,
        "abits": 8,
        "aow-quant-act-oproj": None,
    },

    # fc2 W16A4 & W16A8
    "fc2_W16A4": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc2": None,
    },
    "fc2_W16A8": {
        "wbits": 16,
        "abits": 8,
        "aow-quant-act-fc2": None,
    },

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

    # Modifications

    # qkvproj with FP16 outliers
    "qkvproj_W16A4_ol0.01": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-qkvproj": None,
        "high-prec-ratio": 0.01,
    },

    # fc1 with FP16 outliers
    "fc1_W16A4_ol0.01": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc1": None,
        "high-prec-ratio": 0.01,
    },

    # fc2 with groupwise quantization
    "fc2_W16A4_g128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-fc2": None,
        "act_group_size": 128,
    },

    # oproj with groupwise quantization
    "oproj_W16A4_g128": {
        "wbits": 16,
        "abits": 4,
        "aow-quant-act-oproj": None,
        "act_group_size": 128,
    },
}

def gen_single_experiment_script(
    model_name,
    available_gpus,
    extra_configs: dict = {},
) -> str:
    """
    Generate script for running a single experiment on one model.
    """
    assert "wbits" in extra_configs
    assert "abits" in extra_configs

    scripts = f"""
CUDA_VISIBLE_DEVICES=\"{available_gpus}\" python main.py \\
--eval_ppl --epoch 0 --quant-method aowquant \\
--model $MODEL_DIR/{model_name} \\
--output_dir $OUTPUT_DIR/{model_name} \\
"""
    for config_name, config_val in extra_configs.items():
        if config_val is None:
            scripts += f"--{config_name} \\\n"
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
    for config_name, config_val in CONFIG_DICT.items():
        pack_experiments(
            model_list=model_list,
            model_dir=MODEL_DIR,
            output_dir=f"./output/{config_name}",
            script_path=f"./scripts/aow/{config_name}.sh",
            extra_configs=config_val,
        )
    
    # Generate an entry script
    with open("./scripts/aow/run_all.sh", "w") as f:
        f.write("#!/bin/bash\n\n")
        for config_name in CONFIG_DICT.keys():
            f.write(f"bash ./scripts/aow/{config_name}.sh\n")

    
if __name__ == "__main__":
    gen_all_scripts(DEMO_MODEL_LIST)