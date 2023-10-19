# generate scripts for all experiment configs

import os

def get_experiment_bash_script(
    model_path,
    available_gpus,
    experiment_config,
    output_dir,
    bg=True,
    experiment_name=None,
):
    """
    Bash script for single model, single config
    """

    scripts = f"""
# Experiment: {experiment_name}
mkdir -p {output_dir}
CUDA_VISIBLE_DEVICES=\"{available_gpus}\" python main.py \\
--eval_ppl --epoch 0 --quant_method aowquant \\
--model {model_path} \\
--output_dir {output_dir} \\
"""
    for config_name, config_val in experiment_config.items():
        if config_val is None:
            scripts += f"--{config_name} \\\n"
        elif isinstance(config_val, (list, tuple)):
            scripts += f"--{config_name} {' '.join([str(val) for val in config_val])} \\\n"
        else:
            scripts += f"--{config_name} {config_val} \\\n"
    if bg:
        scripts += "&\n\n"
    return scripts

def get_multi_config_script(
    model_name,
    server_config,
    experiment_config_dict,
    output_pdir,
):
    """
    Bash script for single model, multiple configs
    Results are organized in top_output_pdir/{experiment_name} way
    """
    all_scripts = ""
    server_config.reset_gpu_ids()

    for experiment_name, experiment_config in experiment_config_dict.items():
        model_path = os.path.join(server_config.model_dir, model_name)
        
        gpu_ids, is_full = server_config.allocate_gpu_ids(model_name)
        available_gpus = ",".join([str(gpu_id) for gpu_id in gpu_ids])

        output_dir = os.path.join(output_pdir, experiment_name)

        script = get_experiment_bash_script(
            model_path,
            available_gpus,
            experiment_config,
            output_dir,
            bg=True,
            experiment_name=experiment_name,
        )

        # concatenate to all scripts
        if is_full:
            all_scripts += "wait\n"
        all_scripts += script
    
    return all_scripts

def get_multi_model_script(
    model_name_list,
    server_config,
    experiment_config_dict,
    top_output_dir,
):
    """
    Bash script for multiple models, multiple configs
    Results are organized in top_output_dir/{model_name}/{experiment_name} way
    """
    all_scripts = ""

    for model_name in model_name_list:
        model_output_dir = os.path.join(top_output_dir, model_name)

        all_scripts += get_multi_config_script(
            model_name,
            server_config,
            experiment_config_dict,
            model_output_dir,
        )
    
    return all_scripts