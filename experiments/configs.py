# Experiment Configurations

from itertools import product
from copy import deepcopy

def get_outlier_name(outlier_ratio):
    ol_name = f"1p{int(1/outlier_ratio)}"
    return ol_name

def get_baseline_experiment_configs(**kwargs):
    """
    Baseline experiments
    """
    config_dict = {}

    config_dict['W16A16'] = {
        "wbits": 16,
        "abits": 16,
    }

    layer_type_list = [
        'qkvproj',
        'fc1',
        'oproj',
        'fc2',
        'q',
        'k',
        'v',
    ]
    abits_list = [4, 8]
    for layer_type, abits in product(layer_type_list, abits_list):
        config_name = f"{layer_type}_W16A{abits}"
        config = {
            'wbits': 16,
            'abits': abits,
            f"aow_quant_act_{layer_type}": None,
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_outlier_experiment_configs(**kwargs):
    config_dict = {}

    outlier_ratio_list = [
        1 / 128,
        1 / 64,
        1 / 32,
        1 / 16,
        1 / 8,
        1 / 4,
    ]
    layer_type_list = [
        'qkvproj',
        'fc1',
        'oproj',
        'fc2',
    ] # ignore matmuls
    for layer_type, outlier_ratio in product(layer_type_list, outlier_ratio_list):
        ol_name = get_outlier_name(outlier_ratio)
        config_name = f"{layer_type}_W16A4_ol{ol_name}"
        config = {
            'wbits': 16,
            'abits': 4,
            f"aow_quant_act_{layer_type}": None,
            'act_outlier_ratio': outlier_ratio,
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict


def get_grouping_experiment_configs(**kwargs):
    config_dict = {}
    
    layer_type_list = [
        'qkvproj',
        'fc1',
        'oproj',
        'fc2',
    ] # ignore matmuls
    for layer_type in layer_type_list:
        config_name = f"{layer_type}_W16A4_g128"
        config = {
            'wbits': 16,
            'abits': 4,
            f"aow_quant_act_{layer_type}": None,
            'act_group_size': 128,
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict


def get_efficient_grouping_experiment_configs(**kwargs):
    config_dict = {}

    layer_type_list = [
        'oproj',
        'fc2',
    ]
    use_efficient_accumulation_list = [True, False]
    for layer_type, use_efficient_accumulation in product(
        layer_type_list,
        use_efficient_accumulation_list
        ):
        config_name = f"{layer_type}_W16A4_g128"
        config_name += "_ea" if use_efficient_accumulation else ""
        config = {
            'wbits': 16,
            'abits': 4,
            f'aow_quant_act_{layer_type}': None,
            'act_group_size': 128,
            'act_reorder': None,
        }
        if use_efficient_accumulation:
            config['act_group_efficient_accumulation'] = None
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_unified_postlayernorm_outlier_experiment_configs(**kwargs):
    config_dict = {}

    layer_type_list = [
        'qkvproj',
        'fc1',
    ]
    act_unified_postlayernorm_outlier_list = [True, False]
    for layer_type, act_unified_postlayernorm_outlier in product(
        layer_type_list,
        act_unified_postlayernorm_outlier_list
        ):
        config_name = f"{layer_type}_W16A4_ol1p32"
        config_name += "_uol" if act_unified_postlayernorm_outlier else ""
        config = {
            'wbits': 16,
            'abits': 4,
            f'aow_quant_act_{layer_type}': None,
            'act_outlier_ratio': 1/32,
        }
        if act_unified_postlayernorm_outlier:
            config['act_unified_postlayernorm_outlier'] = None
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_outlier_bits_experiment_configs(**kwargs):
    config_dict = {}

    outlier_bits = [8, 16]
    layer_type_list = [
        'qkvproj',
        'fc1',
        'oproj',
        'fc2',
    ] # ignore matmuls
    for layer_type, outlier_bit in product(layer_type_list, outlier_bits):
        config_name = f"{layer_type}_W16A4_ol1p64_ob{outlier_bit}"
        config = {
            'wbits': 16,
            'abits': 4,
            f"aow_quant_act_{layer_type}": None,
            'act_outlier_ratio': 1/64,
            'act_outlier_bits': outlier_bit,
            'a_dynamic_method': None,
            # 'act_group_size': 128,
            # 'act_reorder': None,

            # 'act_group_efficient_accumulation': None,
            # 'act_unified_postlayernorm_outlier': None,
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_full_model_experiment_configs(**kwargs):
    """
    By default use dynamic quantization
    """
    config_dict = {}

    outlier_ratio_list = [
        1 / 64,
        1 / 32,
        1 / 16,
    ]

    # activation only experiments
    act_outlier_mant_list = [6, 10]
    for outlier_ratio, act_outlier_mant in product(outlier_ratio_list, act_outlier_mant_list):
        ol_ratio_name = get_outlier_name(outlier_ratio)

        config_name = f"all_W16A4O{6+act_outlier_mant}_ol{ol_ratio_name}"
        config = {
            'wbits': 16,
            'abits': 4,
            "aow_quant_act_qkvproj": None,
            "aow_quant_act_oproj": None,
            "aow_quant_act_fc1": None,
            "aow_quant_act_fc2": None,
            "aow_quant_act_q": None,
            "aow_quant_act_k": None,
            "aow_quant_act_v": None,
            'eval_ppl_dataset': 'wikitext2 c4',
            'a_dynamic_method': 'per_token',
            'act_outlier_ratio': outlier_ratio,
            'act_outlier_exp': 5,
            'act_outlier_mant': act_outlier_mant,
        }
        config.update(kwargs)
        config_dict[config_name] = config
        
    # weight activation quantization experiment
    weight_group_size = [None, 128]
    for outlier_ratio, weight_group_size in product(outlier_ratio_list, weight_group_size):
        ol_ratio_name = get_outlier_name(outlier_ratio)

        config_name = f"all_W4A4O12_ol{ol_ratio_name}"
        if weight_group_size is not None:
            config_name += f"_wg{weight_group_size}"
        config = {
            'wbits': 4,
            'abits': 4,
            "aow_quant_act_qkvproj": None,
            "aow_quant_act_oproj": None,
            "aow_quant_act_fc1": None,
            "aow_quant_act_fc2": None,
            "aow_quant_act_q": None,
            "aow_quant_act_k": None,
            "aow_quant_act_v": None,
            'eval_ppl_dataset': 'wikitext2 c4',
            'a_dynamic_method': 'per_token',
            'act_outlier_ratio': outlier_ratio,
            'act_outlier_exp': 5,
            'act_outlier_mant': 6,
        }
        if weight_group_size is not None:
            config['weight_group_size'] = weight_group_size
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_full_model_O8_experiment_configs(**kwargs):
    config_dict = {}

    outlier_ratio_list = [
        1 / 64,
        1 / 32,
        1 / 16,
        0,
    ]

    act_outlier_mant_list = [2]
    for outlier_ratio, act_outlier_mant in product(outlier_ratio_list, act_outlier_mant_list):
        if outlier_ratio == 0:
            config_name = f"all_W4A4"
        else:
            ol_ratio_name = get_outlier_name(outlier_ratio)
            config_name = f"all_W4A4O{6+act_outlier_mant}_ol{ol_ratio_name}"
        config = {
            'wbits': 4,
            'abits': 4,
            "aow_quant_act_qkvproj": None,
            "aow_quant_act_oproj": None,
            "aow_quant_act_fc1": None,
            "aow_quant_act_fc2": None,
            "aow_quant_act_q": None,
            "aow_quant_act_k": None,
            "aow_quant_act_v": None,
            'eval_ppl_dataset': 'wikitext2 c4',
            'a_dynamic_method': 'per_token',
            'act_outlier_ratio': outlier_ratio,
            'act_outlier_exp': 5,
            'act_outlier_mant': act_outlier_mant,
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict


def get_full_model_static_experiment_configs(**kwargs):
    """
    By default use dynamic quantization
    """
    config_dict = {}

    outlier_ratio_list = [
        1 / 32,
    ]
    wbits_list = [4, 16]
    for wbits, outlier_ratio in product(wbits_list, outlier_ratio_list):
        ol_name = get_outlier_name(outlier_ratio)
        config_name = f"all_static_W{wbits}A4_ol{ol_name}"
        config = {
            'wbits': wbits,
            'abits': 4,
            "aow_quant_act_qkvproj": None,
            "aow_quant_act_oproj": None,
            "aow_quant_act_fc1": None,
            "aow_quant_act_fc2": None,
            "aow_quant_act_q": None,
            "aow_quant_act_k": None,
            "aow_quant_act_v": None,
            'act_outlier_ratio': outlier_ratio,
            # 'act_unified_postlayernorm_outlier': None,
            'a_dynamic_method': 'none',
            'act_group_size': 128,
            'act_reorder': None,
            'act_group_efficient_accumulation': None,
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_full_model_accuracy_experiment_configs(**kwargs):
    config_dict = {}

    outlier_ratio_list = [
        1 / 64,
    ]
    act_outlier_mant_list = [6]
    task_list = [
        'piqa',
        'arc_easy',
        'arc_challenge',
        'boolq',
        # 'hellaswag',  # too long!
        'winogrande',
    ]

    for outlier_ratio, act_outlier_mant, task in product(outlier_ratio_list, act_outlier_mant_list, task_list):
        ol_ratio_name = get_outlier_name(outlier_ratio)
        config_name = f"all_W4A4O{6+act_outlier_mant}_ol{ol_ratio_name}_accuracy_{task}"
        config = {
            'wbits': 4,
            'abits': 4,
            "aow_quant_act_qkvproj": None,
            "aow_quant_act_oproj": None,
            "aow_quant_act_fc1": None,
            "aow_quant_act_fc2": None,
            "aow_quant_act_q": None,
            "aow_quant_act_k": None,
            "aow_quant_act_v": None,
            'eval_ppl_dataset': 'wikitext2', # Whatever, this is quick
            'a_dynamic_method': 'per_token',
            'act_outlier_ratio': outlier_ratio,
            'act_outlier_exp': 5,
            'act_outlier_mant': act_outlier_mant,
            'tasks': task,
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_outlier_dse_experiment_configs(**kwargs):
    """
    This is for a figure in the paper
    """
    config_dict = {}

    outlier_ratio_list = [
        1 / 4096,
        1 / 2048,
        1 / 1024,
        1 / 512,
        1 / 256,
        1 / 128,
        1 / 64,
        1 / 32,
        1 / 16,
        1 / 8,
    ]
    act_outlier_mant_list = [2, 6, 10]
    layer_type_list = [
        'qkvproj',
        'fc1',
        'oproj',
        'fc2',
    ] # ignore matmuls

    # activation only experiments
    for layer_type, outlier_ratio, act_outlier_mant in product(layer_type_list, outlier_ratio_list, act_outlier_mant_list):
        ol_ratio_name = get_outlier_name(outlier_ratio)

        config_name = f"{layer_type}_W16A4O{6+act_outlier_mant}_ol{ol_ratio_name}"
        config = {
            'wbits': 16,
            'abits': 4,
            f"aow_quant_act_{layer_type}": None,
            'eval_ppl_dataset': 'wikitext2',
            'a_dynamic_method': 'per_token',
            'act_outlier_ratio': outlier_ratio,
            'act_outlier_exp': 5,
            'act_outlier_mant': act_outlier_mant,
        }
        config.update(kwargs)
        config_dict[config_name] = config

    # W4A4 experiments
    for layer_type in layer_type_list:
        config_name = f"{layer_type}_W16A4"
        config = {
            'wbits': 16,
            'abits': 4,
            f"aow_quant_act_{layer_type}": None,
            'eval_ppl_dataset': 'wikitext2',
            'a_dynamic_method': 'per_token',
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_opt_uniform_mixed_experiment_config():
    """
    Quantization with uniform outlier ratio and mixed outlier ratio
    """
    OPT_001_UNIFORM_CONFIG = {
        'act_outlier_ratio': 1 / 512,
        'act_outlier_exp': 5,
        'act_outlier_mant': 2,
    }
    OPT_01_UNIFORM_CONFIG = {
        'act_outlier_ratio': 1 / 128,
        'act_outlier_exp': 5,
        'act_outlier_mant': 6,
    }
    OPT_001_MIXED_CONFIG = {
        'act_outlier_ratio': 1 / 512,
        'act_outlier_exp': 5,
        'act_outlier_mant': 6,
        'act_outlier_ratio_qkvproj': 1 / 512,
        'act_outlier_ratio_oproj': 1 / 4096,
        'act_outlier_ratio_fc1': 1 / 512,
        'act_outlier_ratio_fc2': 1 / 4096,
    }
    OPT_01_MIXED_CONFIG = {
        'act_outlier_ratio': 1 / 128,
        'act_outlier_exp': 5,
        'act_outlier_mant': 6,
        'act_outlier_ratio_qkvproj': 1 / 64,
        'act_outlier_ratio_oproj': 1 / 128,
        'act_outlier_ratio_fc1': 1 / 64,
        'act_outlier_ratio_fc2': 1 / 128,
    }

    config_dict = {
        'opt_W4A4.01_uniform': OPT_001_UNIFORM_CONFIG,
        'opt_W4A4.1_uniform': OPT_01_UNIFORM_CONFIG,
        'opt_W4A4.01_mixed': OPT_001_MIXED_CONFIG,
        'opt_W4A4.1_mixed': OPT_01_MIXED_CONFIG,
    }

    for config_name, config in config_dict.items():
        config.update({
            'wbits': 4,
            'abits': 4,
            "aow_quant_act_qkvproj": None,
            "aow_quant_act_oproj": None,
            "aow_quant_act_fc1": None,
            "aow_quant_act_fc2": None,
            "aow_quant_act_q": None,
            "aow_quant_act_k": None,
            "aow_quant_act_v": None,
            'eval_ppl_dataset': 'wikitext2 c4',
            'a_dynamic_method': 'per_token',
        })

    return config_dict


def get_llama_uniform_mixed_experiment_config():
    LLAMA_001_UNIFORM_CONFIG = {
        'act_outlier_ratio': 1 / 512,
        'act_outlier_exp': 5,
        'act_outlier_mant': 2,
    }
    LLAMA_001_MIXED_CONFIG = {
        'act_outlier_ratio': 1 / 512,
        'act_outlier_exp': 5,
        'act_outlier_mant': 2,
        # 'act_outlier_ratio_qkvproj': 1 / 256,
        # 'act_outlier_ratio_oproj': 1 / 1024,
        # 'act_outlier_ratio_fc1': 1 / 1024,
        # 'act_outlier_ratio_fc2': 1 / 256,
        
        # The following parameters work better for llama 13b and 30b
        'act_outlier_ratio_qkvproj': 1 / 256,
        'act_outlier_ratio_oproj': 1 / 4096,
        'act_outlier_ratio_fc1': 1 / 512,
        'act_outlier_ratio_fc2': 1 / 256,
    }
    LLAMA_01_MIXED_CONFIG = {
        'act_outlier_ratio': 1 / 128,
        'act_outlier_exp': 5,
        'act_outlier_mant': 2,
        'act_outlier_ratio_qkvproj': 1 / 64,
        'act_outlier_ratio_oproj': 1 / 128,
        'act_outlier_ratio_fc1': 1 / 128,
        'act_outlier_ratio_fc2': 1 / 16,
    }

    config_dict = {
        'llama_W4A4.01_uniform': LLAMA_001_UNIFORM_CONFIG,
        'llama_W4A4.01_mixed': LLAMA_001_MIXED_CONFIG,
        'llama_W4A4.1_mixed': LLAMA_01_MIXED_CONFIG,
    }

    for config_name, config in config_dict.items():
        config.update({
            'wbits': 4,
            'abits': 4,
            "aow_quant_act_qkvproj": None,
            "aow_quant_act_oproj": None,
            "aow_quant_act_fc1": None,
            "aow_quant_act_fc2": None,
            "aow_quant_act_q": None,
            "aow_quant_act_k": None,
            "aow_quant_act_v": None,
            'eval_ppl_dataset': 'wikitext2 c4',
            'a_dynamic_method': 'per_token',
        })

    return config_dict

def get_outlier_dse_v2_experiment_configs(**kwargs):
    """
    This is for a figure in the paper
    """
    config_dict = {}

    outlier_threshold_list = [
        1,
        1.25,
        1.5,
        2,
        3,
    ]
    act_outlier_mant_list = [2, 6]
    layer_type_list = [
        'qkvproj',
        'fc1',
        'oproj',
        'fc2',
    ] # ignore matmuls

    # activation only experiments
    for layer_type, outlier_threshold, act_outlier_mant in product(layer_type_list, outlier_threshold_list, act_outlier_mant_list):
        if act_outlier_mant == 6 and layer_type != 'qkvproj':
            continue

        config_name = f"{layer_type}_W16A4O{6+act_outlier_mant}_ot{outlier_threshold}"
        config = {
            'wbits': 16,
            'abits': 4,
            f"aow_quant_act_{layer_type}": None,
            'eval_ppl_dataset': 'wikitext2',
            'a_dynamic_method': 'per_token',
            'act_outlier_threshold': outlier_threshold,
            'act_outlier_exp': 5,
            'act_outlier_mant': act_outlier_mant,
            'act_outlier_metric': 'threshold',
        }
        config.update(kwargs)
        config_dict[config_name] = config

    return config_dict

def get_olive_experiment_configs():
    """
    Baseline experiments
    """
    config_dict = {}

    config_dict['olive'] = {
        "wbits": 16,
        "abits": 4,
        'quant_method': 'olive',
        'eval_ppl_dataset': 'wikitext2 c4',
    }

    return config_dict