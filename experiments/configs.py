# Experiment Configurations

from itertools import product

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
            'act_group_size': 128,
            'act_reorder': None,

            # 'act_group_efficient_accumulation': None,
            # 'act_unified_postlayernorm_outlier': None,
        }
        config.update(kwargs)
        config_dict[config_name] = config
