# server list

import math

class ServerConfig():
    def __init__(self, **kwargs) -> None:
        self.gpu_list = kwargs.get('gpu_list')
        self.gpu_memory = kwargs.get('gpu_memory')
        self.model_dir = kwargs.get('model_dir')

    def allocate_gpu(self, model_name):
        weight_size_in_GB = float(model_name.split("-")[1].replace("b", ""))
        required_size_in_GB = weight_size_in_GB * 2.5
        num_gpu = math.ceil(required_size_in_GB / self.gpu_memory)
        assert num_gpu <= len(self.gpu_list), "not enough gpu"
        return num_gpu
    
V100_SERVER_CONFIG = {
    'gpu_list': [2, 3, 4, 5, 6, 7],
    'gpu_memory': 32,
    'model_dir': "/home/xuechenhao/hugginface",
}

A100_SERVER_CONFIG = {
    'gpu_list': [0, 1],
    'gpu_memory': 80,
    'model_dir': "/home/xuechenhao/hugginface",
}

V100_FULL_SERVER_CONFIG = {
    'gpu_list': [0, 1, 2, 3, 4, 5, 6, 7],
    'gpu_memory': 32,
    'model_dir': "/home/xuechenhao/hugginface",
}

A6000_SERVER_CONFIG = {
    'gpu_list': [0, 1, 2, 3],
    'gpu_memory': 48,
    'model_dir': "/home/zhangchen/hugginface",
}

def get_server_config(server_name):
    if server_name == "v100":
        return ServerConfig(**V100_SERVER_CONFIG)
    elif server_name == "a100":
        return ServerConfig(**A100_SERVER_CONFIG)
    elif server_name == "v100_full":
        return ServerConfig(**V100_FULL_SERVER_CONFIG)
    elif server_name == "a6000":
        return ServerConfig(**A6000_SERVER_CONFIG)
    else:
        raise ValueError("server name not supported")
