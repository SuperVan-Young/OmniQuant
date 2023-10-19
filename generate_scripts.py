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
parser.add_argument("--model_list", choices=['demo', 'small', 'mediam', 'large', 'all', 'opt_all', 'llama_all'], default='demo', type=str, help="model list")
parser.add_argument("--a_dynamic_method", type=str, default="none", choices=["per_token", 'none'])

args = parser.parse_args()

def write_script(script, script_path):
    with open(script_path, "w") as f:
        f.write(script)

def main():
    server = get_server_config(args.server)
    model_list = get_model_list(args.model_list)

    

    
if __name__ == "__main__":
    main()