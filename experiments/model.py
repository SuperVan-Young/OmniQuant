TINY_MODEL_LIST = [
    "opt-6.7b",
    "llama-7b-meta",
]

SMALL_MODEL_LIST = [
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

def get_model_list(model_list_name):
    if model_list_name == "tiny":
        return TINY_MODEL_LIST
    elif model_list_name == "small":
        return SMALL_MODEL_LIST
    elif model_list_name == "medium":
        return MEDIAM_MODEL_LIST
    elif model_list_name == "large":
        return LARGE_MODEL_LIST
    elif model_list_name == "all":
        return ALL_MODEL_LIST
    elif model_list_name == "opt_all":
        return OPT_ALL_MODEL_LIST
    elif model_list_name == "llama_all":
        return LLAMA_ALL_MODEL_LIST
    else:
        raise ValueError(f"model list name not supported: {model_list_name}")