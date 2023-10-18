from huggingface_hub import snapshot_download

snapshot_download(repo_id="huggyllama/llama-7b", ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors", "model.safetensors.index.json"],  local_dir="/home/xuechenhao/hugginface/llama-7b-huggy", local_dir_use_symlinks=False)
# snapshot_download(repo_id="decapoda-research/llama-13b-hf", ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors"], local_dir="/home/xuechenhao/hugginface/llama-13b", local_dir_use_symlinks=False)
# snapshot_download(repo_id="decapoda-research/llama-30b-hf", ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors"], local_dir="/home/xuechenhao/hugginface/llama-30b", local_dir_use_symlinks=False)
# snapshot_download(repo_id="decapoda-research/llama-65b-hf", ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors"], local_dir="/home/xuechenhao/hugginface/llama-65b", local_dir_use_symlinks=False)
