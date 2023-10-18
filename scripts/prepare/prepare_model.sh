# convert META model to huggingface format
# python /home/xuechenhao/anaconda3/envs/omniquant/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
#     --input_dir /datasets/llama --model_size 7B --output_dir /home/xuechenhao/hugginface/llama-7b-meta

# should be run with sudo

mkdir -p /datasets/llama-hf

python /home/xuechenhao/anaconda3/envs/omniquant/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /datasets/llama --model_size 7B --output_dir /datasets/llama-hf/llama-7b-meta

python /home/xuechenhao/anaconda3/envs/omniquant/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /datasets/llama --model_size 13B --output_dir /datasets/llama-hf/llama-13b-meta

python /home/xuechenhao/anaconda3/envs/omniquant/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /datasets/llama --model_size 30B --output_dir /datasets/llama-hf/llama-30b-meta

python /home/xuechenhao/anaconda3/envs/omniquant/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /datasets/llama --model_size 65B --output_dir /datasets/llama-hf/llama-65b-meta

# scp from A6000 to V100 opt-13b
# scp -r -P 4322 xuechenhao@222.29.98.96:/home/zhangchen/hugginface/opt-13b /home/xuechenhao/hugginface/opt-13b