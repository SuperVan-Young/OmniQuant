# MODEL_DIR=/home/zhangchen/hugginface
MODEL_DIR=/home/xuechenhao/huggingface

# Currently by default, aowquant doesn't quantize weight and activation unless specified!
CUDA_VISIBLE_DEVICES=0 python main.py \
--model  $MODEL_DIR/llama-7b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-7b-aow \
--quant-method aowquant \
--wbits 4 --abits 4 &

CUDA_VISIBLE_DEVICES=1 python main.py \
--model  $MODEL_DIR/llama-7b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-7b-aow-qkv \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-qkv &

CUDA_VISIBLE_DEVICES=2 python main.py \
--model  $MODEL_DIR/llama-7b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-7b-aow-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-fc1 &

CUDA_VISIBLE_DEVICES=3 python main.py \
--model  $MODEL_DIR/opt-6.7b --eval_ppl \
--epoch 0 --output_dir ./log/opt-6.7b-aow \
--quant-method aowquant \
--wbits 4 --abits 4 &

CUDA_VISIBLE_DEVICES=4 python main.py \
--model  $MODEL_DIR/opt-6.7b --eval_ppl \
--epoch 0 --output_dir ./log/opt-6.7b-aow-qkv \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-qkv &

CUDA_VISIBLE_DEVICES=5 python main.py \
--model  $MODEL_DIR/opt-6.7b --eval_ppl \
--epoch 0 --output_dir ./log/opt-6.7b-aow-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-fc1 &

wait

CUDA_VISIBLE_DEVICES=0 python main.py \
--model  $MODEL_DIR/llama-13b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-13b-aow-qkv-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 &

CUDA_VISIBLE_DEVICES=1 python main.py \
--model  $MODEL_DIR/llama-13b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-13b-aow-qkv-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-qkv &

CUDA_VISIBLE_DEVICES="2,3" python main.py \
--model  $MODEL_DIR/llama-13b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-13b-aow-qkv-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 \
--multigpu \
--aow-quant-act-fc1 &

CUDA_VISIBLE_DEVICES="4,5" python main.py \
--model  $MODEL_DIR/opt-13b --eval_ppl \
--epoch 0 --output_dir ./log/opt-13b-aow \
--quant-method aowquant \
--multigpu \
--wbits 4 --abits 4 &

CUDA_VISIBLE_DEVICES="6,7" python main.py \
--model  $MODEL_DIR/opt-13b --eval_ppl \
--epoch 0 --output_dir ./log/opt-13b-aow-qkv \
--quant-method aowquant \
--wbits 4 --abits 4 \
--multigpu \
--aow-quant-act-qkv &

CUDA_VISIBLE_DEVICES="0,1" python main.py \
--model  $MODEL_DIR/opt-13b --eval_ppl \
--epoch 0 --output_dir ./log/opt-13b-aow-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 \
--multigpu \
--aow-quant-act-fc1 &

wait