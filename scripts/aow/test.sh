# MODEL_DIR=/home/zhangchen/hugginface
MODEL_DIR=/home/xuechenhao/huggingface

# CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py \
# --model  $MODEL_DIR/llama-7b-hf-transformers-4.29 --eval_ppl \
# --epoch 0 --output_dir ./log/test \
# --quant-method aowquant \
# --wbits 4 --abits 4 \
# --multigpu

# CUDA_VISIBLE_DEVICES=0 python main.py \
# --model  $MODEL_DIR/opt-1.3b --eval_ppl \
# --epoch 0 --output_dir ./log/test \
# --quant-method omniquant \
# --wbits 4 --abits 4 \
# --group_size 128

CUDA_VISIBLE_DEVICES="2,3" python main.py \
--model  $MODEL_DIR/opt-2.7b --eval_ppl \
--epoch 0 --output_dir ./log/opt-2.7b-aow \
--quant-method aowquant \
--multigpu \
--wbits 4 --abits 4