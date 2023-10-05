MODEL_DIR=/home/xuechenhao/huggingface

# CUDA_VISIBLE_DEVICES="0,1,2,3" python main.py \
# --model  $MODEL_DIR/llama-7b-hf-transformers-4.29 --eval_ppl \
# --epoch 0 --output_dir ./log/test \
# --quant-method aowquant \
# --wbits 4 --abits 4 \
# --multigpu

CUDA_VISIBLE_DEVICES=5 python main.py \
--model  $MODEL_DIR/opt-6.7b --eval_ppl \
--epoch 0 --output_dir ./log/opt-6.7b-aow-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-fc1