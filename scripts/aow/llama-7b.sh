CUDA_VISIBLE_DEVICES=0 python main.py \
--model  /home/zhangchen/hugginface/llama-7b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-7b-aow \
--quant-method aowquant \
--wbits 4 --abits 4 &
# Currently by default, aowquant doesn't quantize weight and activation unless specified!

CUDA_VISIBLE_DEVICES=1 python main.py \
--model  /home/zhangchen/hugginface/llama-7b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-7b-aow-qkv \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-qkv &

CUDA_VISIBLE_DEVICES=2 python main.py \
--model  /home/zhangchen/hugginface/llama-7b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-7b-aow-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-fc1 &

CUDA_VISIBLE_DEVICES=3 python main.py \
--model  /home/zhangchen/hugginface/llama-7b-hf-transformers-4.29 --eval_ppl \
--epoch 0 --output_dir ./log/llama-7b-aow-qkv-fc1 \
--quant-method aowquant \
--wbits 4 --abits 4 \
--aow-quant-act-qkv --aow-quant-act-fc1 &