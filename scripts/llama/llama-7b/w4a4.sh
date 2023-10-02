CUDA_VISIBLE_DEVICES=0 python main.py \
--model  /home/zhangchen/hugginface/llama-7b-hf-transformers-4.29 --eval_ppl \
--epochs 20 --output_dir ./log/llama-7b-w4a4 \
--wbits 4 --abits 4 --lwc --let --aug_loss