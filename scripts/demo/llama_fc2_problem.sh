MODEL_DIR=/home/xuechenhao/hugginface
OUTPUT_DIR=./output_tryout

# fc2_W16A4_ol1p8
# mkdir -p $OUTPUT_DIR/fc2_W16A4_ol1p8/llama-7b-meta
# CUDA_VISIBLE_DEVICES="0" python main.py \
# --eval_ppl --epoch 0 --quant-method aowquant \
# --model $MODEL_DIR/llama-7b-meta \
# --output_dir $OUTPUT_DIR/fc2_W16A4_ol1p8/llama-7b-meta \
# --wbits 16 \
# --abits 4 \
# --aow-quant-act-fc2 \
# --act-outlier-ratio 0.125 \
# --eval-ppl-dataset wikitext2 \
# &

# mkdir -p $OUTPUT_DIR/fc2_W16A4_ol1p4/llama-7b-meta
# CUDA_VISIBLE_DEVICES="0" python main.py \
# --eval_ppl --epoch 0 --quant-method aowquant \
# --model $MODEL_DIR/llama-7b-meta \
# --output_dir $OUTPUT_DIR/fc2_W16A4_ol1p4/llama-7b-meta \
# --a_dynamic_method none \
# --wbits 16 \
# --abits 4 \
# --aow-quant-act-fc2 \
# --act-outlier-ratio 0.25 \
# --eval-ppl-dataset wikitext2 \
# &


# mkdir -p $OUTPUT_DIR/fc2_W16A4_g128_r/llama-7b-meta
# CUDA_VISIBLE_DEVICES="1" python main.py \
# --eval_ppl --epoch 0 --quant-method aowquant \
# --model $MODEL_DIR/llama-7b-meta \
# --output_dir $OUTPUT_DIR/fc2_W16A4_g128_r/llama-7b-meta \
# --a_dynamic_method none \
# --wbits 16 \
# --abits 4 \
# --aow-quant-act-fc2 \
# --act-group-size 128 \
# --act-reorder \
# --eval-ppl-dataset wikitext2 \
# &

wait

# mkdir -p $OUTPUT_DIR/fc2_W16A4_g128_ol1p32_r/llama-7b-meta
# CUDA_VISIBLE_DEVICES="0" python main.py \
# --eval_ppl --epoch 0 --quant-method aowquant \
# --model $MODEL_DIR/llama-7b-meta \
# --output_dir $OUTPUT_DIR/fc2_W16A4_g128_ol1p32_r/llama-7b-meta \
# --a_dynamic_method none \
# --wbits 16 \
# --abits 4 \
# --aow-quant-act-fc2 \
# --act-group-size 128 \
# --act-outlier-ratio 0.03125 \
# --act-reorder \
# --eval-ppl-dataset wikitext2 \
# &

# mkdir -p $OUTPUT_DIR/fc2_W16A4_g128_ol1p16_r/llama-7b-meta
# CUDA_VISIBLE_DEVICES="1" python main.py \
# --eval_ppl --epoch 0 --quant-method aowquant \
# --model $MODEL_DIR/llama-7b-meta \
# --output_dir $OUTPUT_DIR/fc2_W16A4_g128_ol1p16_r/llama-7b-meta \
# --a_dynamic_method none \
# --wbits 16 \
# --abits 4 \
# --aow-quant-act-fc2 \
# --act-group-size 128 \
# --act-outlier-ratio 0.0625 \
# --act-reorder \
# --eval-ppl-dataset wikitext2 \
# &

wait

mkdir -p $OUTPUT_DIR/fc2_W16A8_g128_r/llama-7b-meta
CUDA_VISIBLE_DEVICES="0" python main.py \
--eval_ppl --epoch 0 --quant-method aowquant \
--model $MODEL_DIR/llama-7b-meta \
--output_dir $OUTPUT_DIR/fc2_W16A8_g128_r/llama-7b-meta \
--a_dynamic_method none \
--wbits 16 \
--abits 8 \
--aow-quant-act-fc2 \
--act-group-size 128 \
--act-reorder \
--eval-ppl-dataset wikitext2 \
&

mkdir -p $OUTPUT_DIR/fc2_W16A4_g128_ol1p4_r/llama-7b-meta
CUDA_VISIBLE_DEVICES="1" python main.py \
--eval_ppl --epoch 0 --quant-method aowquant \
--model $MODEL_DIR/llama-7b-meta \
--output_dir $OUTPUT_DIR/fc2_W16A4_g128_ol1p4_r/llama-7b-meta \
--a_dynamic_method none \
--wbits 16 \
--abits 4 \
--aow-quant-act-fc2 \
--act-group-size 128 \
--act-outlier-ratio 0.25 \
--act-reorder \
--eval-ppl-dataset wikitext2 \
&