MODEL_DIR=/home/xuechenhao/hugginface
OUTPUT_DIR=./output_tryout

mkdir -p $OUTPUT_DIR/full_model_ol1p32_static
CUDA_VISIBLE_DEVICES="0" python main.py \
--eval_ppl --epoch 0 --quant-method aowquant \
--model $MODEL_DIR/llama-7b-meta \
--output_dir $OUTPUT_DIR/full_model_ol1p32_static/llama-7b-meta \
--wbits 16 \
--abits 4 \
--aow-quant-act-qkvproj \
--aow-quant-act-oproj \
--aow-quant-act-fc1 \
--aow-quant-act-fc2 \
--aow-quant-act-q \
--aow-quant-act-k \
--aow-quant-act-v \
--act-outlier-ratio 0.03125 \
--act-reorder \
--act-group-size 128 \
--a_dynamic_method none \
--eval-ppl-dataset wikitext2 \
&

mkdir -p $OUTPUT_DIR/full_model_ol1p32
CUDA_VISIBLE_DEVICES="1" python main.py \
--eval_ppl --epoch 0 --quant-method aowquant \
--model $MODEL_DIR/llama-7b-meta \
--output_dir $OUTPUT_DIR/full_model_ol1p32/llama-7b-meta \
--wbits 16 \
--abits 4 \
--aow-quant-act-qkvproj \
--aow-quant-act-oproj \
--aow-quant-act-fc1 \
--aow-quant-act-fc2 \
--aow-quant-act-q \
--aow-quant-act-k \
--aow-quant-act-v \
--act-outlier-ratio 0.03125 \
--act-reorder \
--act-group-size 128 \
--eval-ppl-dataset wikitext2 \
&

mkdir -p $OUTPUT_DIR/full_model_ol1p32_static
CUDA_VISIBLE_DEVICES="2" python main.py \
--eval_ppl --epoch 0 --quant-method aowquant \
--model $MODEL_DIR/opt-6.7b \
--output_dir $OUTPUT_DIR/full_model_ol1p32_static/opt-6.7b \
--wbits 16 \
--abits 4 \
--aow-quant-act-qkvproj \
--aow-quant-act-oproj \
--aow-quant-act-fc1 \
--aow-quant-act-fc2 \
--aow-quant-act-q \
--aow-quant-act-k \
--aow-quant-act-v \
--act-outlier-ratio 0.03125 \
--act-reorder \
--act-group-size 128 \
--a_dynamic_method none \
--eval-ppl-dataset wikitext2 \
&

mkdir -p $OUTPUT_DIR/full_model_ol1p32
CUDA_VISIBLE_DEVICES="3" python main.py \
--eval_ppl --epoch 0 --quant-method aowquant \
--model $MODEL_DIR/opt-6.7b \
--output_dir $OUTPUT_DIR/full_model_ol1p32/opt-6.7b \
--wbits 16 \
--abits 4 \
--aow-quant-act-qkvproj \
--aow-quant-act-oproj \
--aow-quant-act-fc1 \
--aow-quant-act-fc2 \
--aow-quant-act-q \
--aow-quant-act-k \
--aow-quant-act-v \
--act-outlier-ratio 0.03125 \
--act-reorder \
--act-group-size 128 \
--eval-ppl-dataset wikitext2 \
&

wait