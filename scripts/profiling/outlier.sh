
mkdir -p outlier_stats/3_sigma
mkdir -p outlier_stats/6_sigma

# for debugging, add --num-samples 1
# CUDA_VISIBLE_DEVICES="0" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/opt-6.7b --profile-outlier-stats \
# --num-samples 1 \
# --outlier-stats-output-path ./outlier_stats/debug --outlier-threshold 3 &

CUDA_VISIBLE_DEVICES="0" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/opt-6.7b --profile-outlier-stats \
--num-samples 128 \
--outlier-stats-output-path ./outlier_stats/3_sigma/  --outlier-threshold 3 &

CUDA_VISIBLE_DEVICES="1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-7b-meta --profile-outlier-stats \
--num-samples 128 \
--outlier-stats-output-path ./outlier_stats/3_sigma/  --outlier-threshold 3 &

# CUDA_VISIBLE_DEVICES="2" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/opt-6.7b --profile-outlier-stats \
# --num-samples 128 \
# --outlier-stats-output-path ./outlier_stats/6_sigma/  --outlier-threshold 6 &

# CUDA_VISIBLE_DEVICES="3" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-7b-meta --profile-outlier-stats \
# --num-samples 128 \
# --outlier-stats-output-path ./outlier_stats/6_sigma/  --outlier-threshold 6 &

wait