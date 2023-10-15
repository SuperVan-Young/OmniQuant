
# CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/opt-6.7b --profile-outlier-stats --num-samples 1
CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/opt-6.7b --profile-outlier-stats

CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-7b-meta --profile-outlier-stats
# CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-13b-meta --profile-outlier-stats
# CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-30b-meta --profile-outlier-stats
# CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-65b-meta --profile-outlier-stats