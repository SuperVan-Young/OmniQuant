# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/llama-7b-hf-transformers-4.29 --profile-all-stats
# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/llama-13b-hf-transformers-4.29 --profile-all-stats
# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/llama-30b-hf-transformers-4.29 --profile-all-stats
# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/llama-65b-hf-transformers-4.29 --profile-all-stats

# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/opt-1.3b --profile-all-stats
# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/opt-2.7b --profile-all-stats
# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/opt-6.7b --profile-all-stats
# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/opt-13b --profile-all-stats
# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/opt-30b --profile-all-stats
# python generate_act_scale_shift.py --model /home/zhangchen/hugginface/opt-66b --profile-all-stats

CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-7b-meta --profile-all-stats
CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-13b-meta --profile-all-stats
CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-30b-meta --profile-all-stats
CUDA_VISIBLE_DEVICES="0,1" python generate_act_scale_shift.py --model /home/xuechenhao/hugginface/llama-65b-meta --profile-all-stats