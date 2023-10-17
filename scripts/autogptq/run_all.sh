
CUDA_VISIBLE_DEVICES="2" python autogptq.py --model llama-7b-meta &
CUDA_VISIBLE_DEVICES="3" python autogptq.py --model llama-7b-meta --group_size 128 &
CUDA_VISIBLE_DEVICES="4" python autogptq.py --model opt-6.7b &
CUDA_VISIBLE_DEVICES="5" python autogptq.py --model opt-6.7b --group_size 128 &
