# debugging
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --mode train --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --save_folder dump

# Value Generation with Option Prompt
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 3 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type value --use_option_prompt True --save_folder value_generation_with_option_prompt_supervised --use_gpu 0,1
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 3 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type value --use_option_prompt True --save_folder value_generation_with_option_prompt_zeroshot --use_gpu 2,3

# Value Generation w/o Option Prompt
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 3 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_generation_wo_option_prompt_supervised
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 3 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type value --save_folder value_generation_wo_option_prompt_zeroshot

# Option Generation
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 3 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type option --use_option_prompt True --save_folder option_generation_supervised
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 3 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type option --use_option_prompt True --save_folder option_generation_zeroshot