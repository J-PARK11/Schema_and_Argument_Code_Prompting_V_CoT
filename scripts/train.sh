# debugging
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --mode train --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --save_folder dump

# Value Generation with Option Prompt
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type value --use_option_prompt True --save_folder value_generation_with_option_prompt_supervised --use_gpu 0,1
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type value --use_option_prompt True --save_folder value_generation_with_option_prompt_zeroshot --use_gpu 2,3

# Value Generation w/o Option Prompt
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_generation_wo_option_prompt_supervised
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type value --save_folder value_generation_wo_option_prompt_zeroshot

# Option Generation
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type option --use_option_prompt True --save_folder option_generation_supervised
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment zeroshot --answer_type option --use_option_prompt True --save_folder option_generation_zeroshot

# On Air: Value Generation with Option Prompt Use Image and Image Dcp: GPU 0,1
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 4 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_gen_with_opt_dcp_mca --use_gpu 0,1,2 --use_img True --use_option_prompt True
CUDA_VISIBLE_DEVICES=2,3 python train.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 3 --lr 1e-5 --experiment supervised --answer_type value --save_folder value_gen_wo_opt_dcp_mca --use_gpu 0,1 --use_img True


CUDA_VISIBLE_DEVICES=2 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 6 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --use_option_prompt True --load_ckpt_path value_gen_with_opt_dcp_mca/epoch_1/whole_model.pth
CUDA_VISIBLE_DEVICES=3 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 6 --lr 1e-5 --experiment supervised --answer_type value --use_gpu 0 --use_img True --load_ckpt_path value_gen_wo_opt_dcp_mca/epoch_1/whole_model.pth