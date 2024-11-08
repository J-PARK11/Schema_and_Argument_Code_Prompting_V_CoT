python eval.py --mode supervised_test --model_name Qwen2_VL_2B --load_ckpt_path None
python eval.py --mode supervised_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_option_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode supervised_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_value_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode supervised_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_value_no_option_prompt_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode supervised_test --model_name Qwen2_VL_2B_Clf --load_ckpt_path qwen2_vl_2b_clf_batch_8_lr_1e5_epoch5/epoch_5

python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B --load_ckpt_path None
python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_option_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_value_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B --load_ckpt_path qwen2_vl_2b_value_no_option_prompt_batch_8_lr_1e5_epoch5/epoch_5
python eval.py --mode zeroshot_test --model_name Qwen2_VL_2B_Clf --load_ckpt_path qwen2_vl_2b_clf_batch_8_lr_1e5_epoch5/epoch_5

CUDA_VISIBLE_DEVICES=1 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type value --load_ckpt_path value_generation_wo_option_prompt_supervised/epoch_2 --use_gpu 0 --use_img True
CUDA_VISIBLE_DEVICES=2 python eval.py --model_name Qwen2_VL_2B --epochs 5 --batch_size 8 --lr 1e-5 --experiment supervised --answer_type value --load_ckpt_path value_gen_wo_option_lr_scheduler/epoch_2 --use_gpu 0 --use_img True