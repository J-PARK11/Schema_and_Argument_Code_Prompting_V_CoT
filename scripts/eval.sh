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