
CUDA_VISIBLE_DEVICES=2 python single_official_train.py --num_workers 4 --model_name qwen2_vl_2b --num_epochs 1 --loss_type classifier --batch_size 1 --log_freq 10 --train_backbone --word_embed bert --lr 1e-3 --puzzles all --split_type standard --save_root /data/jhpark_checkpoint/schema_and_argument_ckpt/official_classification/supervised/img_baseline --test --seed 5939
CUDA_VISIBLE_DEVICES=3 python single_official_train.py --num_workers 4 --model_name qwen2_vl_2b --num_epochs 1 --loss_type classifier --batch_size 1 --log_freq 10 --train_backbone --word_embed bert --lr 1e-3 --split_type puzzle --save_root /data/jhpark_checkpoint/schema_and_argument_ckpt/official_classification/zeroshot/img_baseline --test --seed 1706

