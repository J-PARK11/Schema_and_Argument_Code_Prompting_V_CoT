# ===== Schema & Argument Code Prompting Train.py ===== #

# Common Library
import os
import torch
# import wandb
import argparse
from transformers import HfArgumentParser, set_seed

import warnings
warnings.filterwarnings('ignore')

# Local Library
import lib.SMART_globvars as gv
from models.build_model import get_model
from datasets_lib.build_dataset import get_dataset
from trainer.trainer_func import trainer_train

def train():
    
    print('\n*****Schema and Argument Train.py Start*****')
    
    # model load...
    model, processor = get_model(args)
    
    # data load...
    train_loader, valid_loader = get_dataset(args, processor)
    
    # exe train...
    trainer_train(args, model, processor, train_loader, valid_loader)
        
    print('\n*****Schema and Argument Train.py Complete*****')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Schema and Argument Code Prompting Train.py")
    
    # Common arguments...
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="Qwen2_VL")    
    parser.add_argument("--data_root", type=str, default="/data/SMART101-release-v1/SMART101-Data/")
    
    # Train arguments...
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--load_ckpt_path", type=str, default="None")
    parser.add_argument("--use_gpu", type=str, default="0,1,2,3")
    parser.add_argument("--seed", type=int, default=1123)
    
    # Path argumnets...
    parser.add_argument("--save_root", type=str, default='/data/jhpark_checkpoint/schema_and_argument_ckpt')
    parser.add_argument("--save_folder", type=str, default="no_lr_scheduler")
    parser.add_argument("--img_dcp_path", type=str, default="img_dcp_b.json")
    parser.add_argument("--pseudo_code_path", type=str, default="smart_20_puzzle_train_pseudo_code.json")
    
    args = parser.parse_args()
    gv.custom_globals_init()  
    set_seed(args.seed)
    
    train()