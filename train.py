# ===== Schema & Argument Code Prompting Train.py ===== #

# Common Library
import os
import torch
# import wandb
import argparse
from transformers import set_seed
 
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
    parser.add_argument("--model_name", type=str, default="Qwen2_VL_2B")    
    parser.add_argument("--data_root", type=str, default="/data/SMART101-release-v1/SMART101-Data/")
    
    # Train arguments...
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--loss_type", type=str, default="classifier")
    parser.add_argument("--load_ckpt_path", type=str, default="None")
    parser.add_argument("--use_gpu", type=str, default="0,1,2")
    parser.add_argument("--seed", type=int, default=1123)
    
    # Experiment arguments...
    parser.add_argument("--experiment", type=str, default='supervised') # [supervised, zeroshot]
    parser.add_argument("--answer_type", type=str, default='value') # [option, value]
    parser.add_argument("--use_option_prompt", type=bool, default=False)
    # parser.add_argument("--use_img", type=bool, default=False)
    # parser.add_argument("--use_img_dcp", type=bool, default=False)
    # parser.add_argument("--use_pseudo_code", type=bool, default=False)    
    
    # Path argumnets...
    parser.add_argument("--save_root", type=str, default='/data/jhpark_checkpoint/schema_and_argument_ckpt')
    parser.add_argument("--save_folder", type=str, default="dump")
    parser.add_argument("--img_dcp_path", type=str, default="puzzle_img_dcp_101.json")
    parser.add_argument("--pseudo_code_path", type=str, default="puzzle_pseudo_code_101.json")
    
    args = parser.parse_args()
    print(args)
    gv.custom_globals_init()  
    set_seed(args.seed)
    
    train()