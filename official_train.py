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
from datasets_lib.official_data_loader import get_official_dataset
from trainer.official_trainer_func import official_trainer_train
from SMART_official.utils import *

def train():
    
    print('\n*****Schema and Argument Official Train.py Start*****')
    
    # model load...
    model, processor = get_model(args)
    
    # data load...
    train_loader, valid_loader = get_official_dataset(args, processor)
    
    # exe train...
    official_trainer_train(args, model, processor, train_loader, valid_loader)
        
    print('\n*****Schema and Argument Official Train.py Complete*****')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Schema and Argument Code Prompting Train.py")
    
    parser.add_argument(
        "--puzzles", default="all", type=str, help="comma separated / all / puzzle groups (counting,math etc.)"
    )
    parser.add_argument("--batch_size", default=64, type=int, help="batch size (16)")
    parser.add_argument("--num_epochs", default=100, type=int, help="epoch")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate (0.001)")
    parser.add_argument("--test_file", type=str, help="csv file for train")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/SMART101-release-v1/SMART101-Data/",
        help="location of the csv files, and location of the images, relative location is provided in the csv file.",
    )
    parser.add_argument("--train_diff", type=str, default="easy", help="easy/medium/hard")
    parser.add_argument("--test_diff", type=str, default="easy", help="easy/medium/hard")
    parser.add_argument(
        "--split_ratio",
        type=str,
        default="80:5:15",
        help="how to split train and val, when both use the same instance list.",
    )
    parser.add_argument("--save_root", type=str, default="/data/jhpark_checkpoint/schema_and_argument_ckpt/official_classification/", help="location to save intermediate files.")
    parser.add_argument("--vocab_path", type=str, default="none", help="location to save intermediate files.")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--pretrained", type=str, help="should use a pretrained model?")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer to use")
    parser.add_argument("--loss_type", type=str, default="classifier", help="classifier/regression")
    parser.add_argument("--seed", type=int, default=1123, help="seed to use")
    parser.add_argument("--data_tot", type=int, default=2000, help="how many instances to use for train+val+test")
    parser.add_argument("--use_clip_text", action="store_true", help="should use clip text embeddings?")
    parser.add_argument("--no_meta", action="store_true", help="do not use meta learning for optimization?")
    parser.add_argument("--log", action="store_true", help="should print detailed log of accuracy?")
    parser.add_argument("--baselines", action="store_true", help="run the baselines from answer distributions?")
    parser.add_argument(
        "--monolithic", action="store_true", help="use a single head for all puzzles (except the sequential ones)?"
    )
    parser.add_argument(
        "--split_type", type=str, default="standard", help="type of data split: stanard/exclude/puzzle/fewshot"
    )
    parser.add_argument("--word_embed", type=str, default="standard", help="standard/gpt/glove")
    parser.add_argument(
        "--use_single_image_head", action="store_true", help="use a single image head for all the puzzles?"
    )
    parser.add_argument(
        "--fsK", type=int, default=100, help="how many samples should we use to train in a fewshot setting?"
    )
    parser.add_argument("--log_freq", type=int, default=50, help="log frequency?")
    parser.add_argument("--test", action="store_true", help="evaluate a model?")
    parser.add_argument("--train_backbone", action="store_true", help="train the image backbone?")
    parser.add_argument("--no_question", action="store_true", help="do not use questions?")
    parser.add_argument("--no_image", action="store_true", help="do not use images?")
    parser.add_argument("--num_meta_updates", type=int, default=1, help="number of meta updates?")
    parser.add_argument("--feat_size", type=int, default=128, help="intermediate feature size for image and language features?")
    
    parser.add_argument("--use_gpu", type=str, default="0,1,2,3")
    
    # Common arguments...
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="Qwen2_VL_2B_Clf")    
    
    # Train arguments...
    parser.add_argument("--gamma", default=0.8, type=float)
    parser.add_argument("--load_ckpt_path", type=str, default="None")
    
    # Path argumnets...
    parser.add_argument("--save_folder", type=str, default="dump")
    parser.add_argument("--img_dcp_path", type=str, default="img_dcp_b.json")
    parser.add_argument("--pseudo_code_path", type=str, default="smart_20_puzzle_train_pseudo_code.json")
    
    args = parser.parse_args()
    
    if args.split_type == "puzzle":  # use only a single head and single output head for PS.
        args.monolithic = True
        args.use_single_image_head = True
        args.no_meta = True  # we do not use meta learning for puzzle split.

    if args.monolithic:  # in this case, we use a single output head, but do not include sequential puzzles.
        args.no_meta = True

    if args.test:
        assert args.seed > -1  # when evaluating we need to use the seed to take the checkpoint.

    gv.globals_init(args)
    
    args.puzzle_ids_str, args.puzzle_ids = get_puzzle_ids(args)
    args.location = os.path.join(args.save_root, "checkpoints")
    args.log_path = os.path.join(args.save_root, "log")
    
    gv.NUM_CLASSES_PER_PUZZLE = get_puzzle_class_info(
        args
    )  # initialize the global with the number of outputs for each puzzle.

    print(args)
    print("num_puzzles=%d" % (len(args.puzzle_ids)))
    
    set_seed(args.seed)
    
    train()