import os
import pdb
import torch
import numpy as np
import lib.utils as utils

def custom_globals_init():
    global puzzle_diff, puzzle_diff_str, osp, rand, MAX_VAL, MAX_DECODE_STEPS, max_qlen
    global num_puzzles, seed, icon_class_ids, signs
    global SEQ_PUZZLES, NUM_CLASSES_PER_PUZZLE, device, SMART_DATASET_INFO_FILE
    global word_dim, word_embed
    global puzzles_not_included, num_actual_puzz
    global PS_VAL_IDX, PS_TEST_IDX
    global VLAR_CHALLENGE_data_root, VLAR_CHALLENGE_submission_root
    global device
    
    # 내가 추가한 변수들 Global 초기화
    global num_puzzle_category, puzzle_category_dict

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    puzzle_diff = {"easy": ""}  # {'easy': 'e', 'medium': 'm', 'hard': 'h'}
    puzzle_diff_str = {"easy": ""}
    osp = os.path.join
    rand = lambda: np.random.rand() > 0.5
    MAX_VAL = 256
    MAX_DECODE_STEPS = 10  # number of steps to decode the LSTM.
    num_puzzles = 101
    max_qlen = 110
    seed = 10
    icon_dataset_path = "/data/SMART101-release-v1/icon-classes.txt"
    icon_class_ids = utils.get_icon_dataset_classes(icon_dataset_path) 
    signs = np.array(["+", "-", "x", "/"])  # puzzle 58
    NUM_CLASSES_PER_PUZZLE = {}
    SEQ_PUZZLES = [16, 18, 35, 39, 63, 100]
    SMART_DATASET_INFO_FILE = "/data/SMART101-release-v1/SMART_info_v2.csv"
    num_actual_puzz = 102
    puzzles_not_included = set([])
    PS_VAL_IDX = [7, 43, 64]
    PS_TEST_IDX = [94, 95, 96, 97, 98, 99, 101, 61, 62, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77]
    
    # 내가 추가한 Global 변수들 상세
    num_puzzle_category = 8
    puzzle_category_dict = {'counting': 1, 'algebra': 2, 'math':3, 'path':4,
                            'spatial':5, 'measure':6, 'logic':7, 'pattern':8}
