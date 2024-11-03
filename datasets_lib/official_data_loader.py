#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import os
import warnings
from functools import partial

import numpy as np
import torch

warnings.filterwarnings("ignore")
import pdb
import pickle

import nltk
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from models.qwen2_vl.qwen_vl_utils import process_vision_info

import SMART_official.baselines
import SMART_official.globvars as gv
import SMART_official.utils

class SMART_Data(Dataset):
    def __init__(self, args):
        vocab_path = args.vocab_path
        self.max_qlen = 110
        self.max_olen = 4  # max option length
        self.use_word_embed = False
        self.word_embed = None
        self.im_side = 224
        self.preprocess = None
        self.no_question = args.no_question
        self.no_image = args.no_image

    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def split_fewshot_puzzles(self, puzzle_ids, split_ratio, split_name, split_type):
        if split_name == "train":
            split_pids = self.split_puzzles(puzzle_ids, split_ratio, "train", split_type)
            other_pids = self.split_puzzles(puzzle_ids, split_ratio, "test", split_type)
            other_pids = other_pids + self.split_puzzles(puzzle_ids, split_ratio, "val", split_type)
            return split_pids, other_pids
        else:
            split_pids = self.split_puzzles(puzzle_ids, split_ratio, split_name, split_type)
            other_pids = None
        return split_pids, other_pids

    def split_puzzles(self, puzzle_ids, split_ratio, split_name, split_type):
        if split_type == "puzzle" or split_type == "fewshot":
            if split_name == "train":
                val_test = gv.PS_VAL_IDX + gv.PS_TEST_IDX
                val_test = set([str(ii) for ii in val_test])
                puzzle_ids = list(set(puzzle_ids).difference(val_test))
                print("number of train puzzles = %d" % (len(puzzle_ids)))
            elif split_name == "val":
                puzzle_ids = [str(ii) for ii in gv.PS_VAL_IDX]
                print("number of val puzzles = %d" % (len(puzzle_ids)))
            else:
                puzzle_ids = [str(ii) for ii in gv.PS_TEST_IDX]
                print("number of test puzzles = %d" % (len(puzzle_ids)))
        else:
            splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
            n = len(puzzle_ids)
            if split_name == "train":
                st = 0
                en = int(np.floor(n * splits[0] / 100.0))
                puzzle_ids = puzzle_ids[st:en]
            elif split_name == "val":
                st = int(np.ceil(n * splits[0] / 100.0))
                en = int(np.floor(n * splits[1] / 100.0))
                puzzle_ids = puzzle_ids[st:en]
            else:
                st = int(np.ceil(n * splits[1] / 100.0))
                puzzle_ids = puzzle_ids[st:]
        print("puzzles for %s =" % (split_name))
        print(puzzle_ids)
        return puzzle_ids

    def split_data(self, info, split_ratio, split_name, split_type="standard"):
        """
        split_type=standard is to use the split_ratio in the instance order
        split_type=exclude is to exclude answers from the split, e.g., train on all answers except say 1, and test 1
        split_type=puzzle is to split the puzzles into the respective ratios. so we don't have to do anything here.
        """
        if split_type == "standard" or split_type == "puzzle" or split_type == "fewshot":
            splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
            n = len(info)
            if split_name == "train":
                st = 0
                en = int(np.floor(n * splits[0] / 100.0))
                info = info[st:en]
            elif split_name == "val":
                st = int(np.ceil(n * splits[0] / 100.0))
                en = int(np.floor(n * splits[1] / 100.0))
                info = info[st:en]
            else:
                st = int(np.ceil(n * splits[1] / 100.0))
                info = info[st:]
        elif split_type == "exclude":
            pid = info[0]["puzzle_id"]
            if int(pid) in gv.SEQ_PUZZLES or int(pid) == 58:
                # we don't do exclude splits for seq_puzzles are as they are most likely always unique
                info = self.split_data(info, split_ratio, split_name, split_type="standard")
            else:
                ans_distr = []
                for t in range(len(info)):
                    ans_distr.append(info[t]["AnswerValue"])
                ans_distr = np.array(ans_distr)
                bclassids = np.arange(gv.NUM_CLASSES_PER_PUZZLE[pid])
                x = np.histogram(ans_distr, bclassids)[0]
                x = x / x.sum()

                # select reasonable answers.
                valid_ans_idx = np.where(x > 0.01)
                x_cls = bclassids[valid_ans_idx]
                x = x[valid_ans_idx]
                median_class = x_cls[x <= np.median(x)][-1]
                try:
                    train_inst = np.array(info)[ans_distr != median_class]
                    test_inst = np.array(info)[ans_distr == median_class]
                except:
                    print(pid)
                    pdb.set_trace()

                n = len(train_inst)
                splits = np.array([int(spl) for spl in split_ratio.split(":")])
                splits[0] = splits[0] + splits[2]
                splits = splits.cumsum()[:2]

                if split_name == "train":
                    st = 0
                    en = int(np.floor(n * splits[0] / 100.0))
                    info = train_inst[st:en].tolist()
                elif split_name == "val":
                    st = int(np.ceil(n * splits[0] / 100.0))
                    en = int(np.floor(n * splits[1] / 100.0))
                    info = train_inst[st:en].tolist()
                else:
                    info = test_inst.tolist()
        else:
            raise "Unknown puzzle split type!!"

        return info


class SMART_TrainData(SMART_Data):
    def __init__(self, args, split):
        super().__init__(args)
        self.data_root = args.data_root
        self.num_tot = args.data_tot  # how many instances per puzzles should we use?
        self.diff = args.train_diff
        self.word_embed = args.word_embed
        self.fewshot_K = args.fsK
        self.puzzle_diff_str = {"easy": ""}
        self.MAX_DECODE_STEPS = 10
        self.SEQ_PUZZLES = [16, 18, 35, 39, 63, 100]
        self.qa_info = []
        train_pids = None

        puzzle_ids = (
            self.split_puzzles(args.puzzle_ids, args.split_ratio, split, args.split_type)
            if args.split_type == "puzzle"
            else args.puzzle_ids
        )
        if args.split_type == "fewshot":
            train_pids, fewshot_other_pids = self.split_fewshot_puzzles(
                args.puzzle_ids, args.split_ratio, split, args.split_type
            )
        
        puzzle_ids = list(set(puzzle_ids) - set(['16', '18', '35', '39', '63', '100']))
            
        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + self.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = SMART_official.utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            if args.split_type == "fewshot" and puzzle_id in fewshot_other_pids:
                qa_info = qa_info[: self.fewshot_K]
            else:
                qa_info = qa_info[: self.num_tot]
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = SMART_official.utils.get_val(qa_info[t], qa_info[t]["Answer"])
            self.qa_info = self.qa_info + self.split_data(qa_info, args.split_ratio, split, args.split_type)
            gv.MAX_VAL = max(gv.MAX_VAL, gv.NUM_CLASSES_PER_PUZZLE[puzzle_id])
        if args.baselines:
            self.baseline_perf = SMART_official.baselines.get_baseline_performance(args, self.qa_info, split, self.num_tot, log=True)
        print("num_train=%d max_answer_value=%d" % (len(self.qa_info), gv.MAX_VAL))
        print("split=%s puzzle_ids=" % (split), end=" ")
        print(puzzle_ids)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = pid + "/" + self.puzzle_diff_str[self.diff] + "/"
        im_path = os.path.join(self.data_root, puzzle_root, "img", info["image"])
        qa = info["Question"]
        opts = 0
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(
            self.MAX_DECODE_STEPS,
        )
        if int(pid) not in self.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            try:
                answer[: len(answer_value)] = answer_value
            except:
                print(info)
                pdb.set_trace()
        return im_path, qa, opts, lbl, answer, pid

    def __len__(self):
        return len(self.qa_info)


class SMART_ValData(SMART_Data):
    def __init__(self, args, split):
        super().__init__(args)
        self.data_root = args.data_root
        self.num_tot = args.data_tot
        self.word_embed = args.word_embed
        self.fewshot_K = args.fsK
        self.qa_info = []

        self.diff = args.test_diff if split == "test" else args.train_diff
        puzzle_ids = (
            self.split_puzzles(args.puzzle_ids, args.split_ratio, split, args.split_type)
            if args.split_type == "puzzle"
            else args.puzzle_ids
        )
        if args.split_type == "fewshot":
            puzzle_ids, fewshot_other_pids = self.split_fewshot_puzzles(
                args.puzzle_ids, args.split_ratio, split, args.split_type
            )
        
        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = SMART_official.utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            if args.split_type == "fewshot":
                qa_info = qa_info[
                    self.fewshot_K : self.num_tot
                ]  # we use the fewshot_K for training. so use the rest for evaluation.
            else:
                qa_info = qa_info[: self.num_tot]
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = SMART_official.utils.get_val(qa_info[t], qa_info[t]["Answer"])
            self.qa_info = self.qa_info + self.split_data(qa_info, args.split_ratio, split, args.split_type)
            gv.MAX_VAL = max(gv.MAX_VAL, gv.NUM_CLASSES_PER_PUZZLE[puzzle_id])
        print("num_val = %d max_answer_value=%d" % (len(self.qa_info), gv.MAX_VAL))
        if args.baselines:
            self.baseline_perf = SMART_official.baselines.get_baseline_performance(args, self.qa_info, split, self.num_tot, log=True)
        print("split=%s puzzle_ids=" % (split), end=" ")
        print(puzzle_ids)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = info["puzzle_id"] + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im_path = gv.osp(self.data_root, puzzle_root, "img", info["image"])
        qa = info["Question"]

        _ = [SMART_official.utils.str_replace_(info, key) for key in ["A", "B", "C", "D", "E"]]
        opts = [SMART_official.utils.get_val(info, key, is_one_of_option=True) for key in ["A", "B", "C", "D", "E"]]
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(
            gv.MAX_DECODE_STEPS,
        )
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            answer[: len(answer_value)] = answer_value
        return im_path, qa, opts, lbl, answer, pid

    def __len__(self):
        return len(self.qa_info)

def SMART_official_collate_fn(data, processor, device):
    
    concat = lambda data_list: torch.cat([x.unsqueeze(0) for x in data_list])
    
    # option_instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided, and answer with the letter corresponding to the options. Answer: ?"
    # value_setting_instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided."

    messages = []
    im_path_list, qa_list, opts_list, lbl_list, answer_list, pid_list = [],[],[],[],[],[]
    for im_path, qa, opts, lbl, answer, puzzle_ids in data:
        # question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nOptions: {options}\nInstruction: {value_setting_instruction_prompt}' # option_instruction_prompt, value_setting_instruction_prompt
        # question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nInstruction: {value_setting_instruction_prompt}'
        question = f'Question: {qa}'
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': im_path},
                    {'type': 'text', 'text': question}]}
            ]
        messages.append(prompt)
        im_path_list.append(im_path), qa_list.append(qa), opts_list.append(opts), lbl_list.append(lbl), answer_list.append(answer), pid_list.append(int(puzzle_ids))
  
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    lbl_list = torch.tensor(lbl_list)
    answer_list = concat(torch.tensor(answer_list))
    pid_list = torch.tensor(pid_list)
    
    inputs['puzzle_ids'] = pid_list

    return inputs, im_path_list, qa_list, opts_list, lbl_list, answer_list, pid_list

def get_official_dataset(args, processor):
    if args.mode == 'train':
        
        print('\n*****Load Train DataLoader*****')
        train_dataset = SMART_TrainData(args, 'train')
        valid_dataset = SMART_ValData(args, 'valid')
        
        collator = partial(SMART_official_collate_fn, processor=processor, device='cuda')
     
        # Train Loader
        train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        # Valid Loader
        valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        return train_loader, valid_loader
    
    elif args.mode in ['supervised_test', 'zeroshot_test']:
        
        print('\n*****Load Test DataLoader*****')
        test_dataset = SMART_ValData(args, args.mode)
        
        collator = partial(SMART_official_collate_fn, processor=processor, device='cuda')
     
        # Test Loader
        test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=collator)
        
        return test_loader