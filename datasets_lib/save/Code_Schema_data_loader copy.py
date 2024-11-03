import os
import pdb
import math
import json
import torch
import random
import pickle
import warnings
import numpy as np
import pandas as pd
from PIL import Image
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset
from transformers.image_utils import load_image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from models.qwen2_vl.qwen_vl_utils import process_vision_info

import lib.utils as utils
import lib.SMART_globvars as gv

class V_COT_SMART101_Dataset(Dataset):
    def __init__(self, args, mode):
        
        self.args = args
        self.mode = mode
        self.diff = 'easy'
        self.data_root = args.data_root
        
        self.use_puzzle_list = "6,10,21,31,46,48,49,56,59,62,63,67,74,88,93,94,96,98,100,101"
                
        self.image_description_path = os.path.join('/data/SMART101-release-v1/schema_and_argument_data',
                                                   args.img_dcp_path)
        self.pseudo_code_path = os.path.join('/data/SMART101-release-v1/schema_and_argument_data',
                                             args.pseudo_code_path)
        
        # Image Description & Pseudo Code Loading.
        with open(self.image_description_path,'r') as f:
            self.image_description = json.load(f)
        
        with open(self.pseudo_code_path,'r') as f:
            self.pseudo_code = json.load(f)    
        
        # Root Puzzle Split
        if self.mode == 'train':
            self.use_puzzle_list = "6,10,21,31,46,48,49,56,59,62,63,74,88,94,96,98,100"
            self.num_tot = 230
        elif self.mode == 'valid':
            self.use_puzzle_list = "6,10,21,31,46,48,49,56,59,62,63,74,88,94,96,98,100"
            self.num_tot = 20
        elif self.mode == 'supervised_test':
            self.use_puzzle_list = "6,10,21,31,46,48,49,56,59,62,63,74,88,94,96,98,100"
            self.num_tot = 50
        elif self.mode == 'zeroshot_test':
            self.use_puzzle_list = "67,93,101"
            self.num_tot = 300
        
        self.puzzle_ids = self.use_puzzle_list.split(',') 
        
        self.qa_info = self.Get_Puzzle_Instance()
        print(f'{self.mode},  Num of Root: {len(self.puzzle_ids)},  Num of Instance per root: {self.num_tot} = {len(self.puzzle_ids)*self.num_tot}')
        
    def Get_Puzzle_Instance(self):
        qa_bundle = []
            
        # 인스턴스 퍼즐 불러오기.
        for puzzle_id in self.puzzle_ids:
            
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            
            # 인스턴스 퍼즐 스플릿.
            if self.mode == 'train':
                qa_info = qa_info[1700:1700+self.num_tot]
            
            elif self.mode == 'valid':
                qa_info = qa_info[1930:1930+self.num_tot]
                
            elif self.mode == 'supervised_test':
                qa_info = qa_info[1950:1950+self.num_tot]
            
            elif self.mode == 'zeroshot_test':
                qa_info = qa_info[1700:1700+self.num_tot]
                
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(qa_info[t], qa_info[t]["Answer"])
                
            qa_bundle = qa_bundle + qa_info
            
        return qa_bundle
        
    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = pid + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im_path = gv.osp(self.data_root, puzzle_root, "img", info["image"])
        im_name = im_path.split('/')[-1]        
        
        # im = load_image(im_path) #.resize((912,912))
        q_stn = info["Question"]
                
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(gv.MAX_DECODE_STEPS)
        
        # 시퀀스 데이터 예외처리.
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            try:
                answer[: len(answer_value)] = answer_value
            except:
                pdb.set_trace()
        
        if not info: info = []
        opts = []
        Answer_Option_phrase = ''
        for op in ["A", "B", "C", "D", "E"]:
            op_val = info[op]
            Answer_Option_phrase += f' {op}. {op_val}'
            opts.append(op_val)
        
        q_stn_out = q_stn
        option_answer= f'Answer: {info["Answer"]}' # info['Answer']
        value_answer = opts[lbl]
        Answer_Option_phrase = Answer_Option_phrase.strip()
        
        
        # Get Image Description & Pseudo Code
        img_dcp = self.image_description[im_name]
        pseudo_code = self.pseudo_code[f'puzzle_{pid}']
                
        return im_name, im_path, pid, q_stn_out, Answer_Option_phrase, option_answer, answer, value_answer, img_dcp, pseudo_code

    def __len__(self):
        return len(self.qa_info)

def img_dcp_train_collator_fn(data, processor, device):
    
    option_instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided, and answer with the letter corresponding to the options. Answer: ?"
    value_setting_instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided."
    
    messages = []
    for im_name, im_path, pid, q_stn_out, options, option_answer, answer, value_answer, img_dcp, pseudo_code in data:
        
        # question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nOptions: {options}\nInstruction: {value_setting_instruction_prompt}' # option_instruction_prompt, value_setting_instruction_prompt
        question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nInstruction: {value_setting_instruction_prompt}'
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question}]}, 
            
            {   'role': 'assistant', 'content': [{'type': 'text', 'text': f'{value_answer}'}]}] # option_answer[-1], value_answer
        # print(prompt)
        messages.append(prompt)
        
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    return inputs, labels_ids

def img_dcp_test_collator_fn(data, processor, device):
    
    option_instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided, and answer with the letter corresponding to the options. Answer: ?"
    value_setting_instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided."
    
    gt = []
    im_name_list = []
    messages = []
    for im_name, im_path, pid, q_stn_out, options, option_answer, answer, value_answer, img_dcp, pseudo_code in data:
        # question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nOptions: {options}\nInstruction: {value_setting_instruction_prompt}' # option_instruction_prompt, value_setting_instruction_prompt
        question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nInstruction: {value_setting_instruction_prompt}'
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question}]}
            ]
        messages.append(prompt)
        gt.append(value_answer)
        im_name_list.append(im_name)
        
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    return inputs, gt, im_name_list

def img_dcp_clf_train_collator_fn(data, processor, device):
    
    instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided."
    
    messages = []
    labels_list = []
    for im_name, im_path, pid, q_stn_out, options, option_answer, answer, value_answer, img_dcp, pseudo_code in data:
        question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nInstruction: {instruction_prompt}'
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question}]}
            
            ]
        
        messages.append(prompt)
        labels_list.append(answer)
        
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    assert len(messages) == len(input_ids_lists)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)
    labels_ids = labels_ids.to(device)
    
    return inputs, labels_ids

def img_dcp_clf_test_collator_fn(data, processor, device):
    
    instruction_prompt = "Please solve the problem using the question, image description, and pseudo code provided."
    
    gt = []
    im_name_list = []
    messages = []
    for im_name, im_path, pid, q_stn_out, options, option_answer, answer, value_answer, img_dcp, pseudo_code in data:
        question = f'Description of image: {img_dcp}\nPseudo_code: {pseudo_code}\nQuestion: {q_stn_out}\nInstruction: {instruction_prompt}'
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to solve a algorithmic problem.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': question}]}
            ]
        messages.append(prompt)
        gt.append(answer)
        im_name_list.append(im_name)
        
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    return inputs, gt, im_name_list

def find_assistant_content_sublist_indexes(l):
    '''
    A message from train_data/data.json may look like below:
        {
            "messages": [
                {'role': 'user', 'content': [{'type': 'image', 'image': 'train_data/1.jpeg'}, {'type': 'text', 'text': '描述一下这个图片'}]}, 
                {'role': 'assistant', 'content': [{'type': 'text', 'text': '这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。'}]}
            ]
        }
    After apply_chat_template, the text will look like below:
        ['<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>描述一下这个图片<|im_end|>\n<|im_start|>assistant\n这张图片展示了一位年轻女子和她的狗在海滩上玩耍的场景。女子穿着格子衬衫和黑色裤子，坐在沙滩上，与她的金毛犬互动。她们的手臂伸展着，似乎在进行某种游戏或训练。背景是广阔的海洋和晴朗的天空，阳光洒在沙滩上，营造出温暖而宁静的氛围。整体画面充满了快乐和放松的感觉。<|im_end|>\n']

    This function tries to find the indexes of the assistant content in the input_ids list to build labels.
    '''
    # (Pdb++) processor.tokenizer.encode("<|im_start|>assistant")
    # [151644, 77091]
    # (Pdb++) processor.tokenizer.encode("<|im_end|>")
    # [151645]

    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))