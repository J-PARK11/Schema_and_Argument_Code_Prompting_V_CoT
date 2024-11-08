import os
import pdb
import math
import json
import torch
import base64
import random
import pickle
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
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
        self.experiment = args.experiment
        self.diff = 'easy'
        self.data_root = args.data_root
        
        # Data Path
        mmcode_json_name = 'mmcode_full_jsonl' # ['mmcode_full_jsonl', 'mmcode_test_jsonl']
        self.mmcode_pseudo_code_path = os.path.join('/data/MMCode/', mmcode_json_name)
        self.smart_pseudo_code_path = os.path.join('/data/SMART101-release-v1/schema_and_argument_data/',
                                             args.pseudo_code_path)

        # MMcode Dataload
        mmcode_dataset = []
        # if mode == 'train':
        #     with open(self.mmcode_pseudo_code_path,'r') as f:
        #         for line in f:
        #             loop = json.loads(line)
        #             selection = {'puzzle_id': 'MMCode',
        #                         'question': loop['question'],
        #                         'images': loop['images'],
        #                         'solutions': loop['solutions'],
        #                         'url': loop['url']}
        #             mmcode_dataset.append(selection)
        #         print(f'MMCode Pseudo Code: {len(mmcode_dataset)}')

        with open(self.smart_pseudo_code_path,'r') as f:
            self.smart_pseudo_code = json.load(f)    
            print(f'SMART Pseudo Code: {len(self.smart_pseudo_code)}\n')            
            
        # SMART Dataload
        self.puzzle_ids = [str(pid) for pid in range(1,102)]
        if mode == 'train':
            self.start_idx = 0
            self.end_idx = 5
        else:
            self.start_idx = 5
            self.end_idx = 10
        smart_data_len = int(len(self.puzzle_ids)*(self.end_idx-self.start_idx))
        self.qa_info = self.Get_Puzzle_Instance()
        
        # Merge Dataset
        self.qa_info.extend(mmcode_dataset)
        
        random.seed(1123)
        random.shuffle(self.qa_info)
        print(f'Code Generation FT Dataloader: SMART data: {smart_data_len}, MMCode data: {len(mmcode_dataset)} = Total: {smart_data_len+len(mmcode_dataset)}')
        
    def Get_Puzzle_Instance(self):
        qa_bundle = []
            
        # 인스턴스 퍼즐 불러오기.
        for puzzle_id in self.puzzle_ids:
            
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            
            qa_info = qa_info[self.start_idx:self.end_idx]
                
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(qa_info[t], qa_info[t]["Answer"])
                
            qa_bundle = qa_bundle + qa_info
            
        return qa_bundle
        
    def ans_encode(self, answer):
        return ord(answer) - ord("A")
    
    def convert_base64_to_pil_image(self, base64_str: str) -> Image:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image
    
    def melt_image(self, images):
        img_bundle, width_bundle, height_bundle = [], [], []
        for img in images:
            img = self.convert_base64_to_pil_image(img)
            img_bundle.append(img), width_bundle.append(img.width), height_bundle.append(img.height)
        return img_bundle, width_bundle, height_bundle
    
    def get_concat_v_img(self, images):
        img_bundle, width_bundle, height_bundle = self.melt_image(images)
        max_width, sum_height = max(width_bundle), sum(height_bundle)
        
        concat_v_img = Image.new('RGB', (max_width, sum_height))
        cur_height = 0
        for img, height in zip(img_bundle, height_bundle):
            concat_v_img.paste(img, (0, cur_height))
            cur_height += img.height
        return concat_v_img

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        
        if pid == 'MMCode':
            data_type = 'MMCode'
        else:
            data_type = 'SMART'
        
        if data_type == 'SMART':
            puzzle_root = pid + "/" + gv.puzzle_diff_str[self.diff] + "/"
            im_path = gv.osp(self.data_root, puzzle_root, "img", info["image"])
            im_name = im_path.split('/')[-1]                    
            im = load_image(im_path)
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
                    
            pseudo_code = self.smart_pseudo_code[f'puzzle_{pid}']
            label = pseudo_code
            id = im_name
        
        else:
            im = self.get_concat_v_img(info['images'])
            q_stn_out = info['question']
            label = info['solutions']
            url = info['url']
            id = pid
                
        return id, im, q_stn_out, label 

    def __len__(self):
        return len(self.qa_info)

def img_train_collator_fn(data, args, processor, device):
    
    messages = []
    for id, im, q_stn_out, label in data:
        
        instruction_prompt = "Please make some python code to solve this question."        
        question = f'Question: {q_stn_out}\nInstruction: {instruction_prompt}'
        # print(question)
        # print(label)        
        # print(im)
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to make some python program to solve the question.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': im},
                    {'type': 'text', 'text': question}]}, 
            
            {   'role': 'assistant', 'content': [{'type': 'text', 'text': f'{label}'}]}]
        
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

def img_test_collator_fn(data, args, processor, device):
    
    messages = []
    img_name = []
    q_stn_list = []
    for id, im, q_stn_out, label in data:
        
        instruction_prompt = "Please make some python code to solve this question."        
        question = f'Question: {q_stn_out}\nInstruction: {instruction_prompt}'
        
        prompt = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are required to make some python program to solve the question.\n"}]},    
            
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': im},
                    {'type': 'text', 'text': question}]}, 
            ]
        
        messages.append(prompt)
        img_name.append(id)
        q_stn_list.append(q_stn_out)
    
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=texts,
        images=image_inputs, # None
        videos=video_inputs, # None
        padding=True,
        return_tensors="pt")

    inputs = inputs.to(device)

    return inputs, img_name, q_stn_list
    

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