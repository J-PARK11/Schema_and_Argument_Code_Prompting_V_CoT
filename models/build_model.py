import os
import torch
from peft import LoraConfig
from models.qwen2_vl import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor

def get_model(args):
    
    if args.model_name == 'Qwen2_VL':
        use_gpu = args.use_gpu.split(',')
        max_memory={
            int(gpu_num): "40GiB" for gpu_num in use_gpu
        }
        
        # Train Setting
        pretrained_path = "Qwen/Qwen2-VL-2B-Instruct"
        if args.load_ckpt_path == 'None':
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="balanced", # [balanced, auto]
                max_memory=max_memory) 
            
            print('\n*****Build Pretrained Qwen2_VL Model*****')
            print(f'Load ckpt path: {pretrained_path}..')
        
        # Generation or Evaluation Setting
        else:
            load_ckpt_path = os.path.join(args.save_root, args.load_ckpt_path)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                load_ckpt_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="balanced", # [balanced, auto]
                max_memory=max_memory)
            
            print('\n*****Build Trained Qwen2_VL Model*****')
            print(f'Load ckpt path: {args.load_ckpt_path}..')

        processor = AutoProcessor.from_pretrained(pretrained_path)
        print(f'Load Processor: {pretrained_path}..')
        print(f'Available GPU num: {use_gpu}....')
    
    return model, processor
