import os
import json
import torch
from tqdm import tqdm
from torch.optim import AdamW
import pytorch_warmup as warmup
from .logutil import init_logger, get_logger
from torch.optim.lr_scheduler import ExponentialLR
from lib.log_and_viz import *

def trainer_train(args, model, processor, train_loader, valid_loader):
    
    model.train()
    logger, save_folder = get_custom_logger(args)
    
    epochs = args.epochs   
    
    train_loss_logger, valid_loss_logger, lr_logger = dict(), dict(), dict()
    len_train, len_valid = len(train_loader), len(valid_loader)
    grad_accumulation_steps = 2
    valid_frequency_steps = 500
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Scheduler and Warmup 
    if args.use_scheduler:
        lr_update_frequency_steps = int(len_train//10)
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=5)
        lr_logger[1] = scheduler.get_lr()[-1]
    
    # Train & Valid Integrated Loop...
    for epoch in tqdm(range(epochs)):
        steps = 0
        accumulated_avg_loss = 0
        
        # Train Loop bundle...
        for batch in train_loader:
            steps += 1
            inputs, labels = batch
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / grad_accumulation_steps
            
            accumulated_avg_loss += loss.item()
            loss.backward()
            
            logging_step = steps+(len_train*epoch)
            if steps % grad_accumulation_steps == 0:
                
                logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, training loss of previous {grad_accumulation_steps} batches: {accumulated_avg_loss:.8f}")
                train_loss_logger[logging_step] = accumulated_avg_loss
                accumulated_avg_loss = 0
                optimizer.step()
                optimizer.zero_grad()
                
            # Valid Loop bundle...
            if steps % valid_frequency_steps == 0:
                
                model.eval()
                accumulated_valid_avg_loss = 0
                for v_batch in valid_loader:
                    v_inputs, v_labels = v_batch
                    
                    v_outputs = model(**v_inputs, labels=v_labels)
                    valid_loss = v_outputs.loss
                    
                    accumulated_valid_avg_loss += valid_loss.item()
                    
                accumulated_valid_avg_loss = accumulated_valid_avg_loss / len_valid
                valid_loss_logger[logging_step] = accumulated_valid_avg_loss
                logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, validation loss: {accumulated_valid_avg_loss:.8f}")
                model.train()
                
                # 학습 결과 JSON 결과 파일 저장
                train_loss_save_path = os.path.join(args.save_root, args.save_folder, 'train_loss.json')
                with open(train_loss_save_path,'w') as f:
                    json.dump(train_loss_logger, f, ensure_ascii=False, indent=4)
                
                valid_loss_save_path = os.path.join(args.save_root, args.save_folder, 'valid_loss.json')
                with open(valid_loss_save_path,'w') as f:
                    json.dump(valid_loss_logger, f, ensure_ascii=False, indent=4) 
                
            # LR Scheduler Update...
            if args.use_scheduler:
                if steps % lr_update_frequency_steps == 0:
                    
                    past_lr = scheduler.get_lr()[-1]
                    with warmup_scheduler.dampening():
                        scheduler.step()
                    current_lr = scheduler.get_lr()[-1]
                    lr_logger[logging_step] = current_lr
                    logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, lr updated: {past_lr:.12f} --> {current_lr:.12f}")            
                    
        # 에폭별로 폴더를 만들기 위한 경로 설정
        epoch_output_dir = os.path.join(save_folder, f"epoch_{epoch + 1}")
        os.makedirs(epoch_output_dir, exist_ok=True)

        # 에폭별로 모델과 프로세서를 저장
        model.save_pretrained(epoch_output_dir)
        processor.save_pretrained(epoch_output_dir)
        
        # 학습 결과 JSON 결과 파일 저장
        train_loss_save_path = os.path.join(args.save_root, args.save_folder, 'train_loss.json')
        with open(train_loss_save_path,'w') as f:
            json.dump(train_loss_logger, f, ensure_ascii=False, indent=4)
        
        valid_loss_save_path = os.path.join(args.save_root, args.save_folder, 'valid_loss.json')
        with open(valid_loss_save_path,'w') as f:
            json.dump(valid_loss_logger, f, ensure_ascii=False, indent=4) 
        
        # 학습 결과 Plot 시각화 파일 저장
        loss_curve_path = os.path.join(args.save_root, args.save_folder, 'loss_curve.png')
        plot_loss(train_loss_logger, valid_loss_logger, epoch+1, len_train, loss_curve_path)
        
        # 학습률 변화 결과 및 시각화 파일 저장
        if args.use_scheduler:
            lr_logger_save_path = os.path.join(args.save_root, args.save_folder, 'lr_logger.json')
            with open(lr_logger_save_path,'w') as f:
                json.dump(lr_logger, f, ensure_ascii=False, indent=4)       
                
            lr_curve_path = os.path.join(args.save_root, args.save_folder, 'lr_scheduler_curve.png')
            plot_lr_loss(lr_logger, epoch+1, len_train, lr_curve_path)
                
        write_chat_template(processor, epoch_output_dir, logger)
                
def get_custom_logger(args):
    save_folder = os.path.join(args.save_root, args.save_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    init_logger(save_folder)
    logger = get_logger()
    return logger, save_folder

def write_chat_template(processor, output_dir, logger):
    output_chat_template_file = os.path.join(output_dir, "chat_template.json")
    chat_template_json_string = json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n"
    with open(output_chat_template_file, "w", encoding="utf-8") as writer:
        writer.write(chat_template_json_string)
        logger.info(f"chat template saved in {output_chat_template_file}")
        
def trainer_generate(args, model, processor, test_loader):
    
    model.eval()
    TP, ALL = 0, 0
    result_dict = dict()
    for idx, batch in enumerate(test_loader):
        
        inputs, puzzle_name = batch
        
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        result_dict[puzzle_name] = {'pred': output_texts}
        
    result_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path}_{args.experiment}_Option_{args.use_option_prompt}_result_dict.json')
    with open(result_save_path,'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4) 