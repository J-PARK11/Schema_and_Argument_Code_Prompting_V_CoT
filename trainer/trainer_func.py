import os
import json
import torch
import seaborn as sns
from tqdm import tqdm
from torch.optim import AdamW
import pytorch_warmup as warmup
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .logutil import init_logger, get_logger
from torch.optim.lr_scheduler import ExponentialLR

# Seaborn 스타일 설정
sns.set(style="whitegrid")
colors = sns.color_palette("husl", 40)

def trainer_train(args, model, processor, train_loader, valid_loader):
    
    model.train()
    logger, save_folder = get_custom_logger(args)
    
    epochs = args.epochs   
    
    train_loss_logger, valid_loss_logger, lr_logger = dict(), dict(), dict()
    len_train, len_valid = len(train_loader), len(valid_loader)
    grad_accumulation_steps = 2
    valid_frequency_steps = 10
    lr_update_frequency_steps = int(len_train//5)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    scheduler = ExponentialLR(optimizer, gamma=0.8)
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
                
                accumulated_valid_avg_loss = 0
                for v_batch in valid_loader:
                    v_inputs, v_labels = v_batch
                    
                    v_outputs = model(**v_inputs, labels=v_labels)
                    
                    valid_loss = v_outputs.loss
                    accumulated_valid_avg_loss += valid_loss.item()
                    
                accumulated_valid_avg_loss = accumulated_valid_avg_loss / len_valid
                valid_loss_logger[logging_step] = accumulated_valid_avg_loss
                logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, validation loss: {accumulated_valid_avg_loss:.8f}")
                
            # LR Scheduler Update...
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
            
        lr_logger_save_path = os.path.join(args.save_root, args.save_folder, 'lr_logger.json')
        with open(lr_logger_save_path,'w') as f:
            json.dump(lr_logger, f, ensure_ascii=False, indent=4)       
        
        # 학습 결과 Plot 시각화 파일 저장
        loss_curve_path = os.path.join(args.save_root, args.save_folder, 'loss_curve.png')
        plot_loss(train_loss_logger, valid_loss_logger, epoch+1, len_train, loss_curve_path)
        
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
    for batch in test_loader:
        
        inputs, gt = batch
        
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        decoded_text = [text[5:] for text in output_texts]
        
        TP, ALL, result_dict = calc_acc(decoded_text, gt, TP, ALL, result_dict)

    result_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path}_{args.mode}_result_dict.json')
    with open(result_save_path,'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4) 

def calc_acc(pred_list, label_list, TP, ALL, result_dict):
    
    for pred, label in zip(pred_list, label_list):
        eval_answer = pred.upper()
        
        find_std_format1 = eval_answer[-100:].find(f'ANSWER: {label[-1]}')
        find_std_format2 = eval_answer[-100:].find(f'ANSWER:{label[-1]}')
        find_std_format3 = eval_answer[-100:].find(f'ANSWER.{label[-1]}')
        find_std_format4 = eval_answer[-100:].find(f'ANSWER. {label[-1]}')
        find_std_format5 = eval_answer[-100:].find(f'ANSWER IS {label[-1]}')      
          
        find_only_answer1 = (len(eval_answer)==1 and eval_answer[0] == label[-1])    
        find_only_answer2 = (eval_answer[:2] == f'{label[-1]}.')     
        find_only_answer3 = (eval_answer[-2:] == f' {label[-1]}') 
        find_only_answer4 = (eval_answer[-2:] == f':{label[-1]}')
        find_only_answer5 = (eval_answer[-2:] == f'.{label[-1]}')
        find_only_answer6 = (eval_answer[-2:] == f'\n{label[-1]}')
        find_only_answer7 = (eval_answer[-3:] == f' {label[-1]}.')
    
        if ((find_only_answer1) or (find_only_answer2) or (find_only_answer3) or (find_only_answer4) or (find_only_answer5) or (find_only_answer6) or (find_only_answer7) or\
            (find_std_format1>=0) or (find_std_format2>=0) or (find_std_format3>=0) or (find_std_format4>=0) or (find_std_format5>=0)):
            TP += 1   
            hit = True
        else:
            hit = False    
        ALL += 1
        
        result_dict[ALL] = {
            'Pred': pred,
            'Label': label,
            'hit': hit}
    
    save_pred = pred.replace('\n',' ')
    try: print(f"Accuracy = {TP}/{ALL} = {TP/ALL:.4f},  Pred: {save_pred},  Label: {label}")
    except: pass
    return TP, ALL, result_dict

def plot_loss(train_loss, valid_loss, epochs, len_step_per_epoch, save_path=None):

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    # 학습 로스
    train_xticks = list(map(int,list(train_loss.keys())))
    train_xvalues = list(train_loss.values())
    plt.plot(train_xticks, train_xvalues, 'b-', marker='o', label='Training Loss', linewidth=2)
    
    # 검증 로스
    valid_xticks = list(map(int,list(valid_loss.keys())))
    valid_xvalues = list(valid_loss.values())
    plt.plot(valid_xticks, valid_xvalues, 'r-', marker='x', label='Validation Loss', linewidth=2)

    # 그래프 제목과 레이블 설정
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # 에포크 별 수렴 추이
    # Rectangle((x좌표, y좌표), 가로길이, 세로길이)
    max_value = max(train_xvalues+valid_xvalues)
    for i in range(epochs):
        rect = patches.Rectangle((len_step_per_epoch*i,0), len_step_per_epoch, max_value, linewidth=0, edgecolor='none', facecolor=colors[i*9], alpha=0.1)
        plt.gca().add_patch(rect)

    # 범례 및 기타 설정
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 저장 경로가 있으면 파일로 저장
    plt.savefig(save_path, format='png', dpi=300)

def plot_lr_loss(lr_log, epochs, len_step_per_epoch, save_path=None):

    # 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    # LR 그래프
    lr_xticks = list(map(int,list(lr_log.keys())))
    lr_xvalues = list(lr_log.values())
    plt.plot(lr_xticks, lr_xvalues, 'b-', marker='o', label='LR', linewidth=2)

    # 그래프 제목과 레이블 설정
    plt.title('LR Scheduler Updated Graph', fontsize=14, fontweight='bold')
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('LR', fontsize=12)
    
    # 에포크 별 수렴 추이
    # Rectangle((x좌표, y좌표), 가로길이, 세로길이)
    max_value = max(lr_xvalues)
    for i in range(epochs):
        rect = patches.Rectangle((len_step_per_epoch*i,0), len_step_per_epoch, max_value, linewidth=0, edgecolor='none', facecolor=colors[i*9], alpha=0.1)
        plt.gca().add_patch(rect)

    # 범례 및 기타 설정
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # 저장 경로가 있으면 파일로 저장
    plt.savefig(save_path, format='png', dpi=300)