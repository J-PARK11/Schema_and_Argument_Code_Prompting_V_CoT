import os
import json
import torch
from tqdm import tqdm
from torch.optim import AdamW
import pytorch_warmup as warmup
from .logutil import init_logger, get_logger
from torch.optim.lr_scheduler import ExponentialLR
from lib.log_and_viz import *
from SMART_official.losses import Criterion

def official_trainer_train(args, model, processor, train_loader, valid_loader):
    
    parameters = model.parameters()
    if not args.no_meta:
        anshead_parameters = list(model.ans_decoder.parameters())
        
    logger, save_folder = get_custom_logger(args)
    
    epochs = args.num_epochs   
    train_loss_logger, valid_loss_logger, lr_logger = dict(), dict(), dict()
    len_train, len_valid = len(train_loader), len(valid_loader)
    valid_frequency_steps = 10
    lr_update_frequency_steps = int(len_train//5)
    
    criterion = Criterion(args)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.99))
        if not args.no_meta:
            anshead_optimizer = torch.optim.Adam(anshead_parameters, lr=args.lr, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.SGD(parameters, lr=args.lr)
        if not args.no_meta:
            anshead_optimizer = torch.optim.SGD(anshead_parameters, lr=args.lr)    
    
    # Train & Valid Integrated Loop...
    model.train()
    print(f'\nRequire Grad Parameter numbers: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'\nFreeze Parameter numbers: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}')
    
    tot_loss = 0.0
    for epoch in tqdm(range(epochs)):
        steps = 0
        accumulated_avg_loss = 0
        
        # Train Loop bundle...
        for inputs, im_path_list, qa_list, opts_list, lbl_list, answer_list, pid_list in tqdm(train_loader):
            steps += 1            
            
            if args.no_meta:
                out = model(**inputs)
                loss = criterion(out, answer_list, pid_list)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # meta learning updates.
                loss_list = [None] * args.num_meta_updates

                for k in range(args.num_meta_updates):
                    out = model(**inputs)
                    loss = criterion(out, answer_list, pid_list)
                    anshead_optimizer.zero_grad()
                    grad = torch.autograd.grad(loss, anshead_parameters, allow_unused=True, retain_graph=True)
                    for (gr, pr) in zip(grad, anshead_parameters):
                        if gr is not None:
                            pr = pr - args.lr * gr
                    loss_list[k] = loss  # the last loss.
                meta_loss = loss_list[-1] / args.num_meta_updates
                optimizer.zero_grad()
                meta_loss.backward()
                optimizer.step()  # meta update.
            if loss.isnan():
                print('nan')
            print(im_path_list)
            print(loss.item())
            tot_loss += loss.item()

        tot_loss /= float(steps)
            
            
        #     if 'Clf' in args.model_name:
        #         loss = loss_func(outputs['logits'], labels) / grad_accumulation_steps
        #     else:
        #         loss = outputs.loss / grad_accumulation_steps
            
        #     accumulated_avg_loss += loss.item()
        #     loss.backward()
            
        #     logging_step = steps+(len_train*epoch)
        #     if steps % grad_accumulation_steps == 0:
                
        #         logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, training loss of previous {grad_accumulation_steps} batches: {accumulated_avg_loss:.8f}")
        #         train_loss_logger[logging_step] = accumulated_avg_loss
        #         accumulated_avg_loss = 0
        #         optimizer.step()
        #         optimizer.zero_grad()
                
        #     # Valid Loop bundle...
        #     if steps % valid_frequency_steps == 0:
                
        #         accumulated_valid_avg_loss = 0
        #         for v_batch in valid_loader:
        #             v_inputs, v_labels = v_batch
                    
        #             v_outputs = model(**v_inputs, labels=v_labels)
                    
        #             if 'Clf' in args.model_name:
        #                 valid_loss = loss_func(v_outputs['logits'], v_labels)
        #             else:
        #                 valid_loss = v_outputs.loss
        #             accumulated_valid_avg_loss += valid_loss.item()
                    
        #         accumulated_valid_avg_loss = accumulated_valid_avg_loss / len_valid
        #         valid_loss_logger[logging_step] = accumulated_valid_avg_loss
        #         logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, validation loss: {accumulated_valid_avg_loss:.8f}")
                
        #     # # LR Scheduler Update...
        #     # if steps % lr_update_frequency_steps == 0:
                
        #     #     past_lr = scheduler.get_lr()[-1]
        #     #     with warmup_scheduler.dampening():
        #     #         scheduler.step()
        #     #     current_lr = scheduler.get_lr()[-1]
        #     #     lr_logger[logging_step] = current_lr
        #     #     logger.info(f"Batch {steps}/{len_train} of epoch {epoch + 1}/{epochs}, lr updated: {past_lr:.12f} --> {current_lr:.12f}")            
                    
        # # 에폭별로 폴더를 만들기 위한 경로 설정
        # epoch_output_dir = os.path.join(save_folder, f"epoch_{epoch + 1}")
        # os.makedirs(epoch_output_dir, exist_ok=True)

        # # 에폭별로 모델과 프로세서를 저장
        # model.save_pretrained(epoch_output_dir)
        # processor.save_pretrained(epoch_output_dir)
        
        # # 학습 결과 JSON 결과 파일 저장
        # train_loss_save_path = os.path.join(args.save_root, args.save_folder, 'train_loss.json')
        # with open(train_loss_save_path,'w') as f:
        #     json.dump(train_loss_logger, f, ensure_ascii=False, indent=4)
        
        # valid_loss_save_path = os.path.join(args.save_root, args.save_folder, 'valid_loss.json')
        # with open(valid_loss_save_path,'w') as f:
        #     json.dump(valid_loss_logger, f, ensure_ascii=False, indent=4) 
            
        # # lr_logger_save_path = os.path.join(args.save_root, args.save_folder, 'lr_logger.json')
        # # with open(lr_logger_save_path,'w') as f:
        # #     json.dump(lr_logger, f, ensure_ascii=False, indent=4)       
        
        # # 학습 결과 Plot 시각화 파일 저장
        # loss_curve_path = os.path.join(args.save_root, args.save_folder, 'loss_curve.png')
        # plot_loss(train_loss_logger, valid_loss_logger, epoch+1, len_train, loss_curve_path)
        
        # # lr_curve_path = os.path.join(args.save_root, args.save_folder, 'lr_scheduler_curve.png')
        # # plot_lr_loss(lr_logger, epoch+1, len_train, lr_curve_path)
                
        # write_chat_template(processor, epoch_output_dir, logger)
                
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
        
        inputs, gt, im_name_list = batch
        
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        TP, ALL, result_dict = calc_value_acc(output_texts, gt, im_name_list, TP, ALL, result_dict)
        # TP, ALL, result_dict = calc_option_acc(output_texts, gt, im_name_list, TP, ALL, result_dict)

    result_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path}_{args.mode}_result_dict.json')
    with open(result_save_path,'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4) 

def trainer_clf_forward(args, model, processor, test_loader):
    
    model.eval()
    TP, ALL = 0, 0
    result_dict = dict()
    for batch in test_loader:
        
        inputs, gt, im_name_list = batch
        
        generated_ids = model(**inputs)['logits']
        output_texts = torch.argmax(generated_ids, 1)
        
        output_texts = [int(pred_max) for pred_max in output_texts]
        gt = [int(gt_label[0]) for gt_label in gt]
        
        TP, ALL, result_dict = calc_clf_value_acc(output_texts, gt, im_name_list, TP, ALL, result_dict)        

    result_save_path = os.path.join(args.save_root, f'{args.load_ckpt_path}_{args.mode}_result_dict.json')
    with open(result_save_path,'w') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4) 

def calc_option_acc(pred_list, label_list, im_name_list, TP, ALL, result_dict):
    
    for pred, label, img_name in zip(pred_list, label_list, im_name_list):
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
        
        result_dict[img_name] = {
            'Pred': pred,
            'Label': label,
            'hit': hit}
    
    save_pred = pred.replace('\n',' ')
    try: print(f"Accuracy = {TP}/{ALL} = {TP/ALL:.4f},  Pred: {save_pred},  Label: {label}")
    except: pass
    return TP, ALL, result_dict

def calc_value_acc(pred_list, label_list, im_name_list, TP, ALL, result_dict):
    
    for pred, label, img_name in zip(pred_list, label_list, im_name_list):
        eval_answer = pred.upper()
        
        find_answer = eval_answer.find(f'{label}')
    
        if find_answer>=0:
            TP += 1   
            hit = True
        else:
            hit = False    
        ALL += 1
        
        result_dict[img_name] = {
            'Pred': pred,
            'Label': label,
            'hit': hit}
    
    save_pred = pred.replace('\n',' ')
    try: print(f"Accuracy = {TP}/{ALL} = {TP/ALL:.4f},  Pred: {save_pred},  Label: {label}")
    except: pass
    return TP, ALL, result_dict

def calc_clf_value_acc(pred_list, label_list, im_name_list, TP, ALL, result_dict):
    
    for pred, label, img_name, in zip(pred_list, label_list, im_name_list):
    
        if pred == label:
            TP += 1   
            hit = True
        else:
            hit = False    
        ALL += 1
        
        result_dict[img_name] = {
            'Pred': pred,
            'Label': label,
            'hit': hit}
    
    try: print(f"Accuracy = {TP}/{ALL} = {TP/ALL:.4f},  Pred: {pred},  Label: {label}")
    except: pass
    return TP, ALL, result_dict