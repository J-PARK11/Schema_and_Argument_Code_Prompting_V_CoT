#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import os
import json

import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "1"

import warnings

warnings.filterwarnings("ignore")
import argparse
import copy
import time

import torch.nn.functional as F
from tqdm import tqdm

import SMART_official.build_vocab as vocab_utils
import SMART_official.data_loader as dl
import SMART_official.globvars as gv
import SMART_official.losses as losses
import SMART_official.net as net
import SMART_official.utils as utils


def reset_state(args):
    # global seed
    gv.seed = np.random.randint(10000) if args.seed == -1 else args.seed
    args.seed = gv.seed
    manualSeed = gv.seed  #
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    print("seed = %d" % (gv.seed))


def train(args, dataloader, im_backbone):
    criterion = losses.Criterion(args)
    if args.model_name in ["qwen2_vl_2b", "qwen2_vl_7b"]:
        for name, param in im_backbone.named_parameters():
            if 'visual' in name:
                param.requires_grad=False
        model = net.SMART_Qwen2_VL_Net(args, VL_backbone=im_backbone)
        print(f'\nRequire Grad Parameter numbers: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
        print(f'Freeze Parameter numbers: {sum(p.numel() for p in model.parameters() if not p.requires_grad)}')
    
    elif args.model_name == "flava":
        model = net.SMART_VL_Net(args, VL_backbone=im_backbone)
    
    else:
        model = net.SMART_Net(args, im_backbone=im_backbone)

    model = model.cuda()
    parameters = model.parameters()
    if not args.no_meta:
        anshead_parameters = list(model.ans_decoder.parameters())

    def normalize(err, pids):
        """this function divides the error by the gt number of classes for each puzzle."""
        pids = np.array(pids)
        for t in range(len(err)):
            err[t] = err[t] / gv.NUM_CLASSES_PER_PUZZLE[str(pids[t])]
        return err

    def get_result(out, ltype):
        if ltype == "classifier":
            pred_max = F.softmax(out, dim=1).argmax(dim=1).cpu()
        elif ltype == "regression":
            pred_max = torch.floor(out).long().cpu()[:, 0]
        else:
            raise "unknown loss type"

        return pred_max

    def save_model(args, net, acc, epoch, location):
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir(location):
            os.mkdir(location)
        loc = os.path.join(location, "ckpt_%s_Epoch_%s_%s.pt" % (args.model_name, str(epoch), args.seed))
        print("\nsaving checkpoint at %s" % (loc))
        # torch.save(state, loc)
        torch.save(net, loc)

    def train_loop(epoch, num_epochs, train_loader, optimizer, train_loss_dict, valid_acc_dict):
        model.train()
        tot_loss = 0.0
        accumulated_avg_loss = 0
        log_freq = 100
        grad_accumulation_steps = 4
        validation_freq = 500
        for i, (batch_input, im_path, q, _, a, av, pids) in tqdm(enumerate(train_loader)):
            
            save_idx = (epoch*len(train_loader)+i)
                   
            if args.no_meta:
                out = model(batch_input, pids)
                loss = criterion(out, av, pids) / grad_accumulation_steps
                accumulated_avg_loss += loss.item()
                # optimizer.zero_grad()
                loss.backward()
                # optimizer.step()
            else:
                # meta learning updates.
                loss_list = [None] * args.num_meta_updates

                for k in range(args.num_meta_updates):
                    out = model(batch_input, pids)
                    loss = criterion(out, av, pids) / grad_accumulation_steps
                    accumulated_avg_loss += loss.item()
                    anshead_optimizer.zero_grad()
                    grad = torch.autograd.grad(loss, anshead_parameters, allow_unused=True, retain_graph=True)
                    for (gr, pr) in zip(grad, anshead_parameters):
                        if gr is not None:
                            pr = pr - args.lr * gr
                    loss_list[k] = loss  # the last loss.
                meta_loss = loss_list[-1] / args.num_meta_updates
                # optimizer.zero_grad()
                meta_loss.backward()
                # optimizer.step()  # meta update.
            
            if (i+1) % grad_accumulation_steps == 0:
                accumulated_avg_loss = 0
                optimizer.step()
                optimizer.zero_grad()
                
            tot_loss += loss.item()
            
            if i % log_freq == 0:
                save_loss = float(loss.cpu())
                train_loss_dict[save_idx] = save_loss
                print(f'\n\nLog*******Epochs: {epoch+1}/{num_epochs} Train batch: {i+1}/{len(train_loader)}*******\n')
                print(f'Pids: {pids}')
                print(f'Im_path: {im_path}')
                print(f'Question: {q}')
                print(f'Pred: {out}')
                # print(f'Pred max: {torch.argmax(out,1)}')
                print(f'Label: {av}')
                print(f'train_loss: {save_loss:.6f}\n')
                print('***************\n')
                
            if (i+1) % validation_freq == 0:
                acc, err, oacc, puz_acc = val_loop(val_loader, model)
                valid_acc_dict[save_idx] = {"S_acc": float(acc), "O_acc": float(oacc)}
                
                with open(os.path.join(args.save_root, 'train_loss.json'),'w') as f:
                    json.dump(train_loss_dict, f, ensure_ascii=False, indent=4)
                    
                with open(os.path.join(args.save_root, 'valid_acc.json'),'w') as f:
                    json.dump(valid_acc_dict, f, ensure_ascii=False, indent=4)
                
                model.train()
            
            # if i==1: break

        tot_loss /= float(i)
        return tot_loss, train_loss_dict, valid_acc_dict

    def val_loop(val_loader, model):
        model.eval()
        acc_mean = 0
        cnt = 0
        err_mean = 0
        opt_mean = 0
        val_log_freq = 100
        puzzle_acc = {}
        with torch.no_grad():
            for i, (batch_input, im_path, q, o, a, av, pids) in tqdm(enumerate(val_loader)):
                
                # if i==10:break
                
                # q = q.cuda()
                o = np.array(o[0])
                out = model(batch_input, puzzle_ids=pids)
                av = av.cpu()
                pids = pids.cpu()
                
                if not args.monolithic:
                    upids = torch.unique(pids)
                    acc = 0
                    error = 0
                    opts_acc = 0
                    for t in upids:
                        idx = pids == t
                        tt = t.item()

                        if t not in gv.SEQ_PUZZLES:
                            pred_max = get_result(out[int(tt)], args.loss_type)
                            pacc = (pred_max == av[idx.cpu(), 0]).sum()
                            perror = normalize(np.abs(pred_max - av[idx, 0]), pids).sum()
                            oacc = utils.get_option_sel_acc(pred_max, o[idx], a[idx], av[idx], t).sum()
                        else:
                            pred_ans = []
                            pacc = 1
                            for k in range(gv.MAX_DECODE_STEPS):
                                pred_max = get_result(out[int(tt)][k], args.loss_type)
                                pred_ans.append(pred_max)
                                pacc = pacc * (pred_max == av[idx][:, k])
                            pacc = pacc.sum()
                            perror = 0
                            oacc = utils.get_option_sel_acc(np.column_stack(pred_ans), o[idx], a[idx], av[idx], t).sum()

                        if str(tt) in puzzle_acc.keys():
                            puzzle_acc[str(tt)][0] += pacc
                            puzzle_acc[str(tt)][1] += oacc
                            puzzle_acc[str(tt)][2] += idx.sum()
                        else:
                            puzzle_acc[str(tt)] = [pacc, oacc, idx.sum()]
                        # we use the ansewr value here.
                        opts_acc += oacc
                        acc += pacc
                        error += perror
                
                else:  # for monolothic architecture, i.e. using only one output head (e.g., in puzzle/FS split)
                    av = av[:, 0]
                    if args.loss_type == "classifier":
                        pred = F.softmax(out, dim=1)
                        pred_max = pred.argmax(dim=1).cpu()
                    elif args.loss_type == "regression":
                        pred_max = torch.floor(out).long().cpu()

                    acc = (pred_max == av).float().sum()
                    opt = torch.tensor(utils.get_option_sel_acc(pred_max, o, a, av, -1))
                    opts_acc = opt.sum()
                    error = normalize(torch.abs(pred_max - av).float(), pids).sum()

                    # compute accuracy per puzzle.()
                    for t in [int(s) for s in pids]:
                        if str(t) in puzzle_acc.keys():
                            puzzle_acc[str(t)][0] += (pred_max == av)[pids == t].sum()
                            puzzle_acc[str(t)][1] += opt[pids == t].sum()
                            puzzle_acc[str(t)][2] += (pids == t).sum()
                        else:
                            puzzle_acc[str(t)] = [
                                (pred_max == av)[pids == t].sum(),
                                opt[pids == t].sum(),
                                (pids == t).sum(),
                            ]

                if i % val_log_freq == 0:
                    print(f'\n\nLog*******Valid batch: {i+1}/{len(val_loader)}*******\n')
                    print(f'Pids: {pids}')
                    print(f'Im_path: {im_path}')
                    print(f'Question: {q}')
                    print(f'Pred: {out}')
                    # print(f'Pred max: {torch.argmax(out,1)}')
                    print(f'Label: {av}')
                    try:
                        print(f'S_acc: {acc_mean / float(cnt)}')
                        print(f'O_acc: {opt_mean / float(cnt)}\n')
                    except:
                        pass
                    print('***************\n')                
                
                opt_mean += opts_acc
                acc_mean += acc
                err_mean += error
                cnt += len(av)

        acc_mean = float(acc_mean)
        opt_mean = float(opt_mean)
        err_mean = float(err_mean)
        
        return acc_mean / float(cnt), err_mean / float(cnt), opt_mean / float(cnt), puzzle_acc

    def test_loop(test_loader, model):
        acc, err, opt, puzzle_acc = val_loop(test_loader, model)
        utils.print_puzz_acc(args, puzzle_acc, log=True)
        print(
            "\n***** Final Test Performance: S_acc = %0.2f O_acc = %0.2f Prediction Variance = %0.2f "
            % (acc * 100, opt * 100, err)
        )

    if args.test:
        model = net.load_pretrained_models(args, args.model_name, model=model)
        test_loop(dataloader["test"], model)
        return

    lr_inflation_rate = 1000
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam([{'params': model.VL_backbone.parameters(), 'lr': args.lr},
                                      {'params': model.qv_fusion.parameters(), 'lr': args.lr*lr_inflation_rate},
                                      {'params': model.ans_decoder.parameters(), 'lr': args.lr*lr_inflation_rate}], betas=(0.9, 0.99), weight_decay=0.9)
        if not args.no_meta:
            anshead_optimizer = torch.optim.Adam(anshead_parameters, lr=args.lr*lr_inflation_rate, betas=(0.9, 0.99), weight_decay=0.9)
    else:
        optimizer = torch.optim.SGD(parameters, lr=args.lr, weight_decay=0.9)
        if not args.no_meta:
            anshead_optimizer = torch.optim.SGD(anshead_parameters, lr=args.lr, weight_decay=0.9)

    train_loader = dataloader["train"]
    val_loader = dataloader["valid"]
    test_loader = dataloader["test"]

    # training loop
    best_model = None
    best_acc = 0
    no_improvement = 0
    num_thresh_epochs = 20
    train_loss_dict = {}
    valid_acc_dict = {}
    
    # stop training if there is no improvement after this.
    print("starting training...")
    for epoch in range(args.num_epochs):
        tt = time.time()
        model.train()
        loss, train_loss_dict, valid_acc_dict = train_loop(epoch, args.num_epochs, train_loader, optimizer, train_loss_dict, valid_acc_dict)
        tt = time.time() - tt

        if epoch % 1 == 0:
            model.eval()
            acc, err, oacc, puz_acc = val_loop(val_loader, model)
            if acc >= best_acc:
                print(f'New Best Model: {best_acc:.6f} --> {acc:.6f} in epoch {epoch+1}')
                best_epoch = epoch
                best_acc = acc
                best_model = copy.deepcopy(model)
                save_model(args, best_model, acc, epoch+1, args.location)
                no_improvement = 0
                del best_model
            else:
                no_improvement += 1
                if no_improvement > num_thresh_epochs:
                    print("no training improvement... stoppsing the training.")
                    utils.print_puzz_acc(args, puz_acc, log=args.log)
                    break
            # if epoch % args.log_freq == 0:
            print(
                "%d) Time taken=%f Epoch=%d Train_loss = %f Valid S_acc = %f Valid O_acc=%f Variance = %f Best Valid S_acc (epoch) = %f (%d)\n"
                % (gv.seed, tt, epoch, loss, acc * 100, oacc * 100, err, best_acc * 100, best_epoch)
            )
            utils.print_puzz_acc(args, puz_acc, log=args.log)

        with open(os.path.join(args.save_root, 'train_loss.json'),'w') as f:
            json.dump(train_loss_dict, f, ensure_ascii=False, indent=4)
            
        with open(os.path.join(args.save_root, 'valid_acc.json'),'w') as f:
            json.dump(valid_acc_dict, f, ensure_ascii=False, indent=4)
        
        # if epoch % args.log_freq == 0:
        #     acc, err, oacc, puz_acc = val_loop(test_loader, model)
        #     print(
        #         "puzzles %s: Test Set: s_acc/o_acc/var = %f/%f/%f (%d)"
        #         % (args.puzzles, acc * 100, oacc * 100, err, best_epoch)
        #     )

    test_loop(test_loader, best_model)
    print('\n***** Single GPU Official Train.py Complet *****')


def get_data_loader(args, split, preprocess, batch_size=100, shuffle=True, num_workers=6, pin_memory=True):
    from functools import partial
    if split == "train":
        dataset = dl.SMART_TrainData(args, split)
        collate_fn = partial(dl.SMART_Qwen2_VL_collate_fn, preprocess, device='cuda')
    else:
        dataset = dl.SMART_ValData(args, split)
        collate_fn = partial(dl.SMART_Qwen2_VL_collate_fn, preprocess, device='cuda')
        batch_size=1
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return data_loader


if __name__ == "__main__":
    
    device = "cuda"

    parser = argparse.ArgumentParser(description="SMART dataset")
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
    parser.add_argument("--model_name", type=str, help="model to use resnet50/resnet18/...")
    parser.add_argument("--seed", type=int, default=-1, help="seed to use")
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
    reset_state(args)
    torch.multiprocessing.set_start_method('spawn')
    
    args.puzzle_ids_str, args.puzzle_ids = utils.get_puzzle_ids(args)
    args.location = os.path.join(args.save_root, "checkpoints")
    args.log_path = os.path.join(args.save_root, "log")
    
    gv.NUM_CLASSES_PER_PUZZLE = utils.get_puzzle_class_info(
        args
    )  # initialize the global with the number of outputs for each puzzle.

    vocab = vocab_utils.process_text_for_puzzle(args)
    if args.vocab_path == "none":
        args.vocab_path = os.path.join(args.save_root, "vocab_puzzle_" + args.puzzle_ids_str + ".pkl")

    im_backbone, preprocess = net.load_pretrained_models(args, args.model_name, model=None)
    args.preprocess = preprocess

    train_loader = get_data_loader(
        args, "train", preprocess, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = get_data_loader(args, "val", preprocess, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = get_data_loader(args, "test", preprocess, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    dataloader = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }

    utils.backup_code_and_start_logger(args, args.log_path, args.seed)

    print(args)
    print("num_puzzles=%d" % (len(args.puzzle_ids)))

    train(args, dataloader, im_backbone)
