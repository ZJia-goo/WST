#!/usr/bin/env python3

import argparse
import time
import yaml
import os
import logging
import numpy as np
import torch.utils.data
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast as amp_autocast
from timm.scheduler.scheduler_factory import CosineLRScheduler
from timm.data import resolve_data_config
from contextlib import suppress

from timm.models import create_model, safe_model_name
from timm.utils import random_seed, AverageMeter, accuracy, NativeScaler, ModelEmaV2
from timm.data import Mixup, FastCollateMixup
from timm.loss import SoftTargetCrossEntropy
from utils.utils import write, create_transform, create_loader

from data.vtab import VTAB
from models import vision_transformer
import utils.utils as util

torch.backends.cudnn.benchmark = False


def mark_trainable_parameters(model: nn.Module, model_type):
    for n, p in model.named_parameters():
        if 'adapter' in n or 'small_' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    if model_type == 'vit_base_patch16_224_in21k':
        model.head.weight.requires_grad = True
        model.head.bias.requires_grad = True
    else:
        raise NotImplementedError

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default=None, type=str, help='data dir')
parser.add_argument('--load_path', default=None, type=str, help='path for loading pretrained checkpoint')

parser.add_argument('--dataset', default='cifar_100', type=str,
                    choices=['cifar_100', 'dtd', 'patch_camelyon', 'eurosat', 'kitti', 'dmlab', 'caltech101',
                             'clevr_count', 'clevr_dist', 'diabetic_retinopathy', 'dsprites_loc', 'dsprites_ori',
                             'pets', 'flowers102', 'resisc45', 'smallnorb_azi', 'smallnorb_ele', 'sun397', 'svhn'])
parser.add_argument('--model', default='vit_base_patch16_224_in21k', type=str)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_size_test', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--warmup_epochs', type=int, default=10)

parser.add_argument('--r', type=int, default=2, help='r for low-rank feature transformations')
parser.add_argument('--scale', type=float, default=1.0, help='scale')
parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 5e-2)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate (default: 0.0)')
parser.add_argument('--mixup', type=float, default=0.0, help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0, help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--smoothing', type=float, default=0.0, help='Label smoothing (default: 0.0)')
parser.add_argument('--ema', default=False, action='store_true', help='EMA for model boost (default: False)')
parser.add_argument('--ema_decay', default=0.9998, type=float, help='EMA decay weights')

parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--prefetcher', default=False, action='store_true', help='prefetcher signal for data loading')
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')

args = parser.parse_args()

if not args.ema:
    args.ema_decay = None

# VTAB classes
if args.dataset == 'cifar_100':
    args.num_classes = 100
elif args.dataset == 'caltech101':
    args.num_classes = 102
elif args.dataset == 'dtd':
    args.num_classes = 47
elif args.dataset == 'flowers102':
    args.num_classes = 102
elif args.dataset == 'pets':
    args.num_classes = 37
elif args.dataset == 'svhn':
    args.num_classes = 10
elif args.dataset == 'sun397':
    args.num_classes = 397
elif args.dataset == 'patch_camelyon':
    args.num_classes = 2
elif args.dataset == 'eurosat':
    args.num_classes = 10
elif args.dataset == 'resisc45':
    args.num_classes = 45
elif args.dataset == 'diabetic_retinopathy':
    args.num_classes = 5
elif args.dataset == 'clevr_count':
    args.num_classes = 8
elif args.dataset == 'clevr_dist':
    args.num_classes = 6
elif args.dataset == 'dmlab':
    args.num_classes = 6
elif args.dataset == 'kitti':
    args.num_classes = 4
elif args.dataset == 'dsprites_loc':
    args.num_classes = 16
elif args.dataset == 'dsprites_ori':
    args.num_classes = 16
elif args.dataset == 'smallnorb_azi':
    args.num_classes = 18
elif args.dataset == 'smallnorb_ele':
    args.num_classes = 9
else:
    raise NotImplementedError

benchmark = 'VTAB'
dataset_func = VTAB
train_transform_type = 'VTAB'
test_transform_type = 'VTAB'
train_split = 'train_val'
test_split = 'test'

args.log_dir = os.path.join('checkpoint', args.model, benchmark, args.dataset,
                            'bs_{}_wd_{}_lr_{}_dp_{}_scale_{}_sed_{}_ema_{}_emadcy_{}_amp_{}_mixup_{}_cutmix_{}_smooth_{}_prefet_{}'
                            .format(args.batch_size, args.weight_decay, args.lr, args.drop_path, args.scale, args.seed, args.ema,
                                    args.ema_decay, args.amp, args.mixup, args.cutmix, args.smoothing, args.prefetcher))

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
args.log_file = os.path.join(args.log_dir, 'log.txt')

def main():
    write(args, args.log_file)
    random_seed(args.seed)
    if args.model == 'vit_base_patch16_224_in21k':
        model = create_model(args.model, num_classes=args.num_classes, checkpoint_path=args.load_path,
                             drop_path_rate=args.drop_path, r=args.r,
                             scale=args.scale, log_file=args.log_file)
    else:
        raise NotImplementedError

    mark_trainable_parameters(model, model_type=args.model)
    model.cuda()

    for n, p in model.named_parameters():
        if p.requires_grad:
            write('requires_grad : {}  with shape {}'.format(n, p.size()), args.log_file)

    best_acc = 0.0
    decay = []
    no_decay = []
    no_decay_name = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.endswith(".bias"):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            decay.append(param)

    params = [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': args.weight_decay}]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=0.0)



    write('number of params for requires grad in backbone: {}'.format({sum(p.numel() for n, p in model.named_parameters() if ((p.requires_grad) and ('head' not in n)))}), args.log_file)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=1e-8,
        cycle_decay=0.5,
        warmup_lr_init=1e-7,
        warmup_t=args.warmup_epochs)
    num_epochs = lr_scheduler.get_cycle_length() + args.warmup_epochs
    

    if args.ema:
        model_ema = ModelEmaV2(model, decay=args.ema_decay)
        write('initialize ema model', args.log_file)
    else:
        model_ema = None
        write('Dont use ema model', args.log_file)

    # create the train and eval datasets
    dataset_train = dataset_func(root=args.data_dir, dataset=args.dataset, split_=train_split,
                                 transform=create_transform(args.prefetcher, aug_type=train_transform_type))
    dataset_eval = dataset_func(root=args.data_dir, dataset=args.dataset, split_=test_split,
                                transform=create_transform(args.prefetcher, aug_type=test_transform_type))



    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.
    write('mixup_active : {}'.format(mixup_active), args.log_file)

    if mixup_active:
        mixup_args = dict(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, label_smoothing=args.smoothing,
                          num_classes=args.num_classes)
        if args.prefetcher:
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    write('collate_fn : {}'.format(collate_fn), args.log_file)
    write('mixup_fn : {}'.format(mixup_fn), args.log_file)

    loader_train = create_loader(
        dataset_train,
        batch_size=args.batch_size,
        is_training=True,
        re_prob=0.0,
        use_prefetcher=args.prefetcher,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        log_file=args.log_file
    )

    loader_eval = create_loader(
        dataset_eval,
        batch_size=args.batch_size_test,
        is_training=False,
        re_prob=0.,
        use_prefetcher=args.prefetcher,
        num_workers=args.num_workers,
        log_file=args.log_file
    )

    if mixup_active:
        criterion = SoftTargetCrossEntropy()
        write('Using SoftTargetCrossEntropy', args.log_file)
    else:
        criterion = nn.CrossEntropyLoss()
        write('Using CrossEntropyLoss', args.log_file)

    loss_scaler = NativeScaler() if args.amp else None
    autocast = amp_autocast if args.amp else suppress

    if (loss_scaler is not None) and (autocast == amp_autocast):
        write('Training in AMP', args.log_file)
    else:
        write('Training in FP32', args.log_file)



    for epoch in range(1, num_epochs + 1):

        train_one_epoch(epoch, model, loader_train,  optimizer, criterion, args,
                                        autocast=autocast, model_ema=model_ema, loss_scaler=loss_scaler,
                                        mixup_fn=mixup_fn)

        lr_scheduler.step(epoch)
        top1_acc_eval = validate(model, loader_eval, autocast=autocast)
        if best_acc < top1_acc_eval.avg:
            best_acc = top1_acc_eval.avg
            util.save(args.log_dir, model, str='best')
        if epoch == num_epochs:
            util.save(args.log_dir, model, str='final')

    if args.ema:
        assert model_ema is not None
        module_for_validate = model_ema.module
    else:
        module_for_validate = model

    top1_acc_eval = validate(module_for_validate, loader_eval, autocast=autocast)
    write('epoch: {}   eval_acc: {:.2f}'.format(epoch, top1_acc_eval.avg), log_file=args.log_file)


def train_one_epoch(epoch, model, loader,  optimizer, loss_fn, args, autocast, model_ema=None,
                    loss_scaler=None, mixup_fn=None):
    losses_m = AverageMeter()

    model.train()

    for batch_idx, (input, target) in enumerate(loader):

        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)

        with autocast():
            output = model(input)
            loss = loss_fn(output, target)

        losses_m.update(loss.item(), input.size(0))
        optimizer.zero_grad()

        if loss_scaler is not None:
            assert autocast == amp_autocast
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            assert autocast == suppress
            loss.backward()
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        lrl = [param_group['lr'] for param_group in optimizer.param_groups]
        lr = sum(lrl) / len(lrl)
    write('epoch: {}  '.format(epoch), log_file=args.log_file)


def validate(model, loader, autocast):
    top1_m = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):

            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            with autocast():

                output = model(input)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1_m.update(acc1.item(), output.size(0))

    write('Acc@1: {top1.avg:>7.4f}'.format(top1=top1_m), args.log_file)
    return top1_m


if __name__ == '__main__':
    main()
