import random
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator

from models.transformer import Dasheng_Encoder
from models.sed_decoder import Decoder, TSED_Wrapper
from dataset.tsed import TSED_AS
from dataset.tsed_val import TSED_Val
from utils import load_yaml_with_includes, get_lr_scheduler, ConcatDatasetBatchSampler
from utils.data_aug import frame_shift, mixup, time_mask, feature_transformation
from val import val_psds


def parse_args():
    parser = argparse.ArgumentParser()

    # Config settings
    parser.add_argument('--config-name', type=str, default='configs/model.yml')

    # Training settings
    parser.add_argument("--amp", type=str, default='fp16')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument('--eval-every-step', type=int, default=5000)
    parser.add_argument('--save-every-step', type=int, default=5000)
    # parser.add_argument('--dataloader', type=str, default='EACaps')
    parser.add_argument("--logit-normal-indices", type=bool, default=False)

    # Log and random seed
    parser.add_argument('--random-seed', type=int, default=2024)
    parser.add_argument('--log-step', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='../logs/')
    parser.add_argument('--save-dir', type=str, default='../ckpts/')
    return parser.parse_args()


def setup_directories(args, params):
    args.log_dir = os.path.join(args.log_dir, params['model_name']) + '/'
    args.save_dir = os.path.join(args.save_dir, params['model_name']) + '/'

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)


def set_device(args):
    torch.set_num_threads(args.num_threads)
    if torch.cuda.is_available():
        args.device = 'cuda'
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        args.device = 'cpu'


if __name__ == '__main__':
    args = parse_args()
    params = load_yaml_with_includes(args.config_name)
    set_device(args)
    setup_directories(args, params)

    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # use accelerator for multi-gpu training
    accelerator = Accelerator(mixed_precision=args.amp,
                              gradient_accumulation_steps=params['opt']['accumulation_steps'],
                              step_scheduler_with_optimizer=False)

    train_set = TSED_AS(**params['data']['train_data'])
    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=params['opt']['batch_size'],
                              num_workers=args.num_workers)

    val_set = TSED_Val(**params['data']['val_data'])
    val_loader = DataLoader(val_set, num_workers=0, batch_size=1, shuffle=False)

    # test_set = TSED_Val(**params['data']['test_data'])
    # test_loader = DataLoader(val_set, num_workers=0, batch_size=1, shuffle=False)

    encoder = Dasheng_Encoder(**params['encoder']).to(accelerator.device)
    pretrained_url = 'https://zenodo.org/records/11511780/files/dasheng_base.pt?download=1'
    dump = torch.hub.load_state_dict_from_url(pretrained_url, map_location='cpu')
    model_parmeters = dump['model']
    # pretrained_url = 'https://zenodo.org/records/13315686/files/dasheng_audioset_mAP497.pt?download=1'
    # dump = torch.hub.load_state_dict_from_url(pretrained_url, map_location='cpu')
    # model_parmeters = dump
    encoder.load_state_dict(model_parmeters)

    decoder = Decoder(**params['decoder']).to(accelerator.device)

    model = TSED_Wrapper(encoder, decoder, params['ft_blocks'], params['frozen_encoder'])
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # model.load_state_dict(torch.load('../ckpts/TSED_AS_filter/20000.0.pt', map_location='cpu')['model'])

    if params['frozen_encoder']:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['opt']['learning_rate'],
            weight_decay=params['opt']['weight_decay'],
            betas=(params['opt']['beta1'], params['opt']['beta2']),
            eps=params['opt']['adam_epsilon'])
    else:
        optimizer = torch.optim.AdamW(
            [
                {'params': model.encoder.parameters(), 'lr': 0.1 * params['opt']['learning_rate']},
                {'params': model.decoder.parameters(), 'lr': params['opt']['learning_rate']}
            ],
            weight_decay=params['opt']['weight_decay'],
            betas=(params['opt']['beta1'], params['opt']['beta2']),
            eps=params['opt']['adam_epsilon'])

    lr_scheduler = get_lr_scheduler(optimizer, 'customized', **params['opt']['lr_scheduler'])

    strong_loss_func = nn.BCEWithLogitsLoss()

    model, optimizer, lr_scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_loader, val_loader)

    global_step = 0.0
    losses = 0.0

    if accelerator.is_main_process:
        model_module = model.module if hasattr(model, 'module') else model
        val_psds(model_module, val_loader, params, epoch='debug', split='val',
                 save_path=args.log_dir + 'output/', device=accelerator.device)

    for epoch in range(args.epochs):
        model.train()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(tqdm(train_loader)):
            with accelerator.accumulate(model):
                audio, cls, label, _ = batch
                mel = model.forward_to_spec(audio)

                # data aug
                mel, label = frame_shift(mel, label, params['net_pooling'])
                mel, label = time_mask(mel, label, params["net_pooling"],
                                       mask_ratios=params['data_aug']["time_mask_ratios"])
                mel, _ = feature_transformation(mel, **params['data_aug']["transform"])

                strong_pred = model(mel, cls)

                B, N, L = label.shape
                label = label.reshape(B * N, L)
                label = label.unsqueeze(1)

                loss = strong_loss_func(strong_pred, label)

                accelerator.backward(loss)

                # clip grad up
                if accelerator.sync_gradients:
                    if 'grad_clip' in params['opt'] and params['opt']['grad_clip'] > 0:
                        accelerator.clip_grad_norm_(model.parameters(),
                                                    max_norm=params['opt']['grad_clip'])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1/params['opt']['accumulation_steps']
                losses += loss.item()/params['opt']['accumulation_steps']

            if accelerator.is_main_process:
                if global_step % args.log_step == 0:
                    current_time = time.asctime(time.localtime(time.time()))
                    epoch_info = f'Epoch: [{epoch + 1}][{args.epochs}]'
                    batch_info = f'Global Step: {global_step}'
                    loss_info = f'Loss: {losses / args.log_step:.6f}'

                    # Extract the learning rate from the optimizer
                    lr = optimizer.param_groups[0]['lr']
                    lr_info = f'Learning Rate: {lr:.6f}'

                    log_message = f'{current_time}\n{epoch_info}    {batch_info}    {loss_info}    {lr_info}\n'

                    with open(args.log_dir + 'log.txt', mode='a') as n:
                        n.write(log_message)

                    losses = 0.0

            # check performance
            if (global_step + 1) % args.eval_every_step == 0:
                if accelerator.is_main_process:
                    model_module = model.module if hasattr(model, 'module') else model
                    val_psds(model_module, val_loader, params, epoch=global_step+1, split='val',
                             save_path=args.log_dir + 'output/', device=accelerator.device)
                    # save model
                    unwrapped_model = accelerator.unwrap_model(model)
                    accelerator.save({
                        "model": model.state_dict(),
                    }, args.save_dir + str(global_step+1) + '.pt')
                accelerator.wait_for_everyone()
                model.train()
