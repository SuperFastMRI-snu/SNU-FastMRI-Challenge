import shutil
import numpy as np
import torch
import torch.nn as nn
import time
import requests
from tqdm import tqdm
from pathlib import Path
import copy
import pprint

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss, seed_fix
from utils.common.loss_function import SSIMLoss

# FIVarNet with channel&spatial-wise attention in UNet
# FIVarNet with channel&spatial-wise attention in UNet / without block attention
# FIVarNet with channel&spatial-wise attention in UNet / instead of block attention
from utils.model.tm_att_fi_varnet import TM_Att_FIVarNet, TM_Att_FIVarNet_n_b, TM_Att_FIVarNet_cs_n_b


from utils.mraugment.data_augment import DataAugmentor
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os, sys

def train_epoch(args, acc_steps, epoch, start_itr, model, data_loader, optimizer, LRscheduler, sum_loss, best_val_loss, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = sum_loss

    # iter 중간에 끊겼을 때 재학습을 위한 if문
    if start_itr < len_loader: 
        for iter, data in enumerate(data_loader):
            if iter < start_itr:
                continue
            mask, kspace, target, maximum, fname, _ = data
            mask = mask.cuda(non_blocking=True)
            kspace = kspace.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            maximum = maximum.cuda(non_blocking=True)

            # 파일이름에서 acceleration 계산한 뒤 각 itr별로 서로 다른 acc기반 attention 시행
            # FIVarNet_acc_fit model에만 사용
            acceleration = int(str(fname)[11: str(fname).rfind('_')])

            output = model(kspace, mask, acceleration)
            loss = loss_type(output, target, maximum)

            # gradient accumulation을 위해 acc_steps로 나누어서 back prop후 optimizer 사용
            loss /= acc_steps
            loss.backward()

            if ((iter + 1) % acc_steps == 0) or (iter + 1 == len_loader):
                optimizer.step()
                optimizer.zero_grad()

            loss *= acc_steps
            total_loss += loss.item()

            if iter % args.report_interval == 0:
                print(
                    f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                    f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                    f'Loss = {loss.item():.4g} '
                    f'Time = {time.perf_counter() - start_iter:.4f}s',
                )
                start_iter = time.perf_counter()

            # 한 epoch내에서 iter가 save_itr_interval의 배수가 되면 model.pt에 저장
            ## train에서 연결이 끊겼을 때 train.sh하면 가장 최근에 저장된 iter부터 train 재시작함
            if iter % args.save_itr_interval == 0:
                save_model(args, args.exp_dir, epoch, iter+1, model, optimizer, LRscheduler, total_loss, best_val_loss, False)

    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch, len_loader


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)

            # 파일이름에서 acceleration 계산한 뒤 각 itr별로 서로 다른 acc기반 attention 시행
            # FIVarNet_acc_fit model에만 사용
            acceleration = int(str(fnames)[11: str(fnames).rfind('_')])

            output = model(kspace, mask, acceleration)

            for i in range(output.shape[0]): # (KYG) output.shape[0] 은 1이다. 왜? for 문에서 data당 슬라이스 한 개씩만 들어오기 때문
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, itr, model, optimizer, LRscheduler, sum_loss, best_val_loss, is_new_best):
    # 한 epoch내에서 iter가 save_itr_interval의 배수일 때 model.pt에 저장
    # 한 epoch에서의 train이 끝나고 validate하기 전에 model.pt에 저장
    # 한 epoch에서의 train과 validate이 모두 끝났을 때 model.pt에 저장
    ## iter, epoch 단위에서 가장 최신의 model을 저장함
    ## 연결이 끊겼을 때 train.sh하면 model.pt를 불러와서 학습 재시작함
    torch.save(
        {
            'epoch': epoch,
            'itr': itr,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'LRscheduler': LRscheduler.state_dict(),
            'sum_loss': sum_loss,
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        }, # (KYG) dictionary type 으로 저장함.
        f=exp_dir / 'model.pt'
    )

    # 각 epoch마다 train과 validate이 끝난 model 개별 저장(단, 한 epoch 내에서 iter가 save_itr_interval의 배수가 되는 건 제외)
    ## 모든 epoch이 끝난 후 epoch별 model 비교를 위해 저장함
    if itr == 0:
        torch.save(
            {
                'epoch': epoch,
                'itr': itr,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'LRscheduler': LRscheduler.state_dict(),
                'sum_loss': 0,
                'best_val_loss': best_val_loss,
                'exp_dir': exp_dir
            },
            f=os.path.join(exp_dir, 'model'+str(epoch)+'.pt')
    )

    # 각 epoch마다 validate이 끝난 후 best_val_loss 가진 model이면 best_model 개별 저장
    ## 모든 epoch이 끝난 후 역대 best_model 비교를 위해 저장함
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', os.path.join(exp_dir, 'best_model'+str(epoch)+'.pt'))


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = TM_Att_FIVarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)

    model.to(device=device)

    loss_type = SSIMLoss().to(device=device)

    # optimizer, LRscheduler 설정
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    LRscheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_scheduler_patience, factor=args.lr_scheduler_factor, verbose=True)

    sum_loss = 0.
    best_val_loss = 1.
    start_epoch = 0
    start_itr = 0

    # train중이던 model을 사용할 경우
    MODEL_FNAMES = args.exp_dir / 'model.pt'
    if Path(MODEL_FNAMES).exists():
      pretrained = torch.load(MODEL_FNAMES)
      pretrained_copy = copy.deepcopy(pretrained['model'])
      for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (args.cascade <= int(layer.split('.',2)[1]) <=11):
            del pretrained['model'][layer]
      
      model.load_state_dict(pretrained['model'])
      optimizer.load_state_dict(pretrained['optimizer'])
      LRscheduler.load_state_dict(pretrained['LRscheduler'])

      sum_loss = pretrained['sum_loss']
      best_val_loss = pretrained['best_val_loss']

      start_epoch = pretrained['epoch']
      start_itr = pretrained['itr']
    
    loss_type = SSIMLoss().to(device=device)

    # -----------------
    # data augmentation
    # -----------------
    # initialize data augmentation pipeline
    current_epoch = start_epoch
    current_epoch_func = lambda: current_epoch
    augmentor = DataAugmentor(args, current_epoch_func)
    # ------------------

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, DataAugmentor = augmentor ,shuffle=True) #여기에 dataaugmentor를 argument 로 넣어줘야 함.
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args, DataAugmentor = augmentor)

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # current_epoch 업데이트
        current_epoch = epoch

        train_loss, train_time, end_itr = train_epoch(args, args.acc_steps, epoch, start_itr, model, train_loader, optimizer, LRscheduler, sum_loss, best_val_loss, loss_type)

        # 한 epoch에서의 train이 끝나고 validate하기 전에 model.pt에 저장
        ## validate에서 연결이 끊겼을 때 train.sh하면 해당 epoch의 train은 패스하고 validate부터 재시작함
        save_model(args, args.exp_dir, epoch, end_itr, model, optimizer, LRscheduler, 0, best_val_loss, False)
        start_itr = 0
        sum_loss = 0 
        
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        # val_loss를 바탕으로 LRscheduler step 진행, lr조정
        LRscheduler.step(val_loss)
        # last_lr = LRscheduler.get_last_lr()
        # print(f'Learning Rate: {last_lr}')

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        # 한 epoch에서의 train과 validate이 모두 끝났을 때 model.pt에 저장
        # 각 epoch마다 train과 validate이 끝난 model 개별 저장
        # 각 epoch마다 validate이 끝난 후 best_val_loss 가진 model이면 best_model 개별 저장
        ## validate의 결과로 나온 best_val_loss 최신화함
        save_model(args, args.exp_dir, epoch + 1, 0, model, optimizer, LRscheduler, 0, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )
        
        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
      
        #매 epoch마다 val reconstructions 저장
        start = time.perf_counter()
        save_reconstructions(reconstructions, args.val_dir, epoch + 1, targets=targets, inputs=inputs)
        print(
            f'Epoch {epoch + 1} val reconstructions saved!'
            f'ForwardTime = {time.perf_counter() - start:.4f}s',
        )

        """
        #스케줄러 조기종료 코드 - bad_epoch 분기점 지난 후에의 추이를 보기 위해 주석처리
        if scheduler.num_bad_epochs > scheduler.patience:
          print(f'Early stopping at epoch {epoch}...')
          break
        """
