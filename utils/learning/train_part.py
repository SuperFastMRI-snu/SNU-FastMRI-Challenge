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
from utils.model.varnet import VarNet
from utils.mraugment.data_augment import DataAugmentor
from utils.mraugment.data_transforms import VarNetDataTransform
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os, sys
import wandb
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=20, help='Report interval')
    parser.add_argument('-i', '--save-itr-interval', type=int, default=100, help='itr interval of model save')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/content/drive/MyDrive/Data/val/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/content/drive/MyDrive/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    add_augmentation_specific_args(parser)
    args = parser.parse_args()
    return args

def add_augmentation_specific_args(parser):
    parser.add_argument(
        '--aug_on', 
        default=True,
        help='This switch turns data augmentation on.',
        action='store_true'
    )
    # --------------------------------------------
    # Related to augmentation strenght scheduling
    # --------------------------------------------
    parser.add_argument(
        '--aug_schedule', 
        type=str, 
        default='exp',
        help='Type of data augmentation strength scheduling. Options: constant, ramp, exp'
    )
    parser.add_argument(
        '--aug_delay', 
        type=int, 
        default=0,
        help='Number of epochs at the beginning of training without data augmentation. The schedule in --aug_schedule will be adjusted so that at the last epoch the augmentation strength is --aug_strength.'
    )
    parser.add_argument(
        '--aug_strength', 
        type=float, 
        default=0.55, 
        help='Augmentation strength, combined with --aug_schedule determines the augmentation strength in each epoch'
    )
    parser.add_argument(
        '--aug_exp_decay', 
        type=float, 
        default=5.0, 
        help='Exponential decay coefficient if --aug_schedule is set to exp. 1.0 is close to linear, 10.0 is close to step function'
    )

    # --------------------------------------------
    # Related to interpolation 
    # --------------------------------------------
    parser.add_argument(
        '--aug_interpolation_order', 
        type=int, 
        default=1,
        help='Order of interpolation filter used in data augmentation, 1: bilinear, 3:bicubic. Bicubic is not supported yet.'
    )
    parser.add_argument(
        '--aug_upsample', 
        default=False,
        action='store_true',
        help='Set to upsample before augmentation to avoid aliasing artifacts. Adds heavy extra computation.',
    )
    parser.add_argument(
        '--aug_upsample_factor', 
        type=int, 
        default=2,
        help='Factor of upsampling before augmentation, if --aug_upsample is set'
    )
    parser.add_argument(
        '--aug_upsample_order', 
        type=int, 
        default=1,
        help='Order of upsampling filter before augmentation, 1: bilinear, 3:bicubic'
    )

    # --------------------------------------------
    # Related to transformation probability weights
    # --------------------------------------------
    parser.add_argument(
        '--aug_weight_translation', 
        type=float, 
        default=1.0, 
        help='Weight of translation probability. Augmentation probability will be multiplied by this constant'
    )
    parser.add_argument(
        '--aug_weight_rotation', 
        type=float, 
        default=1.0, 
        help='Weight of arbitrary rotation probability. Augmentation probability will be multiplied by this constant'
    )  
    parser.add_argument(
        '--aug_weight_shearing', 
        type=float,
        default=1.0, 
        help='Weight of shearing probability. Augmentation probability will be multiplied by this constant'
    )
    parser.add_argument(
        '--aug_weight_scaling', 
        type=float, 
        default=1.0, 
        help='Weight of scaling probability. Augmentation probability will be multiplied by this constant'
    )
    parser.add_argument(
        '--aug_weight_rot90', 
        type=float, 
        default=1.0, 
        help='Weight of probability of rotation by multiples of 90 degrees. Augmentation probability will be multiplied by this constant'
    )  
    parser.add_argument(
        '--aug_weight_fliph', 
        type=float,
        default=1.0, 
        help='Weight of horizontal flip probability. Augmentation probability will be multiplied by this constant'
    )
    parser.add_argument(
        '--aug_weight_flipv',
        type=float,
        default=1.0, 
        help='Weight of vertical flip probability. Augmentation probability will be multiplied by this constant'
    ) 

    # --------------------------------------------
    # Related to transformation limits
    # --------------------------------------------
    parser.add_argument(
        '--aug_max_translation_x', 
        type=float,
        default=0.125, 
        help='Maximum translation applied along the x axis as fraction of image width'
    )
    parser.add_argument(
        '--aug_max_translation_y',
        type=float, 
        default=0.125, 
        help='Maximum translation applied along the y axis as fraction of image height'
    )
    parser.add_argument(
        '--aug_max_rotation', 
        type=float, 
        default=180., 
        help='Maximum rotation applied in either clockwise or counter-clockwise direction in degrees.'
    )
    parser.add_argument(
        '--aug_max_shearing_x', 
        type=float, 
        default=15.0, 
        help='Maximum shearing applied in either positive or negative direction in degrees along x axis.'
    )
    parser.add_argument(
        '--aug_max_shearing_y', 
        type=float, 
        default=15.0, 
        help='Maximum shearing applied in either positive or negative direction in degrees along y axis.'
    )
    parser.add_argument(
        '--aug_max_scaling', 
        type=float, 
        default=0.25, 
        help='Maximum scaling applied as fraction of image dimensions. If set to s, a scaling factor between 1.0-s and 1.0+s will be applied.'
    )
    
    #---------------------------------------------------
    # Additional arguments not specific to augmentations 
    #---------------------------------------------------
    parser.add_argument(
        "--max_train_resolution",
        nargs="+",
        default=None,
        type=int,
        help="If given, training slices will be center cropped to this size if larger along any dimension.",
    )

    #---------------------------------------------------
    # Additional arguments for mask making 
    #---------------------------------------------------
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
        help="Number of center lines to use in mask",
    )
    return parser

def train_epoch(args, epoch, start_itr, model, data_loader, optimizer, LRscheduler, best_val_loss, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        # start_itr부터 iter 돌려서 학습
        if iter < start_itr:
            continue
        
        mask, kspace, target, maximum, _, _ = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

        # iter가 save_itr_interval의 배수가 되면 model.pt에 저장
        if iter % args.save_itr_interval == 0:
            save_model(args, args.exp_dir, epoch, iter+1, model, optimizer, LRscheduler, best_val_loss, False)

    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


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
            output = model(kspace, mask)

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


def save_model(args, exp_dir, epoch, itr, model, optimizer, LRscheduler, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'itr': itr,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'LRscheduler': LRscheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        }, # (KYG) dictionary type 으로 저장함.
        f=exp_dir / 'model.pt'
    )

    # 모든 epoch마다 model과 각종 모듈 저장 (단, epoch 내에서 iter가 save_itr_interval의 배수가 되는 건 제외)
    if itr == 0:
        torch.save(
            {
                'epoch': epoch,
                'itr': itr,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'LRscheduler': LRscheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'exp_dir': exp_dir
            },
            f=os.path.join(exp_dir, 'model'+str(epoch)+'.pt')
    )

    # 모든 best_model 저장
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


        
def train():
    # wandb run 하나 시작
    wandb.init(project = "varnet-sweep-test")
  
    args = parse()

    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.net_name = Path(str(wandb.config.cascade) +","+str(wandb.config.chans)+","+str(wandb.config.sens_chans))

    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result' / args.net_name / __file__
    args.val_loss_dir = '../result' / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    pprint.pprint(wandb.config) # cascade, chans, sens_chans 조합 출력

    model = VarNet(num_cascades=wandb.config.cascade, 
                   chans=wandb.config.chans, 
                   sens_chans=wandb.config.sens_chans)
    model.to(device=device)

    """
    # using pretrained parameter
    VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
    MODEL_FNAMES = "brain_leaderboard_state_dict.pt"
    if not Path(MODEL_FNAMES).exists():
        url_root = VARNET_FOLDER
        download_model(url_root + MODEL_FNAMES, MODEL_FNAMES)
    
    pretrained = torch.load(MODEL_FNAMES)
    pretrained_copy = copy.deepcopy(pretrained)
    for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (args.cascade <= int(layer.split('.',2)[1]) <=11):
            del pretrained[layer]
    model.load_state_dict(pretrained)
    """

    loss_type = SSIMLoss().to(device=device)

    # optimizer, LRscheduler 설정
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    LRscheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # LRscheduler에 관한 hyperparameter wandb의 config에 저장
    wandb.config.LRscheduler_patience = 3
    wandb.config.LRscheduler_factor = 0.5
    wandb.config.LRscheduler_mode = 'min'

    best_val_loss = 1.
    start_epoch = 0
    start_itr = 0

    # train중이던 model을 사용할 경우
    MODEL_FNAMES = args.exp_dir / 'model.pt'
    if Path(MODEL_FNAMES).exists():
      pretrained = torch.load(MODEL_FNAMES)
      pretrained_copy = copy.deepcopy(pretrained['model'])
      for layer in pretrained_copy.keys():
        if layer.split('.',2)[1].isdigit() and (wandconfig.cascade <= int(layer.split('.',2)[1]) <=11):
            del pretrained['model'][layer]
      
      model.load_state_dict(pretrained['model'])
      optimizer.load_state_dict(pretrained['optimizer'])
      LRscheduler.load_state_dict(pretrained['LRscheduler'])
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
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args)

    val_loss_log = np.empty((0, 2))
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        # current_epoch 업데이트
        current_epoch = epoch

        train_loss, train_time = train_epoch(args, epoch, start_itr, model, train_loader, optimizer, LRscheduler, best_val_loss, loss_type)

        # train_loss를 바탕으로 LRscheduler step 진행, lr조정
        LRscheduler.step(train_loss)
        lr = LRscheduler.get_last_lr()

        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, 0, model, optimizer, LRscheduler, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
        
        # wandb에 log
        wandb.log({"train_loss": train_loss, "valid_loss": val_loss})

        """
        #스케줄러 조기종료 코드 - bad_epoch 분기점 지난 후에의 추이를 보기 위해 주석처리
        if scheduler.num_bad_epochs > scheduler.patience:
          print(f'Early stopping at epoch {epoch}...')
          break
        """
