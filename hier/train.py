
import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from pathlib import Path
from functools import partial
import PIL
import multiprocessing
from losses import *

from sampler import UniqueClassSempler, UniqueClassSampler, BalancedSampler
from helpers import get_emb, evaluate
from dataset import CUBirds, SOP, Cars
from dataset.Inshop import Inshop_Dataset
from models.model import init_model
        
def get_args_parser():
    parser = argparse.ArgumentParser('HIER', add_help=False)

    # Model parameters
    parser.add_argument('--model', default='resnet50', type=str,
        choices=['resnet50', 
                 'deit_small_distilled_patch16_224', 'vit_small_patch16_224', 'dino_vits16'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--image_size', type=int, default=224, help="""Size of Global Image""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--emb', default=128, type=int, help="""Dimensionality of output for [CLS] token.""")

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--clip_grad', type=float, default=0.1, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size', default=90, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=0, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=1e-5, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--fc_lr_scale", default=1, type=float)
    parser.add_argument("--warmup_epochs", default=1, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr_scale', type=float, default=0.1, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['sgd', 'adam', 'adamw',  'adamp', 'radam'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--load_from', default=None, help="""Path to load checkpoints to resume training.""")
    parser.add_argument('--pool', default = 'token', type=str, choices=['token', 'avg'], help = 'ViT Pooling')
    parser.add_argument('--lr_decay', default = None, type=str, help = 'Learning decay step setting')
    parser.add_argument('--lr_decay_gamma', default = None, type=float, help = 'Learning decay step setting')
    parser.add_argument('--resize_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--bn_freeze', type=bool, default=True)
    parser.add_argument('--use_lastnorm', type=bool, default=True)

    # Augementation parameters
    parser.add_argument('--global_crops_number', type=int, default=1, help="""Number of global
        views to generate. Default is to use two global crops. """)
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.14, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    
    # Hyperbolic MetricLearning parameters
    parser.add_argument('--IPC', type=int, default=2)
    parser.add_argument('--hyp_c', type=float, default=0.1)
    parser.add_argument('--clip_r', type=float, default=2.3)
    parser.add_argument('--save_emb', type=utils.bool_flag, default=False)
    parser.add_argument('--best_recall', type=int, default=0)
    parser.add_argument('--loss', default='PA', type=str, choices=['PA', 'MS', 'PNCA', 'SoftTriple', 'SupCon'])
    parser.add_argument('--cluster_start', default=0, type=int)
    parser.add_argument('--topk', default=30, type=int)
    parser.add_argument('--num_hproxies', default=512, type=int, help="""Dimensionality of output for [CLS] token.""")
    parser.add_argument('--lambda1', default=1.0, type=float, help="""loss weight for metric learning
        loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--lambda2', default=1.0, type=float, help="""loss weight for clustering loss over [CLS] tokens (Default: 1.0)""")
    parser.add_argument('--mrg', type=float, default=0.1)
    
    # Misc
    parser.add_argument('--dataset', default='CUB', type=str, 
                        choices=["SOP", "CUB", "Cars", "Inshop"], help='Please specify dataset to train')
    parser.add_argument('--data_path', default='/path/to/dataset', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default="./logs/", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--run_name', default="", type=str, help='Wandb run name')
    parser.add_argument('--saveckp_freq', default=40, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--eval_freq', default=1, type=int, help='Evaluation for every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    return parser

def train_one_epoch(model, cluster_loss, sup_metric_loss, get_emb_s, data_loader, optimizer, 
                    lr_schedule, epoch, fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    
    model.train()
    for it, (x, y, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):        
        it = len(data_loader) * epoch + it  
        for i, param_group in enumerate(optimizer.param_groups):
            lr = args.lr * (args.batch_size * utils.get_world_size()) / 180.
            param_group["lr"] = lr * param_group["lr_scale"] * lr_schedule[it]
            
            if epoch < args.warmup_epochs and "pretrained_params" in param_group["name"]:
                param_group["lr"] = 0
            elif "pretrained_params" in param_group["name"]:
                param_group["lr"] = lr * param_group["lr_scale"] * lr_schedule[it]
        
        x = torch.cat([im.cuda(non_blocking=True) for im in x])
        y = y.cuda(non_blocking=True).repeat(args.global_crops_number)
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            z = model(x) 

            if args.loss == 'SupCon' and args.IPC > 0:
                z = z.view(len(z) // args.IPC, args.IPC, args.emb)
            
            if world_size > 1:
                z = utils.all_gather(z, args.local_rank)
                y = utils.all_gather(y, args.local_rank)
            
            loss1 = sup_metric_loss(z, y) * args.lambda1
            if args.lambda2 > 0:
                loss2 = cluster_loss(z, y, args.topk) * args.lambda2
                if epoch < args.cluster_start:
                    loss2 = loss2 * 0
                loss = loss1 + loss2
            else:
                loss = loss1
        
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(False):
            if fp16_scaler is None:
                loss.backward()
                if args.clip_grad > 0:
                    param_norms = utils.clip_gradients_value(model, 10, losses=[cluster_loss, sup_metric_loss])
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if args.clip_grad > 0:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients_value(model, 10, losses=[cluster_loss, sup_metric_loss])
                    
                    
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                
        torch.cuda.synchronize()
        metric_logger.update(metric_loss=loss1.item())
        if args.lambda2 > 0:
            metric_logger.update(cluster_loss=loss2.item())
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
    metric_logger.synchronize_between_processes()
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    rh_model = 0
    if epoch % args.eval_freq == 0 and epoch >= args.warmup_epochs:
        if (args.dataset == "CUB" or args.dataset == "Cars") or epoch >= 40:
            rh_model = evaluate(get_emb_s, args.dataset, args.hyp_c)
            return_dict.update({"R@1_head": rh_model})

    if args.save_emb and epoch % args.eval_freq == 0 and args.best_recall < rh_model:
        ds_type = "gallery" if args.dataset == "Inshop" else "eval"
        x, y, index = get_emb_s(ds_type=ds_type)
        x, y, index  = x.float().cpu(), y.long().cpu(), index.long().cpu()
        torch.save((x, y, index), "{}/{}/{}_{}_eval_{}.pt".format(args.output_dir, args.dataset, args.model, args.run_name, epoch))

        x, y, index = get_emb_s(ds_type="train")
        x, y, index  = x.float().cpu(), y.long().cpu(), index.long().cpu()
        torch.save((x, y, index), "{}/{}/{}_{}_train_{}.pt".format(args.output_dir, args.dataset, args.model, args.run_name, epoch))
        
        x = cluster_loss.to_hyperbolic(cluster_loss.lcas).float().detach().cpu()
        y = (torch.ones(len(x)) * (y.max()+1)).long().cpu()
        torch.save((x,y), "{}/{}/{}_{}_lca_{}.pt".format(args.output_dir, args.dataset, args.model, args.run_name, epoch))
        
        if utils.is_main_process():
            print('Save embeding vectors')
        
    if epoch % args.eval_freq == 0 and epoch >= args.warmup_epochs:
        args.best_recall = rh_model
        
    return return_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser('HIER', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    if args.local_rank == 0:
        wandb.init(project="hyp_metric", name="{}_{}_{}".format(args.dataset, args.model, args.run_name), config=args)
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    world_size = utils.get_world_size()

    if args.model.startswith("vit"):
        mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    elif args.model == "bn_inception":
        mean_std = (104, 117, 128), (1,1,1)
    else:
        mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tr = utils.MultiTransforms(mean_std, model=args.model, view=args.global_crops_number)
    ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
    ds_class = ds_list[args.dataset]
    ds_train = ds_class(args.data_path, "train", train_tr)
    nb_classes = len(list(set(ds_train.ys)))
    if args.IPC > 0:
        sampler = UniqueClassSampler(ds_train.ys, args.batch_size, args.IPC, args.local_rank, world_size)
    else:
        sampler = torch.utils.data.DistributedSampler(ds_train, shuffle=True)
    data_loader = DataLoader(
        dataset=ds_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=True,
    )

    model = init_model(args)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=True, find_unused_parameters=(args.model == 'bn_inception'))
    cluster_loss = HIERLoss(args.num_hproxies, args.emb, mrg=args.mrg, hyp_c=args.hyp_c, clip_r=args.clip_r).cuda()
    if args.loss == 'MS':
        sup_metric_loss = MSLoss_Angle().cuda()
    elif args.loss == 'PA':
        sup_metric_loss = PALoss_Angle(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss == 'SoftTriple':
        sup_metric_loss = SoftTripleLoss_Angle(nb_classes=nb_classes, sz_embed = sz_embed).cuda()
    elif args.loss == 'PNCA':
        sup_metric_loss = PNCALoss_Angle(nb_classes=nb_classes, sz_embed = args.emb).cuda()
    elif args.loss =='SupCon':
        sup_metric_loss = SupCon(hyp_c=args.hyp_c, IPC=args.IPC).cuda()
    
    params_groups = utils.get_params_groups(model, sup_metric_loss, cluster_loss, fc_lr_scale=args.fc_lr_scale, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, eps=1e-4 if args.use_fp16 else 1e-8)  # to use with ViTs
    elif args.optimizer == "adamp":
        from adamp import AdamP
        optimizer = AdamP(params_groups)  # to use with ViTs
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(params_groups, eps=1e-4 if args.use_fp16 else 1e-8)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, momentum=0.9, lr=args.lr)  # lr is set by scheduler
        
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    if args.lr_decay == 'cosine':
        lr_schedule = utils.cosine_scheduler(1, args.min_lr_scale, args.epochs, len(data_loader))
    elif args.lr_decay is not None:
        lr_schedule = utils.step_scheduler(int(args.lr_decay), args.epochs, len(data_loader), gamma=args.lr_decay_gamma)
    else:
        lr_schedule = utils.step_scheduler(args.epochs, args.epochs, len(data_loader), gamma=1)

    get_emb_s = partial(
        get_emb,
        model=model.module,
        ds=ds_class,
        path=args.data_path,
        mean_std=mean_std,
        world_size=world_size,
        resize=args.resize_size,
        crop=args.crop_size,
    )

    cudnn.benchmark = True
    for epoch in range(args.epochs):
        if sampler is not None and args.IPC > 0:
            sampler.set_epoch(epoch)
        # ============ training one epoch ... ============
        train_stats = train_one_epoch(model, cluster_loss, sup_metric_loss, get_emb_s, data_loader, optimizer, 
                                      lr_schedule, epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'stduent': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
            
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if args.local_rank == 0:
            with (Path("{}/{}/{}_{}_log.txt".format(args.output_dir, args.dataset, args.model, args.run_name))).open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            wandb.log(train_stats, step=epoch)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

