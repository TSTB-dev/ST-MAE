import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import random
import argparse

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.st_mae import SiameseTransitionMAE
from datasets import build_dataset, build_transforms
from util import AverageMeter
from mask import RandomMaskCollator

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def parse_args():
    parser = argparse.ArgumentParser(description='ST-MAE [Training]')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--class_name', type=str, required=True, help='Class name of the dataset')
    parser.add_argument('--num_normal_samples', type=int, default=-1, help='Number of normal samples')
    parser.add_argument('--stmae_model', type=str, default='stmae_base', help='ST-MAE model')
    parser.add_argument('--backbone_model', type=str, default='vgg19', help='Backbone model')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--split', type=str, default='train', help='Data split')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask ratio')
    parser.add_argument('--transform', type=str, default='default', help='Transform type')
    parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')    
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping')

    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--save_dir', type=str, default='weights', help='Save directory')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to resume weights')
    return parser.parse_args()
    
def train(args):
    assert args.split in ['train', 'test'], f"Invalid split: {args.split}"
    
    logger.info(args)
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Build dataset
    transform = build_transforms(args)
    dataset = build_dataset(args, transform=transform)
    dataset_name = args.data_root.split('/')[-1]
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of samples: {len(dataset)}")
    
    # Build model 
    model = SiameseTransitionMAE(
        input_res=args.img_size,
        patch_size=args.patch_size,
        backbone_model=args.backbone_model,
        stmae_model=args.stmae_model
    )
    model.to(args.device)
    if args.resume_path is not None:
        model.load_state_dict(torch.load(args.resume_path, weights_only=True))
    patch_size = model.patch_size
    
    # Create mask collator
    mask_collator = RandomMaskCollator(ratio=args.mask_ratio, input_size=model.feature_res, patch_size=patch_size)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mask_collator,
        pin_memory=True
    )
    total_iter = len(dataloader) * args.num_epochs
    
    # Optimizer
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Invalid optimizer: {args.optimizer}")

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iter)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    else:
        scheduler = None
    
    model.train()
    logger.info("Training Started🚀")
    for i in range(args.num_epochs):
        loss_meter = AverageMeter()
        mse_loss_meter = AverageMeter()
        cos_loss_meter = AverageMeter()
        trans_loss_meter = AverageMeter()
        for j, (batch, mask_indices) in enumerate(dataloader):
            x = batch["samples"].to(args.device)
            mask_indices = mask_indices.to(args.device)
            
            # Forward pass
            outputs = model(x, mask_indices)
            total_loss, mse_loss, cos_loss, trans_loss = outputs["total_loss"], outputs["mse_loss"], outputs["cos_loss"], outputs["trans_loss"]
            loss_meter.update(total_loss.item(), x.size(0))
            mse_loss_meter.update(mse_loss.item(), x.size(0))
            cos_loss_meter.update(cos_loss.item(), x.size(0))
            trans_loss_meter.update(trans_loss.item(), x.size(0))
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            if args.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            assert not torch.isnan(total_loss).any(), "Loss is NaN👾"
            
            if j % args.log_interval == 0:
                logger.info(f"Epoch: {i+1}/{args.num_epochs}, Iter: {j+1}/{len(dataloader)}, Loss: {loss_meter.avg:.4f},  MSE Loss: {mse_loss_meter.avg:.4f}, Cos Loss: {cos_loss_meter.avg:.4f}, Transition Loss: {trans_loss_meter.avg:.4f}")
        
        logger.info(
            f"Epoch: {i+1}/{args.num_epochs}, Loss: {loss_meter.avg:.4f}, MSE Loss: {mse_loss_meter.avg:.4f}, Cos Loss: {cos_loss_meter.avg:.4f}, Transition Loss: {trans_loss_meter.avg:.4f}"
        )
    # save model
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, f"{dataset_name}_{args.class_name}_{args.stmae_model}_{args.backbone_model}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved at {model_path}")
    
    logger.info("Training Finished🎉")

if __name__ == '__main__':
    args = parse_args()
    train(args)
        
    
    
