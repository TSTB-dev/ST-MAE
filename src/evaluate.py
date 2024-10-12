import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import random
import argparse

import torch
from torch.nn import functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from models.st_mae import SiameseTransitionMAE
from datasets import build_dataset, build_transforms, EvalDataLoader
from mask import RandomMaskCollator
from util import AverageMeter, gaussian_kernel, pidx_to_pmask

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def parser_args():
    parser = argparse.ArgumentParser(description='ST-MAE [Evaluation]')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--class_name', type=str, required=True, help='Class name of the dataset')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--backbone_model', type=str, default='vgg19', help='Backbone model')
    parser.add_argument('--stmae_model', type=str, default='stmae_base', help='ST-MAE model')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--num_masks', type=int, default=1, help='Number of masks')
    parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask ratio')
    parser.add_argument('--weights_path', type=str, default=None, help='Path to weights')
    parser.add_argument('--gaussian_filter', action='store_true', default=False, help='Apply Gaussian filter')
    parser.add_argument('--gaussian_sigma', type=float, default=4, help='Gaussian sigma')
    parser.add_argument('--gaussian_ksize', type=int, default=7, help='Gaussian kernel size')
    parser.add_argument('--transform', type=str, default='default', help='Transform type')
    parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()
    return args

def compute_err_map(org_features, recon):
    """Coompute Error map between original features and reconstructed features
    Args:
        org_features (tensor): Original features. Shape: (B, C, H', W')
        recon (tensor): Reconstructed features. Shape: (B, C, H', W')
    Returns:
        err_map (tensor): Error map. Shape: (B, H', W')
    """
    err_l2 = F.mse_loss(org_features, recon, reduction='none')  # (B, C, H', W')
    err_l2 = torch.mean(err_l2, dim=1)  # (B, H', W')
    
    err_cos = 1 - F.cosine_similarity(org_features, recon, dim=1)  # (B, H', W')
    assert torch.all(err_cos > 0), f"Cosine similarity should be positive: {err_cos}"
    assert torch.all(err_l2 > 0), f"MSE loss should be positive: {err_l2}"
    err_map = err_l2 * err_cos  # (B, H', W')
    return err_map

def calculate_mask_coverage(mask_batch, h, w):
    """Calculate mask coverage. 

    Args:
        mask_batch (tensor): Indices of masked patches. Shape: (B, N)
        h (int): Height of the feature map
        w (int): Width of the feature map
    Returns:
        mask_coverage (float): Mask coverage
    """
    mask = pidx_to_pmask(mask_batch, h, w)  # (B, H, W)
    mask_or = torch.any(mask, dim=0).float()  # (H, W)
    mask_coverage = torch.mean(mask_or)  # scalar
    return mask_coverage  

def gaussian_filter(err_map, sigma=1.4, ksize=7):
    """Apply Gaussian filter to the error map

    Args:
        err_map (tensor): Error map. Shape: (B, H, W)
        sigma (float, optional): Standard deviation of the Gaussian filter. Defaults to 1.4.
        ksize (int, optional): Kernel size of the Gaussian filter. Defaults to 7.
    Returns:
        err_map (tensor): Error map after applying Gaussian filter, Shape: (B, H, W)
    """
    err_map = err_map.detach().cpu()
    kernel = gaussian_kernel(ksize, sigma) 
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(err_map.device)  # (1, 1, ksize, ksize)
    padding = ksize // 2
    err_map = F.pad(err_map, (padding, padding, padding, padding), mode='reflect')
    err_map = F.conv2d(err_map.unsqueeze(1), kernel, padding=0).squeeze(1)
    return err_map

def evaluate(args):
    assert args.weights_path is not None, 'Please specify the path to the weights'
    args.num_normal_samples = -1
    logger.info(args)
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Build Model
    model = SiameseTransitionMAE(
        input_res=args.img_size, \
        patch_size=args.patch_size, \
        backbone_model=args.backbone_model, \
        stmae_model=args.stmae_model
    )
    model.to(args.device)
    h, w = model.feature_res // model.patch_size, model.feature_res // model.patch_size
    
    # Load weights
    model.load_state_dict(torch.load(args.weights_path, weights_only=True))
    
    # Build dataset
    args.split = 'test'
    transform = build_transforms(args)
    dataset = build_dataset(args, transform=transform)
    mask_collator = RandomMaskCollator(ratio=args.mask_ratio, input_size=model.feature_res, patch_size=args.patch_size)
    dataloader = EvalDataLoader(dataset, args.num_masks, collate_fn=mask_collator)
    logger.info(f'Dataset: {args.class_name}')
    logger.info(f'Number of samples: {len(dataset)}')
    
    # Evaluation
    model.eval()
    model.to(args.device)
    
    logger.info(f"Evalution Started🚀")
    loss_meter = AverageMeter()
    trans_loss_meter = AverageMeter()
    mask_coverage_meter = AverageMeter()
    
    results = {
        "err_maps": [],
        "filenames": [],
        "cls_names": [],
        "labels": [],
        "anom_types": []
    }
    for i, (batch, mask_indices) in enumerate(dataloader):
        images = batch["samples"].to(args.device)  # (B, C, H, W)
        results["labels"].append(batch["labels"][0].item())
        results["anom_types"].append(batch["anom_type"][0])
        results["filenames"].append(batch["filenames"][0])
        results["cls_names"].append(batch["clsnames"][0])
        
        mask_coverage = calculate_mask_coverage(mask_indices, h, w)
        mask_coverage_meter.update(mask_coverage, 1)
        
        mask_indices = mask_indices.to(args.device)
        
        with torch.no_grad():
            outputs = model(images, mask_indices)
            loss, recon_features, org_features, trans_loss = outputs["total_loss"], outputs["recon_features"], outputs["org_features"], outputs["trans_loss"]
            trans_loss_meter.update(trans_loss.item(), images.size(0))
            loss_meter.update(loss.item(), images.size(0))
            err_map = compute_err_map(org_features, recon_features)
            
            if args.gaussian_filter:
                err_map = gaussian_filter(err_map, sigma=args.gaussian_sigma, ksize=args.gaussian_ksize)
            err_map = torch.mean(err_map, dim=0)  # (H, W)
            results["err_maps"].append(err_map)
        
        if i % args.log_interval == 0:
            logger.info(f'Iter: [{i}/{len(dataloader)}]\t'
                        f'Loss: {loss_meter.avg:.4f}\t'
                        f'Transition Loss: {trans_loss_meter.avg:.4f}\t'
                        f'Mask coverage: {mask_coverage_meter.avg:.4f}')
    
    logger.info(f'Loss: {loss_meter.avg:.4f}\t'
                f'Transition Loss: {trans_loss_meter.avg:.4f}\t'
                f'Mask coverage: {mask_coverage_meter.avg:.4f}')
    
    # Calculate metrics
    global_err_scores = [torch.max(err_map) for err_map in results["err_maps"]]
    global_err_scores = torch.stack(global_err_scores).cpu().numpy()
    
    auc = roc_auc_score(results["labels"], global_err_scores)
    logger.info(f'auROC: {auc:.4f} on {args.class_name}')
    
    # Calculate the auROC score for each anomaly type
    unique_anom_types = list(sorted(set(results["anom_types"])))
    normal_indices = [i for i, x in enumerate(results["anom_types"]) if x == "good"]
    for anom_type in unique_anom_types:
        if anom_type == "good":
            continue
        anom_indices = [i for i, x in enumerate(results["anom_types"]) if x == anom_type]
        normal_scores = global_err_scores[normal_indices]
        anom_scores = global_err_scores[anom_indices]
        scores = np.concatenate([normal_scores, anom_scores])
        labels = [0] * len(normal_scores) + [1] * len(anom_scores)
        auc = roc_auc_score(labels, scores)
        logger.info(f'auROC: {auc:.4f} on {anom_type}')
    logger.info(f'Evaluation Finished🎉')
            
if __name__ == '__main__':
    args = parser_args()
    evaluate(args)