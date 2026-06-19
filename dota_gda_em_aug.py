import random
import os

import argparse
from datetime import datetime
import torch
from tqdm import tqdm
import clip
from utils import *
from torch import nn
import logging
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

class DOTA(nn.Module):
    def __init__(self, cfg, input_shape, num_classes, clip_weights, streaming_update_Sigma=True):
        super(DOTA, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.streaming_update_Sigma = streaming_update_Sigma
        self.epsilon = cfg['epsilon']
        self.tau = cfg.get('tau', 10000.0)
        self.top_k = cfg.get('top_k', None)
        self.G = cfg.get('block_groups', 8)
        self.B = input_shape // self.G

        src_path = cfg.get('src_stats', None)
        if src_path is not None and os.path.exists(src_path):
            src = torch.load(src_path, map_location='cpu')
            self.mu = src['mu'].to(self.device)
            sigma2_init = src['sigma2']
            sigma2_blocks = sigma2_init.reshape(num_classes, self.G, self.B).mean(dim=2)
            self.Sigma = sigma2_blocks.unsqueeze(2).unsqueeze(3) * torch.eye(self.B, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.Sigma = self.Sigma.to(self.device)
        else:
            self.mu = clip_weights.T.to(self.device)
            self.Sigma = cfg['sigma'] * torch.eye(self.B, dtype=torch.float32).repeat(num_classes, self.G, 1, 1).to(self.device)
        self.c = torch.ones(num_classes, dtype=torch.float32).to(self.device)
        self.overall_Sigma = torch.mean(self.Sigma, dim=0)
        reg = (1 - self.epsilon) * self.overall_Sigma + self.epsilon * torch.eye(self.B, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.Lambda = torch.inverse(reg.double()).to(self.device).half()

    def fit(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            if self.top_k is not None:
                _, topk_idx = y.topk(self.top_k, dim=1)
                y_filt = torch.zeros_like(y)
                y_filt.scatter_(1, topk_idx, y.gather(1, topk_idx))
                y = y_filt / y_filt.sum(dim=1, keepdim=True).clamp(min=1e-8)

            sum_weights = torch.sum(y, dim=0)
            weighted_x = torch.matmul(y.T, x)
            new_mu = (weighted_x + self.c.unsqueeze(1) * self.mu) / (sum_weights.unsqueeze(1) + self.c.unsqueeze(1))
            new_c = self.c + sum_weights

            if self.streaming_update_Sigma:
                x_minus_mu = x.unsqueeze(1) - self.mu.unsqueeze(0)
                x_mm_blocks = x_minus_mu.reshape(x.size(0), self.num_classes, self.G, self.B)
                weighted_blocks = y.unsqueeze(2).unsqueeze(3) * x_mm_blocks
                delta_blocks = torch.einsum('bcgi,bcgj->cgij', weighted_blocks, x_mm_blocks)
                self.Sigma = (self.c[:, None, None, None] * self.Sigma + delta_blocks) / (self.c[:, None, None, None] + sum_weights[:, None, None, None].clamp(min=1e-8))

            self.overall_Sigma = torch.mean(self.Sigma, dim=0)
            self.mu = new_mu
            self.c = new_c

    def update(self):
        reg = (1 - self.epsilon) * self.overall_Sigma + self.epsilon * torch.eye(self.B, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.Lambda = torch.inverse(reg.double()).half()

    def predict(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            M = self.mu.transpose(1, 0).half()
            M_blocks = M.reshape(self.G, self.B, self.num_classes)
            W_blocks = torch.matmul(self.Lambda, M_blocks)
            c = 0.5 * torch.sum(torch.sum(M_blocks * W_blocks, dim=1), dim=0)

            X_blocks = X.half().reshape(1, self.G, self.B).permute(1, 0, 2).float()
            scores = torch.bmm(X_blocks, W_blocks.float()).squeeze(1).sum(dim=0) - c
            return scores.float()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='configs', help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--datasets', dest='datasets', default='I', type=str, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, default='ViT-B/16', choices=['ViT-B/16'], help='CLIP model backbone to use: ViT-B/16.')
    parser.add_argument('--log-path', dest='log_path', type=str, default='./log', help='Path to the log file.')
    args = parser.parse_args()
    return args


def run_test_dota(params, loader, clip_model, clip_weights, dota_model, logger):
    recent_sample_count = 1000
    fusion_accuracies = []
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits_aug(images, clip_model, clip_weights)
            pred, target = torch.tensor(pred).cuda(), target.cuda()

            dota_logits = dota_model.predict(image_features.mean(0).unsqueeze(0))

            dota_weights = torch.clamp(params['rho'] * dota_model.c.mean() / image_features.size(0), max=params['eta'])
            final_logits = clip_logits + dota_weights*dota_logits

            fusion_acc = cls_acc(final_logits, target)
            fusion_accuracies.append(fusion_acc)

            dota_model.fit(image_features, prob_map)
            dota_model.update()

            if (i + 1) % recent_sample_count == 0:
                recent_fusion_accuracy = sum(fusion_accuracies[-recent_sample_count:]) / recent_sample_count
                logger.info(
                    "Last {} samples' accuracies - Fusion: {:.2f}% | "
                    "Overall accuracies - Fusion: {:.2f}% ".format(
                        recent_sample_count, recent_fusion_accuracy,
                        sum(fusion_accuracies) / len(fusion_accuracies),
                    )
                )

        return {
            'overall_fusion_accuracy': sum(fusion_accuracies) / len(fusion_accuracies),
            'recent_fusion_accuracy': sum(fusion_accuracies[-recent_sample_count:]) / min(recent_sample_count, len(fusion_accuracies)),
        }


def main():
    args = get_arguments()
    config_path = args.config
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        setup_seeds(1)
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        backbone_safe = args.backbone.replace('/', '_')
        group_name = f"{backbone_safe}_{dataset_name}_{date}"
        logging.basicConfig(filename=os.path.join(args.log_path, group_name), level=logging.INFO, format='%(asctime)s %(message)s')
        logger = logging.getLogger()
        logger.info(f"Processing {dataset_name} dataset.")

        cfg = get_config_file(config_path, dataset_name)
        logger.info("\nRunning dataset configurations:")
        logger.info(cfg)

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)
        tensor_matrix = torch.full((clip_weights.shape[0], clip_weights.shape[1]), 0.001)
        dota_model = DOTA(cfg, input_shape=clip_weights.shape[0], num_classes=clip_weights.shape[1], clip_weights=tensor_matrix)
        dota_model.eval()

        acc = run_test_dota(cfg, test_loader, clip_model, clip_weights, dota_model, logger)
        logger.info(acc)


if __name__ == "__main__":
    main()
