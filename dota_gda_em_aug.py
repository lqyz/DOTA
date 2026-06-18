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
import bisect
from sortedcontainers import SortedList
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
        self.rank = cfg.get('rank', 4)
        self.beta_ema = cfg.get('beta_ema', 0.01)

        self.mu = clip_weights.T.to(self.device)
        self.c = torch.ones(num_classes, dtype=torch.float32).to(self.device)
        self.sigma2 = cfg['sigma'] * torch.ones(num_classes, input_shape, dtype=torch.float32).to(self.device)
        self.sigma2_global = cfg['sigma'] * torch.ones(input_shape, dtype=torch.float32).to(self.device)
        self.precision = (1.0 / (self.sigma2_global + self.epsilon)).unsqueeze(0).repeat(num_classes, 1).half()

        self.Sigma_global = cfg['sigma'] * torch.eye(input_shape, dtype=torch.float32).to(self.device)
        self.mu_global = torch.zeros(input_shape, dtype=torch.float32).to(self.device)
        self.U = torch.randn(input_shape, self.rank, dtype=torch.float32).to(self.device)
        self.U, _ = torch.linalg.qr(self.U)
        self.U_prec = torch.zeros(num_classes, self.rank, dtype=torch.float16).to(self.device)
        self.U_corr = torch.zeros(num_classes, dtype=torch.float16).to(self.device)

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
                sq_diff = x_minus_mu * x_minus_mu
                weighted_sq_diff = y.unsqueeze(2) * sq_diff
                delta_per_class = torch.sum(weighted_sq_diff, dim=0)
                self.sigma2 = (self.c.unsqueeze(1) * self.sigma2 + delta_per_class) / (self.c.unsqueeze(1) + sum_weights.unsqueeze(1).clamp(min=1e-8))

            self.mu = new_mu
            self.c = new_c
            self.sigma2_global = torch.sum(self.c.unsqueeze(1) * self.sigma2, dim=0) / self.c.sum()

            x_flat = x.mean(dim=0)
            self.mu_global = (1 - self.beta_ema) * self.mu_global + self.beta_ema * x_flat
            outer = torch.outer(x_flat - self.mu_global, x_flat - self.mu_global)
            self.Sigma_global = (1 - self.beta_ema) * self.Sigma_global + self.beta_ema * outer

    def update(self):
        alpha = self.c / (self.c + self.tau)
        sigma2_shrunk = (1 - alpha).unsqueeze(1) * self.sigma2_global.unsqueeze(0) + alpha.unsqueeze(1) * self.sigma2
        self.precision = (1.0 / (sigma2_shrunk + self.epsilon)).half()

        Delta = self.Sigma_global - torch.diag(torch.diag(self.Sigma_global))
        Delta = torch.clamp(Delta, min=-1.0, max=1.0)
        U_new = Delta @ self.U
        self.U, _ = torch.linalg.qr(U_new)

        P_half = self.precision.float()
        e_k = P_half * self.mu
        a_k = e_k @ self.U

        U_T = self.U.T
        P_weighted_U = P_half.unsqueeze(2) * self.U.unsqueeze(0)
        U_T_batch = U_T.unsqueeze(0).expand(self.num_classes, -1, -1)
        UTPU = torch.bmm(U_T_batch, P_weighted_U)
        M_k = torch.eye(self.rank, device=self.device).unsqueeze(0) + UTPU
        L_k = torch.linalg.cholesky(M_k)
        a_k_solved = torch.cholesky_solve(a_k.unsqueeze(2), L_k).squeeze(2)
        self.U_prec = a_k_solved.half()
        self.U_corr = (0.5 * torch.sum(a_k * a_k_solved, dim=1)).half()

    def predict(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            M = self.mu.half()
            W = self.precision * M
            c = 0.5 * torch.sum(M * W, dim=1)
            scores = torch.matmul(X.half(), W.T) - c.unsqueeze(0)

            e_x = X.float() * self.precision.float()
            a_x = e_x @ self.U
            scores = scores - torch.sum(a_x * self.U_prec.float(), dim=1) + self.U_corr.float()

            return scores.float()

def get_arguments():
    """Get arguments of the test-time adaptation."""
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
            pred, target, prop_entropy = torch.tensor(pred).cuda(), target.cuda(), get_entropy(loss, clip_weights)
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
