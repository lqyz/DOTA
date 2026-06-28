"""
BLOCK-DIAGONAL TEST-TIME ADAPTATION (BD-TTA)
=============================================
Pure unsupervised, BP-free TTA. Feature-dimension-aware adaptive drift.

D-small (e.g., CIFAR D=64): eta → 0, mu nearly frozen (safe anchor)
D-large (e.g., ImageNet D=2048): eta → 0.01, gentle mu drift
"""
import torch
import numpy as np

__all__ = ['BDTTA']


class BDTTA:
    def __init__(self, D, C, G=4, sigma=1.0, epsilon=1e-4,
                 alpha_slow=0.002, alpha_fast=0.02,
                 N_eff_slow=100, N_eff_fast=20,
                 W_init=None, device='cuda'):
        self.device = torch.device(device)
        self.C, self.D, self.G = C, D, G
        self.eta = 0.01 * (D / 2048.0)  # auto-scale: 64→0.0003, 2048→0.01
        self.slow = self._mk(C, D, G, sigma, epsilon, alpha_slow, N_eff_slow, W_init)
        self.fast = self._mk(C, D, G, sigma, epsilon, alpha_fast, N_eff_fast, W_init)

    def _mk(self, C, D, G, sigma, eps, alpha, Ne, W_init):
        B = D // G
        return {
            'C': C, 'D': D, 'G': G, 'B': B,
            'alpha': alpha, 'N_eff': Ne, 'eps': eps,
            'mu': W_init.clone().to(self.device) if W_init is not None
                  else torch.zeros(C, D, device=self.device),
            'mu_env': torch.zeros(D, device=self.device),
            'S': [sigma * torch.eye(B, device=self.device) for _ in range(G)],
            'L': [torch.inverse(sigma * torch.eye(B, device=self.device) + eps * torch.eye(B, device=self.device))
                  for _ in range(G)],
            'rng': [(i * B, (i + 1) * B) for i in range(G)],
        }

    def _fit_branch(self, br, x, y=None):
        x = x.float().to(self.device)
        br['mu_env'] = (1.0 - br['alpha']) * br['mu_env'] + br['alpha'] * x.mean(dim=0)
        xc = x - br['mu_env'].unsqueeze(0)

        if y is not None:
            y = y.float().to(self.device)
            w = y.sum(0); wx = y.T @ xc
            N = br['N_eff']
            new_mu = (wx + N * br['mu']) / (w.unsqueeze(1) + N)
            br['mu'] = br['mu'] + self.eta * (new_mu - br['mu'])

        N = br['N_eff']
        for g, (l, r) in enumerate(br['rng']):
            xg = xc[0, l:r]
            delta = torch.outer(xg, xg)
            br['S'][g] = (N * br['S'][g] + delta) / (N + 1.0)

    def _update_branch(self, br):
        for g in range(len(br['rng'])):
            Sg = br['S'][g]
            reg = (1.0 - br['eps']) * Sg + br['eps'] * torch.eye(br['B'], device=self.device)
            br['L'][g] = torch.inverse(reg)

    def _predict_branch(self, br, x):
        xd = x.float().to(self.device) - br['mu_env'].unsqueeze(0)
        scores = torch.zeros(1, br['C'], device=self.device)
        for g, (l, r) in enumerate(br['rng']):
            M = br['mu'][:, l:r].T
            W = br['L'][g] @ M
            bias = 0.5 * (M * W).sum(dim=0)
            scores += xd[:, l:r] @ W - bias
        return scores / br['G']

    def reset(self):
        for br in [self.slow, self.fast]:
            br['mu_env'].zero_()
            for g in range(len(br['rng'])):
                br['S'][g] = 1.0 * torch.eye(br['B'], device=self.device)
                br['L'][g] = torch.inverse(1.0 * torch.eye(br['B'], device=self.device)
                                            + br['eps'] * torch.eye(br['B'], device=self.device))
