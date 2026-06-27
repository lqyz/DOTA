"""Unified TTA Architecture: Block-Diag + Env Decouple + Dual-Path Entropy Routing.

This single class implements all our contributions:
  Layer 1: Block-diagonal covariance (G blocks of B×B each)
  Layer 2: Environment decoupling (mu_env EMA tracking)
  Layer 3: Dual-path entropy routing (slow α=0.002 + fast α=0.02)
  Layer 4: N_eff fixed window (constant effective count, no freezing)
  Layer 5: Top-K soft label filtering (only confident classes updated)

Usage:
  model = DualPathTTA(D=64, C=10, G=4, device='cuda')
  for each sample x:
      pred = model.step(x, model_logits)
"""

import torch
import numpy as np

class DualPathTTA:
    def __init__(self, D, C, G=4, sigma=0.1, epsilon=1e-4,
                 alpha_slow=0.002, alpha_fast=0.02,
                 N_eff_slow=100, N_eff_fast=20,
                 soft_power=2.0, device='cuda'):
        self.device = device
        self.C = C
        self.max_entropy = np.log(C)
        self.soft_power = soft_power
        self._init_branch('slow', D, C, G, sigma, epsilon, alpha_slow, N_eff_slow)
        self._init_branch('fast', D, C, G, sigma, epsilon, alpha_fast, N_eff_fast)
        self.omega = 0.01

    def _init_branch(self, name, D, C, G, sigma, epsilon, alpha, N_eff):
        B = D // G
        branch = {
            'D': D, 'C': C, 'G': G, 'B': B,
            'alpha': alpha, 'N_eff': N_eff,
            'epsilon': epsilon,
            'mu': torch.zeros(C, D, device=self.device),
            'mu_env': torch.zeros(D, device=self.device),
            'Sigma': [sigma * torch.eye(B, device=self.device).repeat(C, 1, 1) for _ in range(G)],
            'Lambda': [torch.inverse(sigma * torch.eye(B, device=self.device) + epsilon * torch.eye(B, device=self.device)) for _ in range(G)],
            'rng': [(i*B, (i+1)*B) for i in range(G)],
        }
        setattr(self, name, branch)

    def _soft_weight(self, y):
        entropy = -(y * (y + 1e-8).log()).sum(-1)
        return (1.0 - entropy / self.max_entropy).clamp(min=0.0).pow(self.soft_power)

    def _fit_branch(self, br, x, y):
        x = x.float().to(self.device)
        gamma = self._soft_weight(y.to(self.device))
        y_w = (gamma.unsqueeze(1) * y.to(self.device)).float()
        if y_w.sum() < 1e-6:
            return

        br['mu_env'] = (1 - br['alpha']) * br['mu_env'] + br['alpha'] * x.mean(0)
        x_dec = x - br['mu_env'].unsqueeze(0)

        w = y_w.sum(0); wx = y_w.T @ x_dec; N = br['N_eff']
        br['mu'] = (wx + N * br['mu']) / (w.unsqueeze(1) + N)

        xmm = x_dec.unsqueeze(1) - br['mu'].unsqueeze(0)
        for g, (l, r) in enumerate(br['rng']):
            wm = y_w.unsqueeze(2) * xmm[..., l:r]
            delta = torch.einsum('bci,bcj->cij', wm, xmm[..., l:r])
            br['Sigma'][g] = (N * br['Sigma'][g] + delta) / (N + w[:, None, None].clamp(1e-8))

    def _update_branch(self, br):
        for g in range(br['G']):
            ov = br['Sigma'][g].mean(0)
            reg = (1 - br['epsilon']) * ov + br['epsilon'] * torch.eye(br['B'], device=self.device)
            br['Lambda'][g] = torch.inverse(reg)

    def _predict_branch(self, br, x):
        xd = x.float().to(self.device) - br['mu_env'].unsqueeze(0)
        sc = torch.zeros(1, br['C'], device=self.device)
        for g, (l, r) in enumerate(br['rng']):
            M = br['mu'][:, l:r].T; W = br['Lambda'][g] @ M
            sc += xd[:, l:r] @ W - 0.5 * (M * W).sum(0)
        return sc

    def step(self, x, model_logits):
        with torch.no_grad():
            sc_s = self._predict_branch(self.slow, x)
            sc_f = self._predict_branch(self.fast, x)

            ls = model_logits.cpu() + self.omega * sc_s.cpu()
            lf = model_logits.cpu() + self.omega * sc_f.cpu()

            ms = ls.softmax(1).max(1)[0].item()
            mf = lf.softmax(1).max(1)[0].item()

            if ms > mf:
                winning, wp = ls, ls.softmax(1)
            else:
                winning, wp = lf, lf.softmax(1)

            self._fit_branch(self.slow, x, wp)
            self._fit_branch(self.fast, x, wp)
            self._update_branch(self.slow)
            self._update_branch(self.fast)

            self.omega = min(0.01 * 100 / 10, 0.5)
            return winning
