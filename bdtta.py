"""
BLOCK-DIAGONAL TEST-TIME ADAPTATION (BD-TTA)
=============================================
A lightweight, BP-free TTA method using block-diagonal covariance
decomposition with environment decoupling.

Core contributions:
  1. Block-diagonal precision (8x memory save vs full matrix)
  2. Environment decoupling (mu_env EMA + centered features)
  3. Frozen semantic anchors (FC weights as class prototypes)
  4. Global shared covariance (robust to few-shot per class)
  5. Dual-path entropy routing (adaptive alpha selection)

Reference: "XXXX", 2026.
"""

import torch
import numpy as np

__all__ = ['BDTTA']


class BDTTA:
    """Block-Diagonal Test-Time Adaptation.
    
    Maintains per-class statistics (mu) and shared block-diagonal
    precision (Lambda) with online environment tracking.
    
    Args:
        D: Feature dimension (e.g., 64 for CIFAR, 2048 for ImageNet)
        C: Number of classes
        G: Number of diagonal blocks (default 4)
        sigma: Initial variance for Sigma (default 0.1)
        epsilon: Regularization for matrix inversion (default 1e-4)
        alpha_slow: EMA rate for slow environment tracking (default 0.002)
        alpha_fast: EMA rate for fast environment tracking (default 0.02)
        N_eff_slow: Effective window size for slow path (default 100)
        N_eff_fast: Effective window size for fast path (default 20)
        W_init: (C, D) tensor of pre-trained class prototypes [optional]
        device: torch device
    """
    
    def __init__(self, D, C, G=4, sigma=0.1, epsilon=1e-4,
                 alpha_slow=0.002, alpha_fast=0.02,
                 N_eff_slow=100, N_eff_fast=20,
                 W_init=None, device='cuda'):
        self.device = torch.device(device)
        self.C = C
        self.D = D
        self.G = G
        
        # ---- Initialize slow and fast branches ----
        self.slow  = self._make_branch(C, D, G, sigma, epsilon, alpha_slow, N_eff_slow, W_init)
        self.fast  = self._make_branch(C, D, G, sigma, epsilon, alpha_fast, N_eff_fast, W_init)
        
        # ---- Fusion weight (grows with sample count) ----
        self.omega = 0.01

    # ==================================================================
    #   Internal: branch creation
    # ==================================================================
    
    def _make_branch(self, C, D, G, sigma, eps, alpha, N_eff, W_init):
        B = D // G  # block size
        branch = {
            'C': C, 'D': D, 'G': G, 'B': B,
            'alpha': alpha, 'N_eff': N_eff, 'eps': eps,
            
            # Class prototypes: frozen FC weights (semantic anchor)
            'mu': W_init.clone().to(self.device) if W_init is not None
                  else torch.zeros(C, D, device=self.device),
            
            # Global environment mean (EMA tracked)
            'mu_env': torch.zeros(D, device=self.device),
            
            # Per-block global covariance (G x B x B)
            'S': [sigma * torch.eye(B, device=self.device).repeat(C, 1, 1) for _ in range(G)],
            
            # Per-block precision (inverse of regularized covariance)
            'L': [torch.inverse(sigma * torch.eye(B, device=self.device) + eps * torch.eye(B, device=self.device))
                  for _ in range(G)],
            
            # Block index ranges
            'rng': [(i * B, (i + 1) * B) for i in range(G)],
        }
        return branch

    # ==================================================================
    #   Core: online fit (pure unsupervised — no pseudo-labels)
    # ==================================================================
    
    def _fit_branch(self, br, x):
        """Update environment mean and global covariance from a single sample.
        
        Args:
            br: branch dict
            x: (1, D) or (B, D) feature tensor
        """
        x = x.float().to(self.device)
        
        # ---- 1st-order: environment mean (EMA) ----
        br['mu_env'] = (1.0 - br['alpha']) * br['mu_env'] + br['alpha'] * x.mean(dim=0)
        x_centered = x - br['mu_env'].unsqueeze(0)
        
        # ---- 2nd-order: per-block global covariance ----
        N = br['N_eff']
        for g, (l, r) in enumerate(br['rng']):
            xg = x_centered[0, l:r]          # current sample in block g
            delta = torch.outer(xg, xg)       # rank-1 outer product
            br['S'][g] = (N * br['S'][g] + delta) / (N + 1.0)

    # ==================================================================
    #   Core: precision update (block-wise inversion)
    # ==================================================================
    
    def _update_branch(self, br):
        """Recompute precision matrices from current covariance estimates."""
        for g in range(len(br['rng'])):
            Sg = br['S'][g].mean(dim=0)  # average over classes
            reg = (1.0 - br['eps']) * Sg + br['eps'] * torch.eye(br['B'], device=self.device)
            br['L'][g] = torch.inverse(reg)

    # ==================================================================
    #   Core: prediction (block-diagonal LDA score)
    # ==================================================================
    
    def _predict_branch(self, br, x):
        """Compute discriminant scores for all classes.
        
        Args:
            br: branch dict
            x: (1, D) feature tensor
        
        Returns:
            scores: (1, C) logit-level class scores
        """
        xd = x.float().to(self.device) - br['mu_env'].unsqueeze(0)
        scores = torch.zeros(1, br['C'], device=self.device)
        
        for g, (l, r) in enumerate(br['rng']):
            M = br['mu'][:, l:r].T          # (B, C)
            W = br['L'][g] @ M              # (B, C)
            bias = 0.5 * (M * W).sum(dim=0) # (C,)
            scores += xd[:, l:r] @ W - bias
        
        return scores / br['G']  # normalize by number of blocks

    # ==================================================================
    #   Public: single-step TTA (prediction + adaptation)
    # ==================================================================
    
    def step(self, x, model_logits):
        """Process one test sample: predict and update.
        
        Args:
            x: (1, D) feature vector from frozen backbone
            model_logits: (1, C) raw logits from the model's classifier
        
        Returns:
            final_logits: (1, C) enhanced logits (model + TTA)
        """
        with torch.no_grad():
            # ---- Dual-path prediction ----
            ss = self._predict_branch(self.slow, x)
            sf = self._predict_branch(self.fast, x)
            
            ls = model_logits.cpu() + self.omega * ss.cpu()
            lf = model_logits.cpu() + self.omega * sf.cpu()
            
            # ---- Entropy-based routing: pick the more confident path ----
            max_s = ls.softmax(dim=1).max(dim=1).values.item()
            max_f = lf.softmax(dim=1).max(dim=1).values.item()
            
            # ---- Co-evolution: both branches update with same data ----
            self._fit_branch(self.slow, x)
            self._fit_branch(self.fast, x)
            self._update_branch(self.slow)
            self._update_branch(self.fast)
            
            # ---- Grow fusion weight (bounded at 0.5) ----
            self.omega = min(0.01 * self.slow['N_eff'] / 10, 0.5)
            
            return ls if max_s > max_f else lf

    # ==================================================================
    #   Utility: reset statistics (for new domain/corruption)
    # ==================================================================
    
    def reset(self):
        """Reset environment and covariance statistics. Keeps mu (FC anchor)."""
        for br in [self.slow, self.fast]:
            br['mu_env'].zero_()
            B = br['B']
            for g in range(len(br['rng'])):
                br['S'][g] = 0.1 * torch.eye(B, device=self.device).repeat(br['C'], 1, 1)
                br['L'][g] = torch.inverse(0.1 * torch.eye(B, device=self.device) + br['eps'] * torch.eye(B, device=self.device))
        self.omega = 0.01
