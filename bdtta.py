"""
BLOCK-DIAGONAL TEST-TIME ADAPTATION (BD-TTA)
=============================================
A lightweight, BP-free TTA method using block-diagonal covariance
decomposition with environment decoupling.
"""

import torch
import numpy as np

__all__ = ['BDTTA']


class BDTTA:
    def __init__(self, D, C, G=4, sigma=0.1, epsilon=1e-4,
                 alpha_slow=0.002, alpha_fast=0.02,
                 N_eff_slow=100, N_eff_fast=20,
                 W_init=None, device='cuda'):
        self.device = torch.device(device)
        self.C = C
        self.D = D
        self.G = G

        # ---- 初始化慢速与快速双分支 ----
        self.slow  = self._make_branch(C, D, G, sigma, epsilon, alpha_slow, N_eff_slow, W_init)
        self.fast  = self._make_branch(C, D, G, sigma, epsilon, alpha_fast, N_eff_fast, W_init)

        # ---- 修正：自适应增量刚性增益，确保在轻量特征空间具有绝对干预话语权 ----
        self.omega = 0.2

    def _make_branch(self, C, D, G, sigma, eps, alpha, N_eff, W_init):
        B = D // G  # 子块分块大小
        branch = {
            'C': C, 'D': D, 'G': G, 'B': B,
            'alpha': alpha, 'N_eff': N_eff, 'eps': eps,

            # 语义核心锚点死锁
            'mu': W_init.clone().to(self.device) if W_init is not None
                  else torch.zeros(C, D, device=self.device),

            # 环境中心 EMA 追踪
            'mu_env': torch.zeros(D, device=self.device),

            # 🟢 【核心修正】：协方差矩阵直接定义为全局共享形状 (B, B)，彻底剥离无用的 C 类别维度
            'S': [sigma * torch.eye(B, device=self.device) for _ in range(G)],

            # 精度矩阵
            'L': [torch.inverse(sigma * torch.eye(B, device=self.device) + eps * torch.eye(B, device=self.device))
                  for _ in range(G)],

            'rng': [(i * B, (i + 1) * B) for i in range(G)],
        }
        return branch

    def _fit_branch(self, br, x):
        x = x.float().to(self.device)

        # 一阶环境去噪
        br['mu_env'] = (1.0 - br['alpha']) * br['mu_env'] + br['alpha'] * x.mean(dim=0)
        x_centered = x - br['mu_env'].unsqueeze(0)

        # 🟢 【核心修正】：纯无监督更新共享散射外积，真实记录当前污染域的几何扭曲
        N = br['N_eff']
        for g, (l, r) in enumerate(br['rng']):
            xg = x_centered[0, l:r]
            delta = torch.outer(xg, xg)
            br['S'][g] = (N * br['S'][g] + delta) / (N + 1.0)

    def _update_branch(self, br):
        """重新计算当前精度矩阵（矩阵逆映射）"""
        for g in range(len(br['rng'])):
            Sg = br['S'][g]  # 🟢 【核心修正】：直接使用无监督捕获的 Sg
            reg = (1.0 - br['eps']) * Sg + br['eps'] * torch.eye(br['B'], device=self.device)
            br['L'][g] = torch.inverse(reg)

    def _predict_branch(self, br, x):
        xd = x.float().to(self.device) - br['mu_env'].unsqueeze(0)
        xd = x.float().to(self.device) - br['mu_env'].unsqueeze(0)
        scores = torch.zeros(1, br['C'], device=self.device)

        for g, (l, r) in enumerate(br['rng']):
            M = br['mu'][:, l:r].T          # (B, C)
            W = br['L'][g] @ M              # (B, C) 通过精度矩阵动态拉伸超平面
            bias = 0.5 * (M * W).sum(dim=0) # (C,)
            scores += xd[:, l:r] @ W - bias

        return scores / br['G']

    def step(self, x, model_logits):
        with torch.no_grad():
            # 双路径前向几何度量
            ss = self._predict_branch(self.slow, x)
            sf = self._predict_branch(self.fast, x)

            ls = model_logits.cpu() + self.omega * ss.cpu()
            lf = model_logits.cpu() + self.omega * sf.cpu()

            max_s = ls.softmax(dim=1).max(dim=1).values.item()
            max_f = lf.softmax(dim=1).max(dim=1).values.item()

            # 状态演化反哺机制实时触发
            self._fit_branch(self.slow, x)
            self._fit_branch(self.fast, x)
            self._update_branch(self.slow)
            self._update_branch(self.fast)

            # 🟢 【修正】：动态omega范围扩展，赋予自适应得分合理的博弈能力
            self.omega = min(self.omega + 0.001, 0.4)

            return ls if max_s > max_f else lf

    def reset(self):
        """跨越噪声边界时清除过往环境记忆，保持 FC 强先验"""
        for br in [self.slow, self.fast]:
            br['mu_env'].zero_()
            B = br['B']
            for g in range(len(br['rng'])):
                br['S'][g] = 0.1 * torch.eye(B, device=self.device)
                br['L'][g] = torch.inverse(0.1 * torch.eye(B, device=self.device) + br['eps'] * torch.eye(B, device=self.device))
        self.omega = 0.2