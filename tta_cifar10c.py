"""v6 block-diagonal TTA on CIFAR-10-C."""
import torch, torch.nn as nn, numpy as np, torchvision.transforms as T
from tqdm import tqdm
import sys

class BlockDiagTTA(nn.Module):
    def __init__(self, D, C, G=4, sigma=0.1, epsilon=0.0001):
        super().__init__()
        self.D, self.C, self.G = D, C, G
        B = D // G
        self.B, self.eps = B, epsilon
        self.mu = torch.zeros(C, D)
        self.count = torch.ones(C)
        self.Sigma = [sigma * torch.eye(B).repeat(C, 1, 1) for _ in range(G)]
        self.Lambda = [torch.inverse(sigma * torch.eye(B) + epsilon * torch.eye(B)) for _ in range(G)]
        self.block_ranges = [(i * B, (i + 1) * B) for i in range(G)]

    def fit(self, x, y):
        x, y = x.float(), y.float()
        w = y.sum(0)
        weighted_x = y.T @ x
        self.mu = (weighted_x + self.count.unsqueeze(1) * self.mu) / (w.unsqueeze(1) + self.count.unsqueeze(1))
        self.count = self.count + w

        x_m_mu = x.unsqueeze(1) - self.mu.unsqueeze(0)
        for g, (l, r) in enumerate(self.block_ranges):
            x_mm = x_m_mu[..., l:r]
            w_mm = y.unsqueeze(2) * x_mm
            delta = torch.einsum('bci,bcj->cij', w_mm, x_mm)
            self.Sigma[g] = (self.count[:, None, None] * self.Sigma[g] + delta) / (self.count[:, None, None] + w[:, None, None].clamp(1e-8))

    def update(self):
        for g in range(self.G):
            overall = self.Sigma[g].mean(0)
            reg = (1 - self.eps) * overall + self.eps * torch.eye(self.B)
            self.Lambda[g] = torch.inverse(reg)

    def predict(self, x):
        x = x.float()
        scores = torch.zeros(1, self.C, device=x.device)
        for g, (l, r) in enumerate(self.block_ranges):
            M_g = self.mu[:, l:r].T
            W = self.Lambda[g] @ M_g
            c = 0.5 * (M_g * W).sum(0)
            scores = scores + (x[:, l:r] @ W - c)
        return scores


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load model
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    model.to(device).eval()
    W = model.fc.weight.data  # (10, 64)
    b = model.fc.bias.data    # (10,)

    # Hook
    feats = {}
    def hk(m, i, o): feats['x'] = o.flatten(1)
    model.avgpool.register_forward_hook(hk)

    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])

    import os, numpy as np
    path = '/root/data/picture/CIFAR-10-C'
    labels_all = np.load(os.path.join(path, 'labels.npy'))
    corruptions = sorted([f for f in os.listdir(path) if f.endswith('.npy') and f != 'labels.npy'])

    for corr_name in corruptions:
        data = np.load(os.path.join(path, corr_name))
        cmodel = BlockDiagTTA(D=64, C=10, G=4, sigma=0.1)
        cmodel.mu = cmodel.mu.to(device)
        cmodel.count = cmodel.count.to(device)
        cmodel.Sigma = [s.to(device) for s in cmodel.Sigma]
        cmodel.Lambda = [l.to(device) for l in cmodel.Lambda]

        correct, total = 0, 0
        omega = 0.01
        for i in tqdm(range(len(data)), desc=corr_name.replace('.npy',''), leave=False):
            img = tf(data[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(img)
                x = feats['x']  # (1, 64)

            # Model prediction (soft label)
            with torch.no_grad():
                logits = x @ W.T + b  # (1, 10)
                y = logits.softmax(1)

            # DOTA prediction
            dota_s = cmodel.predict(x.to(device))

            # Fuse
            final = logits.cpu() + omega * dota_s.cpu()
            pred = final.argmax(1).item()
            correct += int(pred == labels_all[i])

            # Update
            cmodel.fit(x.to(device), y.to(device))
            cmodel.update()
            # Adaptive omega
            omega = min(0.01 * cmodel.count.mean().item() / 10, 0.5)

            total += 1

        acc = 100 * correct / total
        print(f'{corr_name.replace(".npy",""):20s}: {acc:.2f}%')

if __name__ == '__main__':
    main()
