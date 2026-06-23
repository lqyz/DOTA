"""v6 block-diagonal + BN TTA on CIFAR-10-C."""
import torch, torch.nn as nn, numpy as np, torchvision.transforms as T
from tqdm import tqdm
import os

class BlockDiagTTA:
    def __init__(self, D, C, G, sigma, epsilon, alpha, device):
        self.D, self.C, self.G, self.device = D, C, G, device
        B = D // G
        self.B, self.eps, self.alpha = B, epsilon, alpha
        self.mu = torch.zeros(C, D, device=device)
        self.mu_env = torch.zeros(D, device=device) if alpha else None
        self.count = torch.ones(C, device=device)
        self.Sigma = [sigma * torch.eye(B, device=device).repeat(C, 1, 1) for _ in range(G)]
        self.Lambda = [torch.inverse(sigma * torch.eye(B, device=device) + epsilon * torch.eye(B, device=device)) for _ in range(G)]
        self.block_ranges = [(i * B, (i + 1) * B) for i in range(G)]

    def fit(self, x, y):
        x, y = x.float().to(self.device), y.float().to(self.device)
        if self.alpha:
            self.mu_env = (1 - self.alpha) * self.mu_env + self.alpha * x.mean(0)
            x = x - self.mu_env.unsqueeze(0)
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
            reg = (1 - self.eps) * overall + self.eps * torch.eye(self.B, device=self.device)
            self.Lambda[g] = torch.inverse(reg)

    def predict(self, x):
        x = x.float().to(self.device)
        s = torch.zeros(1, self.C, device=self.device)
        for g, (l, r) in enumerate(self.block_ranges):
            M_g = self.mu[:, l:r].T
            W = self.Lambda[g] @ M_g
            c = 0.5 * (M_g * W).sum(0)
            s = s + (x[:, l:r] @ W - c)
        return s


class FullMatrixTTA:
    def __init__(self, D, C, sigma, epsilon, device):
        self.C, self.D, self.device = C, D, device
        self.eps = epsilon
        self.mu = torch.zeros(C, D, device=device)
        self.count = torch.ones(C, device=device)
        self.Sigma = sigma * torch.eye(D, device=device).repeat(C, 1, 1)
        self.Lambda = torch.inverse(sigma * torch.eye(D, device=device) + epsilon * torch.eye(D, device=device))

    def fit(self, x, y):
        x, y = x.float().to(self.device), y.float().to(self.device)
        w = y.sum(0)
        weighted_x = y.T @ x
        self.mu = (weighted_x + self.count.unsqueeze(1) * self.mu) / (w.unsqueeze(1) + self.count.unsqueeze(1))
        self.count = self.count + w
        x_m_mu = x.unsqueeze(1) - self.mu.unsqueeze(0)
        w_mm = y.unsqueeze(2) * x_m_mu
        delta = torch.einsum('bci,bcj->cij', w_mm, x_m_mu)
        self.Sigma = (self.count[:, None, None] * self.Sigma + delta) / (self.count[:, None, None] + w[:, None, None].clamp(1e-8))

    def update(self):
        overall = self.Sigma.mean(0)
        reg = (1 - self.eps) * overall + self.eps * torch.eye(self.D, device=self.device)
        self.Lambda = torch.inverse(reg)

    def predict(self, x):
        x = x.float().to(self.device)
        M = self.mu.T
        W = self.Lambda @ M
        c = 0.5 * (M * W).sum(0)
        return x @ W - c


def run_tta_full(model, device, tf, path, labels, sigma=0.1):
    accs = {}
    W = model.fc.weight.data; b = model.fc.bias.data
    feats = {}
    model.avgpool.register_forward_hook(lambda m,i,o: feats.__setitem__('x', o.flatten(1)))
    corrs = sorted([f for f in os.listdir(path) if f.endswith('.npy') and f != 'labels.npy'])
    for cname in corrs:
        data = np.load(os.path.join(path, cname))
        cm = FullMatrixTTA(D=64, C=10, sigma=sigma, epsilon=0.0001, device=device)
        correct, omega = 0, 0.01
        for i in tqdm(range(len(data)), desc=f'Full    |{cname[:12]}', leave=False):
            img = tf(data[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(img); x = feats['x']
                logits = x @ W.T + b; y = logits.softmax(1)
            ds = cm.predict(x)
            correct += int((logits.cpu() + omega * ds.cpu()).argmax(1).item() == labels[i])
            cm.fit(x, y); cm.update()
            omega = min(0.01 * cm.count.mean().item() / 10, 0.5)
        accs[cname.replace('.npy','')] = 100 * correct / len(data)
    return accs
    accs = {}
    corrs = sorted([f for f in os.listdir(path) if f.endswith('.npy') and f != 'labels.npy'])
    for cname in corrs:
        data = np.load(os.path.join(path, cname))
        correct = 0
        with torch.no_grad():
            for i in tqdm(range(len(data)), desc=f'Base    |{cname[:12]}', leave=False):
                img = tf(data[i]).unsqueeze(0).to(device)
                correct += int(model(img).argmax(1).item() == labels[i])
        accs[cname.replace('.npy', '')] = 100 * correct / len(data)
    return accs


def run_tta(model, device, tf, path, labels, G=4, adapt_bn=False):
    if adapt_bn:
        model.train()
    W = model.fc.weight.data
    b = model.fc.bias.data
    feats = {}
    model.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__('x', o.flatten(1)))

    accs = {}
    corrs = sorted([f for f in os.listdir(path) if f.endswith('.npy') and f != 'labels.npy'])
    for cname in corrs:
        data = np.load(os.path.join(path, cname))
        cm = BlockDiagTTA(D=64, C=10, G=G, sigma=0.1, epsilon=0.0001, device=device)
        correct, omega = 0, 0.01
        for i in tqdm(range(len(data)), desc=f'TTA     |{cname[:12]}', leave=False):
            img = tf(data[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(img)
                x = feats['x']
                logits = x @ W.T + b
                y = logits.softmax(1)
            ds = cm.predict(x)
            final = logits.cpu() + omega * ds.cpu()
            correct += int(final.argmax(1).item() == labels[i])
            cm.fit(x, y)
            cm.update()
            omega = min(0.01 * cm.count.mean().item() / 10, 0.5)
        accs[cname.replace('.npy', '')] = 100 * correct / len(data)
    return accs


if __name__ == '__main__':
    device = 'cuda'
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    path = '/root/data/picture/CIFAR-10-C'
    labels = np.load(os.path.join(path, 'labels.npy'))

    results = {}

    print("=== BASELINE (no adaptation) ===")
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device).eval()
    results['Base'] = run_baseline(model, device, tf, path, labels)

    print("\n=== v6 BLOCK-DIAG (feature only) ===")
    model2 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device).eval()
    results['BlockDiag'] = run_tta(model2, device, tf, path, labels, G=4)

    print("\n=== v6 + BN Adaptation ===")
    import copy
    model3 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device)
    results['BlockDiag+BN'] = run_tta(model3, device, tf, path, labels, G=4, adapt_bn=True)

    names = ['Base', 'BlockDiag', 'BlockDiag+BN']
    base_k = sorted(results['Base'].keys())
    print(f"\n{'Corruption':25s} {'Baseline':>8s} {'BlockDiag':>9s} {'+BN':>9s}")
    print("-" * 55)
    for k in base_k:
        v = [results[n].get(k, 0) for n in names]
        print(f"{k:25s} {v[0]:7.2f}% {v[1]:8.2f}% {v[2]:8.2f}%")
    print("-" * 55)
    avgs = [sum(results[n].values()) / len(results[n]) for n in names]
    print(f"{'AVERAGE':25s} {avgs[0]:7.2f}% {avgs[1]:8.2f}% {avgs[2]:8.2f}%")

    with open('tta_cifar10c_full.txt', 'w') as f:
        f.write(f"{'Corruption':25s} {'Baseline':>8s} {'BlockDiag':>9s} {'+BN':>9s}\n")
        f.write("-" * 55 + "\n")
        for k in base_k:
            v = [results[n].get(k, 0) for n in names]
            f.write(f"{k:25s} {v[0]:7.2f}% {v[1]:8.2f}% {v[2]:8.2f}%\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'AVERAGE':25s} {avgs[0]:7.2f}% {avgs[1]:8.2f}% {avgs[2]:8.2f}%\n")
    print("Results saved to tta_cifar10c_full.txt")
