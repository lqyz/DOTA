"""Quick test: count clamp + confidence gating on 2 corruptions."""
import torch, numpy as np, torchvision.transforms as T, os
from tqdm import tqdm

class BlockDiagTTA:
    def __init__(self, D, C, G, sigma, epsilon, device):
        self.D, self.C, self.G, self.device = D, C, G, device
        B = D // G
        self.B, self.eps = B, epsilon
        self.mu = torch.zeros(C, D, device=device)
        self.count = torch.ones(C, device=device)
        self.Sigma = [sigma * torch.eye(B, device=device).repeat(C, 1, 1) for _ in range(G)]
        self.Lambda = [torch.inverse(sigma * torch.eye(B, device=device) + epsilon * torch.eye(B, device=device)) for _ in range(G)]
        self.block_ranges = [(i * B, (i + 1) * B) for i in range(G)]

    def fit(self, x, y, use_clamp=True):
        x, y = x.float().to(self.device), y.float().to(self.device)
        w = y.sum(0)
        weighted_x = y.T @ x
        self.mu = (weighted_x + self.count.unsqueeze(1) * self.mu) / (w.unsqueeze(1) + self.count.unsqueeze(1))
        self.count = torch.clamp(self.count + w, max=100) if use_clamp else self.count + w
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


device = 'cuda'
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to(device).eval()
W = model.fc.weight.data; b = model.fc.bias.data
feats = {}
model.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__('x', o.flatten(1)))
tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
path = '/root/data/picture/CIFAR-10-C'
labels = np.load(os.path.join(path, 'labels.npy'))

for cname in ['gaussian_noise.npy', 'glass_blur.npy']:
    data = np.load(os.path.join(path, cname))
    for use_clamp, use_gate, label in [(False, False, 'original'), (True, False, 'clamp100'), (True, True, 'clamp+gate')]:
        cm = BlockDiagTTA(D=64, C=10, G=4, sigma=0.1, epsilon=0.0001, device=device)
        correct, omega, skipped = 0, 0.01, 0
        for i in range(len(data)):
            img = tf(data[i]).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model(img); x = feats['x']
                logits = x @ W.T + b; y = logits.softmax(1)
            ds = cm.predict(x)
            correct += int((logits.cpu() + omega * ds.cpu()).argmax(1).item() == labels[i])
            if (not use_gate) or y.max().item() >= 0.5:
                cm.fit(x, y, use_clamp=use_clamp); cm.update()
            omega = min(0.01 * cm.count.mean().item() / 10, 0.5)
        print(f'{cname[:14]:15s} {label:12s}: {100*correct/len(data):.2f}%')
    print()
