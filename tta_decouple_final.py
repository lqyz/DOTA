"""Decoupled TTA: environment subtraction + block-diagonal on CIFAR-10-C.
Reference: DOTA-blockdiag v6 with environment decoupling (alpha=0.5).
"""
import torch, numpy as np, torchvision.transforms as T, os, sys
from tqdm import tqdm

class DecoupleBlockTTA:
    """Block-diagonal TTA with online environment estimation and decoupling.
    
    Instead of updating per-class statistics directly on raw features x,
    we track a global environment mean mu_env via EMA and subtract it:
        x_dec = x - mu_env
    This separates class-specific content from global corruption shift.
    
    Args:
        D: feature dimension (64 for ResNet-20)
        C: number of classes (10 for CIFAR-10)
        G: number of diagonal blocks (4)
        sigma: initial variance
        epsilon: precision regularization
        alpha: EMA rate for environment tracking (0.5 optimal)
        device: torch device
    """
    def __init__(self, D, C, G, sigma, epsilon, alpha, device):
        self.D, self.C, self.G, self.device = D, C, G, device
        B = D // G
        self.B, self.eps, self.alpha = B, epsilon, alpha
        self.mu = torch.zeros(C, D, device=device)
        self.mu_env = torch.zeros(D, device=device)
        self.count = torch.ones(C, device=device)
        self.Sigma = [sigma * torch.eye(B, device=device).repeat(C, 1, 1) for _ in range(G)]
        self.Lambda = [torch.inverse(sigma * torch.eye(B, device=device) + epsilon * torch.eye(B, device=device)) for _ in range(G)]
        self.block_ranges = [(i * B, (i + 1) * B) for i in range(G)]

    def fit(self, x, y):
        x, y = x.float().to(self.device), y.float().to(self.device)
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
        x = x.float().to(self.device) - self.mu_env.unsqueeze(0)
        s = torch.zeros(1, self.C, device=self.device)
        for g, (l, r) in enumerate(self.block_ranges):
            M_g = self.mu[:, l:r].T
            W = self.Lambda[g] @ M_g
            c = 0.5 * (M_g * W).sum(0)
            s = s + (x[:, l:r] @ W - c)
        return s


if __name__ == '__main__':
    import torch, numpy as np, torchvision.transforms as T, os, sys
    sys.path.insert(0, '/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
    from pytorch_cifar_models.resnet import cifar10_resnet20

    device = 'cuda'
    tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    path = '/root/data/picture/CIFAR-10-C'
    labels = np.load(os.path.join(path, 'labels.npy'))

    sd = torch.load('/root/.cache/torch/hub/checkpoints/cifar10_resnet20-4118986f.pt', map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    results = {}
    for fn in sorted([f for f in os.listdir(path) if f.endswith('.npy') and f != 'labels.npy']):
        data = np.load(os.path.join(path, fn))
        m = cifar10_resnet20(); m.load_state_dict(sd); m.to(device).eval()
        feats = {}
        m.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__('x', o.flatten(1)))
        cm = DecoupleBlockTTA(D=64, C=10, G=4, sigma=0.1, epsilon=0.0001, alpha=0.5, device=device)
        correct, omega = 0, 0.01
        with torch.no_grad():
            for i in tqdm(range(len(data)), desc=fn[:14], leave=False):
                img = tf(data[i]).unsqueeze(0).to(device)
                _ = m(img); x = feats['x']
                logits = x @ m.fc.weight.data.T + m.fc.bias.data
                y = logits.softmax(1)
                ds = cm.predict(x)
                correct += int((logits.cpu() + omega * ds.cpu()).argmax(1).item() == labels[i])
                cm.fit(x, y); cm.update()
                omega = min(0.01 * cm.count.mean().item() / 10, 0.5)
        results[fn.replace('.npy', '')] = 100 * correct / len(data)
        print(f'{fn[:14]}: {results[fn.replace(".npy","")]:.2f}%')

    avg = sum(results.values()) / len(results)
    print(f'\nAVERAGE: {avg:.2f}%')
