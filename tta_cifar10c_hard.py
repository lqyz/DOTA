"""Quick TTA on hardest 2 corruptions: baseline vs BN vs BN+blockdiag."""
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

    def fit(self, x, y):
        x, y = x.float().to(self.device), y.float().to(self.device)
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


device = 'cuda'
tf = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
path = '/root/data/picture/CIFAR-10-C'
labels = np.load(os.path.join(path, 'labels.npy'))

def load_m(): return torch.hub.load("chenyaofo/pytorch-cifar-models","cifar10_resnet20",pretrained=True,force_reload=False).to(device)

for cname in ['gaussian_noise.npy', 'glass_blur.npy']:
    data = np.load(os.path.join(path, cname))
    
    # 1. Baseline (frozen eval)
    m = load_m().eval()
    c_b = 0
    with torch.no_grad():
        for i in tqdm(range(5000), desc=f'{cname[:14]}|base 5k', leave=False):
            c_b += int(m(tf(data[i]).unsqueeze(0).to(device)).argmax(1).item() == labels[i])
    
    # 2. BN adaptation (train mode, BN stats update from test data)
    m2 = load_m().train()
    W2, b2 = m2.fc.weight.data, m2.fc.bias.data
    c_bn = 0
    for i in tqdm(range(5000), desc=f'{cname[:14]}|bn   5k', leave=False):
        img = tf(data[i]).unsqueeze(0).to(device)
        logits = m2(img)
        c_bn += int(logits.argmax(1).item() == labels[i])

    # 3. BN + block-diag
    m3 = load_m().train()
    W3, b3 = m3.fc.weight.data, m3.fc.bias.data
    feats3 = {}
    m3.avgpool.register_forward_hook(lambda m,i,o: feats3.__setitem__('x', o.flatten(1)))
    cm = BlockDiagTTA(D=64, C=10, G=4, sigma=0.1, epsilon=0.0001, device=device)
    c_tot, omega = 0, 0.01
    for i in tqdm(range(5000), desc=f'{cname[:14]}|bn+bd5k', leave=False):
        img = tf(data[i]).unsqueeze(0).to(device)
        logits = m3(img); x = feats3['x']; y = logits.softmax(1).detach()
        ds = cm.predict(x)
        final = logits.cpu() + omega * ds.cpu()
        c_tot += int(final.argmax(1).item() == labels[i])
        cm.fit(x, y); cm.update()
        omega = min(0.01 * cm.count.mean().item() / 10, 0.5)

    print(f'{cname[:14]}: base={100*c_b/5000:.2f}%  bn={100*c_bn/5000:.2f}%  bn+block={100*c_tot/5000:.2f}%')
