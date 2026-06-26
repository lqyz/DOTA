"""Continual Domain Adaptation: 5 hardest corruptions, 5000 samples each, sequential."""
import torch, numpy as np, torchvision.transforms as T, os, sys
from tqdm import tqdm
sys.path.insert(0,'/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
from pytorch_cifar_models.resnet import cifar10_resnet20

device='cuda'
tf=T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
path='/root/data/picture/CIFAR-10-C';labels=np.load(os.path.join(path,'labels.npy'))
sd=torch.load('/root/.cache/torch/hub/checkpoints/cifar10_resnet20-4118986f.pt',map_location='cpu')
sd={k.replace('module.',''):v for k,v in sd.items()}

class DB:
    def __init__(s,D,C,G,sigma,eps,alpha,device,max_count=100.0):
        s.D,s.C,s.G,s.device=D,C,G,device;B=D//G;s.B,s.eps,s.alpha=B,eps,alpha
        s.mu=torch.zeros(C,D,device=device);s.mu_env=torch.zeros(D,device=device)
        s.count=torch.ones(C,device=device);s.max_count=max_count
        s.S=[sigma*torch.eye(B,device=device).repeat(C,1,1) for _ in range(G)]
        s.L=[torch.inverse(sigma*torch.eye(B,device=device)+eps*torch.eye(B,device=device)) for _ in range(G)]
        s.rng=[(i*B,(i+1)*B) for i in range(G)]
    def fit(s,x,y):
        x,y=x.float().to(s.device),y.float().to(s.device)
        s.mu_env=(1-s.alpha)*s.mu_env+s.alpha*x.mean(0);x=x-s.mu_env.unsqueeze(0)
        w=y.sum(0);wx=y.T@x;s.mu=(wx+s.count.unsqueeze(1)*s.mu)/(w.unsqueeze(1)+s.count.unsqueeze(1))
        s.count=(s.count+w).clamp(max=s.max_count)
        xmm=x.unsqueeze(1)-s.mu.unsqueeze(0)
        for g,(l,r) in enumerate(s.rng):
            wm=y.unsqueeze(2)*xmm[...,l:r];d=torch.einsum('bci,bcj->cij',wm,xmm[...,l:r])
            s.S[g]=(s.count[:,None,None]*s.S[g]+d)/(s.count[:,None,None]+w[:,None,None].clamp(1e-8))
    def update(s):
        for g in range(s.G):
            ov=s.S[g].mean(0);rg=(1-s.eps)*ov+s.eps*torch.eye(s.B,device=s.device);s.L[g]=torch.inverse(rg)
    def predict(s,x):
        xd=x.float().to(s.device)-s.mu_env.unsqueeze(0);sc=torch.zeros(1,s.C,device=s.device)
        for g,(l,r) in enumerate(s.rng):
            M=s.mu[:,l:r].T;W=s.L[g]@M;sc+=xd[:,l:r]@W-0.5*(M*W).sum(0)
        return sc

# All 19 corruptions in order
seq=[(f.replace('.npy',''),f) for f in sorted([f for f in os.listdir(path) if f.endswith('.npy') and f!='labels.npy'])]

cm_s=DB(64,10,4,0.1,0.0001,0.002,device,max_count=100)
cm_f=DB(64,10,4,0.1,0.0001,0.02,device,max_count=20)
w=0.01;N=5000

print(f"{'Phase':>5s} {'Corruption':15s} {'Acc':>7s} {'Slow%':>7s}")
print('-'*40)
for phase,(name,fn) in enumerate(seq):
    data=np.load(os.path.join(path,fn))[:N];m=cifar10_resnet20();m.load_state_dict(sd);m.to(device).eval()
    f={};m.avgpool.register_forward_hook(lambda m,i,o:f.__setitem__('x',o.flatten(1)))
    c,sw,fw=0,0,0
    with torch.no_grad():
        for i in range(N):
            img=tf(data[i]).unsqueeze(0).to(device);_=m(img);x=f['x'];lgs=x@m.fc.weight.data.T+m.fc.bias.data
            sc_s=cm_s.predict(x);sc_f=cm_f.predict(x)
            ls=lgs.cpu()+w*sc_s.cpu();lf=lgs.cpu()+w*sc_f.cpu()
            ms=ls.softmax(1).max(1)[0].item();mf=lf.softmax(1).max(1)[0].item()
            if ms>mf:sw+=1;wl=ls;wp=ls.softmax(1)
            else:fw+=1;wl=lf;wp=lf.softmax(1)
            c+=int(wl.argmax(1).item()==labels[i])
            cm_s.fit(x,wp.to(device));cm_f.fit(x,wp.to(device));cm_s.update();cm_f.update()
            w=min(0.01*cm_s.count.mean().item()/10,0.5)
    acc=100*c/N;sr=100*sw/(sw+fw)if sw+fw>0 else 50
    print(f'{phase:5d} {name:15s} {acc:6.2f}% {sr:6.0f}%')
