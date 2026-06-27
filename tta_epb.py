"""EPB Test: Environment Prototype Bank for chaotic streams."""
import torch, numpy as np, torchvision.transforms as T, os, sys, random
sys.path.insert(0,'/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
from pytorch_cifar_models.resnet import cifar10_resnet20
from tqdm import tqdm
device='cuda';tf=T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
path='/root/data/picture/CIFAR-10-C';labels=np.load(os.path.join(path,'labels.npy'))
sd=torch.load('/root/.cache/torch/hub/checkpoints/cifar10_resnet20-4118986f.pt',map_location='cpu')
sd={k.replace('module.',''):v for k,v in sd.items()}

class EPB_TTA:
    def __init__(s,D,C,G,sigma,eps,alpha_s,alpha_f,N_eff_s,N_eff_f,K,device):
        s.D,s.C,s.G,s.device=D,C,G,device;K=min(K,D);B=D//G
        s.B,s.eps=B,eps;s.K=K
        # Environment prototypes (K x D)
        s.env_bank=torch.randn(K,D,device=device)*0.01
        # Per-branch params
        for nm,al,ne in [('s',alpha_s,N_eff_s),('f',alpha_f,N_eff_f)]:
            br={'alpha':al,'N_eff':ne,'mu':torch.zeros(C,D,device=device),
                'Sigma':[sigma*torch.eye(B,device=device).repeat(C,1,1) for _ in range(G)],
                'Lambda':[torch.inverse(sigma*torch.eye(B,device=device)+eps*torch.eye(B,device=device)) for _ in range(G)],
                'rng':[(i*B,(i+1)*B) for i in range(G)]}
            setattr(s,nm,br)

    def _route_env(self,x):
        xm=x.float().to(self.device).mean(0)
        dist=torch.norm(self.env_bank-xm.unsqueeze(0),dim=1)
        k=dist.argmin().item()
        self.env_bank[k]=(1-self.s.alpha)*self.env_bank[k]+self.s.alpha*xm
        return k,self.env_bank[k]

    def _fit_br(self,br,x,y,env):
        xd=x.float().to(self.device)-env.unsqueeze(0)
        w=y.sum(0);wx=y.T@xd;N=br['N_eff']
        br['mu']=(wx+N*br['mu'])/(w.unsqueeze(1)+N)
        xmm=xd.unsqueeze(1)-br['mu'].unsqueeze(0)
        for g,(l,r) in enumerate(br['rng']):
            wm=y.unsqueeze(2)*xmm[...,l:r];d=torch.einsum('bci,bcj->cij',wm,xmm[...,l:r])
            br['Sigma'][g]=(N*br['Sigma'][g]+d)/(N+w[:,None,None].clamp(1e-8))
    def _upd(self,br):
        for g in range(br['G']):ov=br['Sigma'][g].mean(0);rg=(1-s.eps)*ov+s.eps*torch.eye(s.B,device=s.device);br['Lambda'][g]=torch.inverse(rg)
    def _pred(self,br,x,env):
        xd=x.float().to(self.device)-env.unsqueeze(0);sc=torch.zeros(1,s.C,device=s.device)
        for g,(l,r) in enumerate(br['rng']):M=br['mu'][:,l:r].T;W=br['Lambda'][g]@M;sc+=xd[:,l:r]@W-0.5*(M*W).sum(0)
        return sc

    def step(self,x,logits):
        k,env=self._route_env(x)
        with torch.no_grad():
            ss=self._pred(self.s,x,env);sf=self._pred(self.f,x,env)
            ls=logits.cpu()+wv*ss.cpu();lf=logits.cpu()+wv*sf.cpu()
            ms=ls.softmax(1).max(1)[0].item();mf=lf.softmax(1).max(1)[0].item()
            if ms>mf:wl=ls;wp=ls.softmax(1)
            else:wl=lf;wp=lf.softmax(1)
            self._fit_br(self.s,x,wp.to(self.device),env);self._fit_br(self.f,x,wp.to(self.device),env)
            self._upd(self.s);self._upd(self.f)
        return wl

A,B=[],[]
for fn in sorted([f for f in os.listdir(path) if f.endswith('.npy') and f!='labels.npy']):
    d=np.load(os.path.join(path,fn))[:5000]
    for i in range(5000):A.append(d[i]);B.append(labels[i])
idx=list(range(len(A)));random.shuffle(idx);A=[A[i]for i in idx];B=[B[i]for i in idx]
m=cifar10_resnet20();m.load_state_dict(sd);m.to(device).eval()
f={};m.avgpool.register_forward_hook(lambda m,i,o:f.__setitem__('x',o.flatten(1)))

c0=0
with torch.no_grad():
    for i in range(len(A)):c0+=int(m(tf(A[i]).unsqueeze(0).to(device)).argmax(1).item()==B[i])
print(f'Baseline: {100*c0/len(A):.2f}%')

for K in [2,5,10,19]:
    tta=EPB_TTA(64,10,4,0.1,1e-4,0.01,0.05,100,20,K,device)
    c,wv=0,0.01
    with torch.no_grad():
        for i in tqdm(range(len(A)),desc=f'K={K}',leave=False):
            img=tf(A[i]).unsqueeze(0).to(device);_=m(img);x=f['x'];lgs=x@m.fc.weight.data.T+m.fc.bias.data
            wl=tta.step(x,lgs);c+=int(wl.argmax(1).item()==B[i]);wv=min(0.01*100/10,0.5)
    print(f'K={K}: {100*c/len(A):.2f}%')
