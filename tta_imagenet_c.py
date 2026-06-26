"""ImageNet-C Severity 5: global shared block-diagonal + FC-init mu."""
import torch, torchvision, numpy as np, os
from tqdm import tqdm
from PIL import Image

device='cuda'
model=torchvision.models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
feats={};model.avgpool.register_forward_hook(lambda m,i,o:feats.__setitem__('x',o.flatten(1)))
tf=torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

W=model.fc.weight.data;b=model.fc.bias.data
C=1000;D=2048

class ImgNetTTA:
    def __init__(s,C,D,G,sigma,eps,alpha,device,N_eff=100.0,W_init=None):
        s.C,s.D,s.G,s.device=C,D,G,device;B=D//G;s.B,s.eps,s.alpha=B,eps,alpha;s.N=N_eff
        if W_init is not None:
            s.mu=W_init.clone().to(device)
            s.mu=s.mu/s.mu.norm(dim=-1,keepdim=True)
        else:
            s.mu=torch.zeros(C,D,device=device)
        s.mu_env=torch.zeros(D,device=device)
        s.S_global=sigma*torch.eye(B,device=device).repeat(G,1,1)
        s.L=[torch.inverse(sigma*torch.eye(B,device=device)+eps*torch.eye(B,device=device)) for _ in range(G)]
        s.rng=[(i*B,(i+1)*B) for i in range(G)]
    def fit(s,x,y):
        x,y=x.float().to(s.device),y.float().to(s.device)
        s.mu_env=(1-s.alpha)*s.mu_env+s.alpha*x.mean(0);x=x-s.mu_env.unsqueeze(0)
        w=y.sum(0);wx=y.T@x;n=s.N
        s.mu=(wx+n*s.mu)/(w.unsqueeze(1)+n)
        xmm=x.unsqueeze(1)-s.mu.unsqueeze(0)
        for g,(l,r) in enumerate(s.rng):
            xg=xmm[...,l:r].flatten(0,1);d=torch.outer(xg,xg).mean(0,keepdim=True)
            s.S_global[g]=(n*s.S_global[g]+d[0])/(n+1.0)
    def update(s):
        for g in range(s.G):
            rg=(1-s.eps)*s.S_global[g]+s.eps*torch.eye(s.B,device=s.device);s.L[g]=torch.inverse(rg)
    def predict(s,x):
        xd=x.float().to(s.device)-s.mu_env.unsqueeze(0);sc=torch.zeros(1,s.C,device=s.device)
        for g,(l,r) in enumerate(s.rng):
            M=s.mu[:,l:r].T;Wg=s.L[g]@M;sc+=xd[:,l:r]@Wg-0.5*(M*Wg).sum(0)
        return sc

root='/root/data/picture/ImageNet-C/brightness/5'
folders=sorted(os.listdir(root))[:200]
C=len(folders);print(f'Classes: {C}')

imgs,labels=[],[]
for lbl,folder in enumerate(folders):
    for fname in os.listdir(os.path.join(root,folder))[:1]:
        imgs.append(os.path.join(root,folder,fname));labels.append(lbl)

# Init mu from FC weights of selected classes
W_sub=W[list(range(C))]  # Take first C FC weight vectors

cm_s=ImgNetTTA(C,D,32,0.1,0.0001,0.002,device,N_eff=100,W_init=W_sub)
cm_f=ImgNetTTA(C,D,32,0.1,0.0001,0.02,device,N_eff=20,W_init=W_sub)
w=0.01

# Baseline
c0=0
with torch.no_grad():
    for i in range(len(imgs)):
        img=tf(Image.open(imgs[i]).convert('RGB')).unsqueeze(0).to(device)
        logits=model(img)
        c0+=int(logits[0,list(range(C))].argmax(0).item()==labels[i])
print(f'Baseline: {100*c0/len(imgs):.2f}%')

# Dual-path
c,sw,fw=0,0,0
for i in tqdm(range(len(imgs)),desc='ImgNet-C'):
    img=tf(Image.open(imgs[i]).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        _=model(img);x=feats['x'];lgs_full=x@W.T+b
        lgs=lgs_full[0,list(range(C))];y=lgs.softmax(0)
    sc_s=cm_s.predict(x);sc_f=cm_f.predict(x)
    ls=lgs.cpu()+w*sc_s.cpu();lf=lgs.cpu()+w*sc_f.cpu()
    ms=ls.softmax(0).max(0)[0].item();mf=lf.softmax(0).max(0)[0].item()
    if ms>mf:sw+=1;wl=ls;wp=ls.softmax(0)
    else:fw+=1;wl=lf;wp=lf.softmax(0)
    c+=int(wl.argmax(0).item()==labels[i])
    cm_s.fit(x,wp.to(device));cm_f.fit(x,wp.to(device));cm_s.update();cm_f.update()
    w=min(0.01*100/10,0.2)
print(f'Dual-path: {100*c/len(imgs):.2f}%  slow={100*sw/(sw+fw):.0f}%')
