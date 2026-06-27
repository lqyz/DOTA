"""ImageNet-C: 3 hardest corruptions, Severity 5, 200 classes, dual-path."""
import torch, torchvision, numpy as np, os, random
from tqdm import tqdm
from PIL import Image

device='cuda'
model=torchvision.models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
feats={};model.avgpool.register_forward_hook(lambda m,i,o:feats.__setitem__('x',o.flatten(1)))
tf=torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])
W,b=model.fc.weight.data,model.fc.bias.data
C,D,G=200,2048,32

class ImgNetTTA:
    def __init__(s,C,D,G,sigma,eps,alpha,device,N_eff=100.0,W_init=None):
        s.C,s.D,s.G,s.device=C,D,G,device;s.B=D//G;s.eps,s.alpha=eps,alpha;s.N=N_eff
        if W_init is not None:s.mu=W_init.clone().to(device);s.mu=s.mu/s.mu.norm(dim=-1,keepdim=True)
        else:s.mu=torch.zeros(C,D,device=device)
        s.mu_env=torch.zeros(D,device=device)
        s.S_global=sigma*torch.eye(s.B,device=device).repeat(G,1,1)
        s.L=[torch.inverse(sigma*torch.eye(s.B,device=device)+eps*torch.eye(s.B,device=device)) for _ in range(G)]
        s.rng=[(i*s.B,(i+1)*s.B) for i in range(G)]
    def fit(s,x,y):
        x,y=x.float().to(s.device),y.float().to(s.device)
        s.mu_env=(1-s.alpha)*s.mu_env+s.alpha*x.mean(0);x=x-s.mu_env.unsqueeze(0)
        w=y.sum(0);wx=y.T@x;n=s.N;s.mu=(wx+n*s.mu)/(w.unsqueeze(1)+n)
        xmm=x.unsqueeze(1)-s.mu.unsqueeze(0)
        for g,(l,r) in enumerate(s.rng):
            xg=xmm[0,:,l:r];d=(xg.T@xg)/xg.shape[0]
            s.S_global[g]=(n*s.S_global[g]+d)/(n+1.0)
    def update(s):
        for g in range(s.G):
            rg=(1-s.eps)*s.S_global[g]+s.eps*torch.eye(s.B,device=s.device);s.L[g]=torch.inverse(rg)
    def predict(s,x):
        xd=x.float().to(s.device)-s.mu_env.unsqueeze(0);sc=torch.zeros(1,s.C,device=s.device)
        for g,(l,r) in enumerate(s.rng):
            M=s.mu[:,l:r].T;Wg=s.L[g]@M;sc+=xd[:,l:r]@Wg-0.5*(M*Wg).sum(0)
        return sc/s.G

W_sub=W[list(range(C))]
base_root='/root/data/picture/ImageNet-C'
corrs=['gaussian_noise','shot_noise','glass_blur']
sev='5'

for corr in corrs:
    root=f'{base_root}/{corr}/{sev}'
    folders=sorted(os.listdir(root))[:C]
    imgs,labels=[],[]
    for lbl,folder in enumerate(folders):
        for fname in os.listdir(os.path.join(root,folder))[:1]:
            imgs.append(os.path.join(root,folder,fname));labels.append(lbl)
    
    # Baseline on ALL severities (same data as TTA)
    c0=0
    with torch.no_grad():
        for i in range(len(imgs2)):
            img=tf(Image.open(imgs2[i]).convert('RGB')).unsqueeze(0).to(device)
            c0+=int(model(img)[0,list(range(C))].argmax(0).item()==labels2[i])
    
    cm_s=ImgNetTTA(C,D,G,0.1,0.0001,0.002,device,N_eff=100,W_init=W_sub)
    cm_f=ImgNetTTA(C,D,G,0.1,0.0001,0.02,device,N_eff=20,W_init=W_sub)
    c,sw,fw=0,0,0;wv=0.002
    for i in tqdm(range(len(imgs2)),desc=f'{corr}',leave=False):
        img=tf(Image.open(imgs2[i]).convert('RGB')).unsqueeze(0).to(device)
        with torch.no_grad():_=model(img);x=feats['x'];lgs=x@W.T+b;lgs=lgs[0,list(range(C))]
        sc_s=cm_s.predict(x);sc_f=cm_f.predict(x)
        ls=lgs.cpu()+wv*sc_s.cpu().squeeze(0);lf=lgs.cpu()+wv*sc_f.cpu().squeeze(0)
        ms=ls.softmax(0).max(0).values.item();mf=lf.softmax(0).max(0).values.item()
        if ms>mf:sw+=1;wl=ls;wp=ls.softmax(0)
        else:fw+=1;wl=lf;wp=lf.softmax(0)
        c+=int(wl.argmax(0).item()==labels2[i])
        cm_s.fit(x,wp.unsqueeze(0).to(device));cm_f.fit(x,wp.unsqueeze(0).to(device));cm_s.update();cm_f.update()
        wv=min(0.002*(i+1)/100,0.02)
    print(f'{corr}: base={100*c0/len(imgs):.2f}%  TTA={100*c/len(imgs2):.2f}%  slow={100*sw/(sw+fw):.0f}%')
