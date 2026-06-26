"""ImageNet-C Severity 5: global shared block-diag + FC-init mu."""
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

W=model.fc.weight.data
C=200;D=2048;G=32;B=D//G

class ImgNetTTA:
    def __init__(self,C,D,G,sigma,eps,alpha,device,N_eff=100.0,W_init=None):
        self.C,self.D,self.G,self.device=C,D,G,device
        self.B=D//G;self.eps,self.alpha=eps,alpha;self.N=N_eff
        if W_init is not None:
            self.mu=W_init.clone().to(device)
            self.mu=self.mu/self.mu.norm(dim=-1,keepdim=True)
        else:
            self.mu=torch.zeros(C,D,device=device)
        self.mu_env=torch.zeros(D,device=device)
        self.S_global=sigma*torch.eye(self.B,device=device).repeat(G,1,1)
        self.L=[torch.inverse(sigma*torch.eye(self.B,device=device)+eps*torch.eye(self.B,device=device)) for _ in range(G)]
        self.rng=[(i*self.B,(i+1)*self.B) for i in range(G)]

    def fit(self,x,y):
        x,y=x.float().to(self.device),y.float().to(self.device)
        self.mu_env=(1-self.alpha)*self.mu_env+self.alpha*x.mean(0)
        x=x-self.mu_env.unsqueeze(0)
        w=y.sum(0);wx=y.T@x
        n=self.N
        self.mu=(wx+n*self.mu)/(w.unsqueeze(1)+n)
        xmm=x.unsqueeze(1)-self.mu.unsqueeze(0)
        for g,(l,r) in enumerate(self.rng):
            xg=xmm[0,:,l:r]
            d=(xg.T@xg)/xg.shape[0]
            self.S_global[g]=(n*self.S_global[g]+d)/(n+1.0)

    def update(self):
        for g in range(self.G):
            rg=(1-self.eps)*self.S_global[g]+self.eps*torch.eye(self.B,device=self.device)
            self.L[g]=torch.inverse(rg)

    def predict(self,x):
        xd=x.float().to(self.device)-self.mu_env.unsqueeze(0)
        sc=torch.zeros(1,self.C,device=self.device)
        for g,(l,r) in enumerate(self.rng):
            M=self.mu[:,l:r].T;Wg=self.L[g]@M
            sc+=xd[:,l:r]@Wg-0.5*(M*Wg).sum(0)
        return sc/self.G


root='/root/data/picture/ImageNet-C/brightness/5'
folders=sorted(os.listdir(root))[:C]
imgs,labels=[],[]
for lbl,folder in enumerate(folders):
    for fname in os.listdir(os.path.join(root,folder))[:1]:
        imgs.append(os.path.join(root,folder,fname));labels.append(lbl)

W_sub=W[list(range(C))]
cm_s=ImgNetTTA(C,D,G,0.1,0.0001,0.002,device,N_eff=100,W_init=W_sub)
cm_f=ImgNetTTA(C,D,G,0.1,0.0001,0.02,device,N_eff=20,W_init=W_sub)

# Baseline
c0=0
with torch.no_grad():
    for i in range(len(imgs)):
        img=tf(Image.open(imgs[i]).convert('RGB')).unsqueeze(0).to(device)
        logits=model(img)
        c0+=int(logits[0,list(range(C))].argmax(0).item()==labels[i])
print(f'Baseline: {100*c0/len(imgs):.2f}%')

# Shuffle
random.seed(42);combined=list(zip(imgs,labels));random.shuffle(combined)
imgs,labels=zip(*combined)

# Dual-path
c,sw,fw=0,0,0;w_val=0.002
for i in tqdm(range(len(imgs)),desc='ImgNet-C'):
    img=tf(Image.open(imgs[i]).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        _=model(img);x=feats['x'];lgs_full=x@W.T+model.fc.bias
        lgs=lgs_full[0,list(range(C))]
    sc_s=cm_s.predict(x);sc_f=cm_f.predict(x)
    ls=lgs.cpu()+w_val*sc_s.cpu().squeeze(0)
    lf=lgs.cpu()+w_val*sc_f.cpu().squeeze(0)
    ms=ls.softmax(0).max(0).values.item()
    mf=lf.softmax(0).max(0).values.item()
    if ms>mf:sw+=1;wl=ls;wp=ls.softmax(0)
    else:fw+=1;wl=lf;wp=lf.softmax(0)
    c+=int(wl.argmax(0).item()==labels[i])
    cm_s.fit(x,wp.unsqueeze(0).to(device))
    cm_f.fit(x,wp.unsqueeze(0).to(device))
    cm_s.update();cm_f.update()
    w_val=min(0.002*(i+1)/100,0.02)
print(f'Dual-path: {100*c/len(imgs):.2f}%  slow={100*sw/(sw+fw):.0f}%')
