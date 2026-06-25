"""Dynamic Domain Shift: sequential corruption flow without model reset."""
import torch, numpy as np, torchvision.transforms as T, os, sys
from tqdm import tqdm
sys.path.insert(0,'/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
from pytorch_cifar_models.resnet import cifar10_resnet20

device='cuda'
tf = T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
path = '/root/data/picture/CIFAR-10-C'
labels = np.load(os.path.join(path,'labels.npy'))
sd = torch.load('/root/.cache/torch/hub/checkpoints/cifar10_resnet20-4118986f.pt',map_location='cpu')
sd = {k.replace('module.',''):v for k,v in sd.items()}

class DB:
    def __init__(s,D,C,G,sigma,eps,alpha,device):
        s.D,s.C,s.G,s.device=D,C,G,device;B=D//G;s.B,s.eps,s.alpha=B,eps,alpha
        s.mu=torch.zeros(C,D,device=device);s.mu_env=torch.zeros(D,device=device)
        s.count=torch.ones(C,device=device)
        s.S=[sigma*torch.eye(B,device=device).repeat(C,1,1) for _ in range(G)]
        s.L=[torch.inverse(sigma*torch.eye(B,device=device)+eps*torch.eye(B,device=device)) for _ in range(G)]
        s.rng=[(i*B,(i+1)*B) for i in range(G)]
    def fit(s,x,y):
        x,y=x.float().to(s.device),y.float().to(s.device)
        s.mu_env=(1-s.alpha)*s.mu_env+s.alpha*x.mean(0);x=x-s.mu_env.unsqueeze(0)
        w=y.sum(0);wx=y.T@x;s.mu=(wx+s.count.unsqueeze(1)*s.mu)/(w.unsqueeze(1)+s.count.unsqueeze(1));s.count+=w
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

# Test sequence: mild -> severe -> mild -> severe -> mild
test_seq = [
    ('brightness',     'brightness.npy'),
    ('gaussian_noise', 'gaussian_noise.npy'),
    ('fog',            'fog.npy'),
    ('glass_blur',     'glass_blur.npy'),
    ('contrast',       'contrast.npy'),
    ('shot_noise',     'shot_noise.npy'),
]

cm_s=DB(64,10,4,0.1,0.0001,0.1,device); cm_f=DB(64,10,4,0.1,0.0001,0.5,device)
w=0.01; log=[]

for phase_name, fn in test_seq:
    data=np.load(os.path.join(path,fn)); m=cifar10_resnet20();m.load_state_dict(sd);m.to(device).eval()
    f={};m.avgpool.register_forward_hook(lambda m,i,o:f.__setitem__('x',o.flatten(1)))
    c_ph,sw,fw=0,0,0
    N=5000
    with torch.no_grad():
        for i in tqdm(range(N),desc=phase_name[:14],leave=False):
            img=tf(data[i]).unsqueeze(0).to(device);_=m(img);x=f['x'];lgs=x@m.fc.weight.data.T+m.fc.bias.data
            sc_s=cm_s.predict(x);sc_f=cm_f.predict(x)
            lgs_s=lgs.cpu()+w*sc_s.cpu();lgs_f=lgs.cpu()+w*sc_f.cpu()
            ms=lgs_s.softmax(1).max(1)[0].item();mf=lgs_f.softmax(1).max(1)[0].item()
            if ms>mf:sw+=1;wl=lgs_s;wp=lgs_s.softmax(1);rt='slow'
            else:fw+=1;wl=lgs_f;wp=lgs_f.softmax(1);rt='fast'
            c_ph+=int(wl.argmax(1).item()==labels[i])
            cm_s.fit(x,wp.to(device));cm_f.fit(x,wp.to(device));cm_s.update();cm_f.update()
            w=min(0.01*cm_s.count.mean().item()/10,0.2)
            log.append({'step':len(log),'phase':phase_name,'route':rt,'margin':ms-mf})
    
    acc=100*c_ph/N;sr=100*sw/(sw+fw)if sw+fw>0 else 50
    print(f'{phase_name:15s}: {acc:.2f}%  slow={sr:.0f}%  fast={100-sr:.0f}% (w={w:.3f})')

torch.save(log,'/root/DOTA/dynamic_shift_log.pt')
print('\nLog saved to dynamic_shift_log.pt')
