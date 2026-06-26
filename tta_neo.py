"""NEO Protocol: 50-shot single-class adaptation, test generalization to unseen classes."""
import torch, numpy as np, torchvision.transforms as T, os, sys
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

# Test 3 hardest corruptions
for cname in ['gaussian_noise.npy','glass_blur.npy','shot_noise.npy']:
    data=np.load(os.path.join(path,cname))
    print(f'\n=== {cname[:14]} ===')
    
    # Baseline: no adaptation at all
    m=cifar10_resnet20();m.load_state_dict(sd);m.to(device).eval()
    base={}
    with torch.no_grad():
        for cl in range(10):
            mask=labels==cl
            c0=0
            for i in np.where(mask)[0][:500]:
                c0+=int(m(tf(data[i]).unsqueeze(0).to(device)).argmax(1).item()==labels[i])
            base[cl]=100*c0/500
    print(f'  Baseline  : {" ".join(f"{base[cl]:5.1f}" for cl in range(10))}')
    
    # NEO protocol: adapt on class 0 (50 shots), test all classes
    for adapt_class in [0,3,7]:  # test 3 different adaptation classes
        per_cls_acc={}
        for seed in range(3):  # 3 random seeds for reliability
            cm_s=DB(64,10,4,0.1,0.0001,0.002,device,max_count=100)
            cm_f=DB(64,10,4,0.1,0.0001,0.02,device,max_count=20)
            w=0.01
            
            # Collect 50 random samples of adaptation class
            adapt_mask=labels==adapt_class
            adapt_indices=list(np.where(adapt_mask)[0])
            np.random.seed(seed);np.random.shuffle(adapt_indices)
            adapt_indices=adapt_indices[:50]
            
            # Adapt on 50 samples
            m2=cifar10_resnet20();m2.load_state_dict(sd);m2.to(device).eval()
            f={};m2.avgpool.register_forward_hook(lambda m,i,o:f.__setitem__('x',o.flatten(1)))
            for j in adapt_indices:
                img=tf(data[j]).unsqueeze(0).to(device)
                with torch.no_grad():_=m2(img);x=f['x'];lgs=x@m2.fc.weight.data.T+m2.fc.bias.data
                sc_s=cm_s.predict(x);sc_f=cm_f.predict(x)
                lgs_s=lgs.cpu()+w*sc_s.cpu();lgs_f=lgs.cpu()+w*sc_f.cpu()
                ms=lgs_s.softmax(1).max(1)[0].item();mf=lgs_f.softmax(1).max(1)[0].item()
                if ms>mf:wp=lgs_s.softmax(1)
                else:wp=lgs_f.softmax(1)
                cm_s.fit(x,wp.to(device));cm_f.fit(x,wp.to(device));cm_s.update();cm_f.update()
                w=min(0.01*cm_s.count.mean().item()/10,0.5)
            
            # Test on ALL classes (freeze model, only TTA stats active)
            if seed==0:
                for cl in range(10):
                    mask=labels==cl;c0=0
                    for i in np.where(mask)[0][:200]:
                        img=tf(data[i]).unsqueeze(0).to(device)
                        with torch.no_grad():_=m2(img);x=f['x'];lgs=x@m2.fc.weight.data.T+m2.fc.bias.data
                        sc_s=cm_s.predict(x);sc_f=cm_f.predict(x)
                        final=lgs.cpu()+w*(sc_s.cpu()+sc_f.cpu())/2
                        c0+=int(final.argmax(1).item()==labels[i])
                    per_cls_acc[cl]=100*c0/200
        
        summary=' '.join(f'{per_cls_acc.get(cl,0):5.1f}' for cl in range(10))
        print(f'  Adapt-C{adapt_class}: {summary}')
