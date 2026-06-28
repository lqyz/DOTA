"""BD-TTA Continual: CIFAR-10-C Sev5, 19 corruptions sequential."""
import torch, torchvision.transforms as T, numpy as np, os, sys
sys.path.insert(0,'/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
from pytorch_cifar_models.resnet import cifar10_resnet20
sys.path.insert(0,'/root/DOTA')
from bdtta import BDTTA
from tqdm import tqdm

device='cuda';C,D,G=10,64,4
tf=T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
path='/root/data/picture/CIFAR-10-C';labels=np.load(os.path.join(path,'labels.npy'))
sev5l=labels[5000:10000]
sd=torch.load('/root/.cache/torch/hub/checkpoints/cifar10_resnet20-4118986f.pt',map_location='cpu')
sd={k.replace('module.',''):v for k,v in sd.items()}

corrs=sorted([f for f in os.listdir(path) if f.endswith('.npy') and f!='labels.npy'])
tta=BDTTA(D,C,G=G,sigma=0.1,device=device)
m=cifar10_resnet20();m.load_state_dict(sd);m.to(device).eval()
feats={};m.avgpool.register_forward_hook(lambda m,i,o:feats.__setitem__('x',o.flatten(1)))

results={}
for ph,(fn,name) in enumerate([(fn,fn.replace('.npy','')) for fn in corrs]):
    tta.reset()
    data=np.load(os.path.join(path,fn))[45000:50000]
    c0=0;c=0
    with torch.no_grad():
        for i in range(len(data)):
            m(tf(data[i]).unsqueeze(0).to(device))
            lgs=feats['x']@m.fc.weight.data.T+m.fc.bias.data
            c0+=int(lgs.argmax(1).item()==sev5l[i])
    with torch.no_grad():
        for i in tqdm(range(len(data)),desc=f'{ph:2d} {name[:14]}',leave=False):
            m(tf(data[i]).unsqueeze(0).to(device));x=feats['x']
            lgs=x@m.fc.weight.data.T+m.fc.bias.data;y=lgs.softmax(1)
            sc_s=tta._predict_branch(tta.slow,x);sc_f=tta._predict_branch(tta.fast,x)
            ls=lgs.cpu()+0.2*sc_s.cpu();lf=lgs.cpu()+0.2*sc_f.cpu()
            ms=ls.softmax(1).max(1)[0].item();mf=lf.softmax(1).max(1)[0].item()
            if ms>mf:wl=ls;wp=ls.softmax(1)
            else:wl=lf;wp=lf.softmax(1)
            c+=int(wl.argmax(1).item()==sev5l[i])
            tta._fit_branch(tta.slow,x,wp.to(device));tta._fit_branch(tta.fast,x,wp.to(device))
            tta._update_branch(tta.slow);tta._update_branch(tta.fast)
    acc=100*c/len(data);base=100*c0/len(data);results[name]=(base,acc)
    print(f'Phase {ph:2d} {name:15s}: base={base:.2f}%  tta={acc:.2f}%')

ab=sum(v[0] for v in results.values())/len(results)
at=sum(v[1] for v in results.values())/len(results)
print(f'\nAVERAGE: base={ab:.2f}%  tta={at:.2f}%')
np.save('/root/DOTA/result_continual.npy',{'results':results,'avg_base':ab,'avg_tta':at})
