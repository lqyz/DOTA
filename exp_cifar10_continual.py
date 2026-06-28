"""BD-TTA Level 2: Continual Stream (CIFAR-10-C Sev5, 19 corruptions)."""
import torch, torchvision.transforms as T, numpy as np, os, sys
sys.path.insert(0,'/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
from pytorch_cifar_models.resnet import cifar10_resnet20
sys.path.insert(0,'/root/DOTA')
from bdtta import BDTTA
from tqdm import tqdm

device='cuda';C,D,G=10,64,4
tf=T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
path='/root/data/picture/CIFAR-10-C';labels=np.load(os.path.join(path,'labels.npy'))
sev5_labels=labels[5000:10000]
sd=torch.load('/root/.cache/torch/hub/checkpoints/cifar10_resnet20-4118986f.pt',map_location='cpu')
sd={k.replace('module.',''):v for k,v in sd.items()}

results={}
corrs=sorted([f for f in os.listdir(path) if f.endswith('.npy') and f!='labels.npy'])
tta=BDTTA(D,C,G=G,device=device)
m=cifar10_resnet20();m.load_state_dict(sd);m.to(device).eval()
feats={};m.avgpool.register_forward_hook(lambda m,i,o:feats.__setitem__('x',o.flatten(1)))

for ph,(fn,name) in enumerate([(fn,fn.replace('.npy','')) for fn in corrs]):
    tta.reset()  # Clear env bias from previous phase
    data=np.load(os.path.join(path,fn))[45000:50000]
    c0=0;c=0
    with torch.no_grad():
        for i in range(len(data)):
            m(tf(data[i]).unsqueeze(0).to(device))
            lgs=feats['x']@m.fc.weight.data.T+m.fc.bias.data
            c0+=int(lgs.argmax(1).item()==sev5_labels[i])
    with torch.no_grad():
        for i in tqdm(range(len(data)),desc=f'{ph:2d} {name[:14]}',leave=False):
            m(tf(data[i]).unsqueeze(0).to(device));x=feats['x']
            lgs=x@m.fc.weight.data.T+m.fc.bias.data;wl=tta.step(x,lgs);c+=int(wl.argmax(1).item()==sev5_labels[i])
    acc=100*c/len(data);base=100*c0/len(data);results[name]=(base,acc)
    print(f'Phase {ph:2d} {name:15s}: base={base:.2f}%  tta={acc:.2f}%')

av_base=sum(v[0] for v in results.values())/len(results)
av_tta=sum(v[1] for v in results.values())/len(results)
print(f'\nAVERAGE: base={av_base:.2f}%  tta={av_tta:.2f}%')
np.save('/root/DOTA/result_continual.npy',{'results':results,'avg_base':av_base,'avg_tta':av_tta})
