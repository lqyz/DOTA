"""BD-TTA Level 3: Chaos Stream (CIFAR-10-C, 19 corruptions shuffled, 95k)."""
import torch, torchvision.transforms as T, numpy as np, os, sys, random
sys.path.insert(0,'/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
from pytorch_cifar_models.resnet import cifar10_resnet20
sys.path.insert(0,'/root/DOTA')
from bdtta import BDTTA
from tqdm import tqdm

device='cuda';C,D,G=10,64,4
tf=T.Compose([T.ToTensor(),T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
path='/root/data/picture/CIFAR-10-C';labels=np.load(os.path.join(path,'labels.npy'))
sd=torch.load('/root/.cache/torch/hub/checkpoints/cifar10_resnet20-4118986f.pt',map_location='cpu')
sd={k.replace('module.',''):v for k,v in sd.items()}

A,B=[],[]
for fn in sorted([f for f in os.listdir(path) if f.endswith('.npy') and f!='labels.npy']):
    d=np.load(os.path.join(path,fn))[45000:50000]
    for i in range(5000):A.append(d[i]);B.append(labels[5000+i])
idx=list(range(len(A)));random.shuffle(idx)
A=[A[i]for i in idx];B=[B[i]for i in idx]
print(f'Chaos stream: {len(A)} samples')

m=cifar10_resnet20();m.load_state_dict(sd);m.to(device).eval()
feats={};m.avgpool.register_forward_hook(lambda m,i,o:feats.__setitem__('x',o.flatten(1)))

# Baseline
c0=0
with torch.no_grad():
    for i in tqdm(range(len(A)),desc='Baseline',leave=False):
        m(tf(A[i]).unsqueeze(0).to(device));c0+=int(feats['x']@m.fc.weight.data.T+m.fc.bias.data).argmax(1).item()==B[i]
baseline=100*c0/len(A);print(f'Baseline: {baseline:.2f}%')

# BD-TTA
tta=BDTTA(D,C,G=G,device=device);c=0
with torch.no_grad():
    for i in tqdm(range(len(A)),desc='BD-TTA',leave=False):
        m(tf(A[i]).unsqueeze(0).to(device));x=feats['x'];lgs=x@m.fc.weight.data.T+m.fc.bias.data
        wl=tta.step(x,lgs);c+=int(wl.argmax(1).item()==B[i])
tta_acc=100*c/len(A);print(f'BD-TTA: {tta_acc:.2f}%')
np.save('/root/DOTA/result_chaos.npy',{'baseline':baseline,'tta':tta_acc})
