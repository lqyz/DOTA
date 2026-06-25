"""CIFAR-100-C baseline accuracy."""
import torch, numpy as np, torchvision.transforms as T, os, sys
from tqdm import tqdm
sys.path.insert(0,'/root/.cache/torch/hub/chenyaofo_pytorch-cifar-models_master')
from pytorch_cifar_models.resnet import cifar100_resnet20

device='cuda'
tf=T.Compose([T.ToTensor(),T.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761))])
path='/root/data/picture/CIFAR-100-C';labels=np.load(os.path.join(path,'labels.npy'))
sd=torch.load('/root/.cache/torch/hub/checkpoints/cifar100_resnet20-23dac2f1.pt',map_location='cpu')
sd={k.replace('module.',''):v for k,v in sd.items()}

r={}
for fn in sorted([f for f in os.listdir(path) if f.endswith('.npy') and f!='labels.npy']):
    data=np.load(os.path.join(path,fn));m=cifar100_resnet20();m.load_state_dict(sd);m.to(device).eval()
    c=0
    with torch.no_grad():
        for i in tqdm(range(50000),desc=fn[:14],leave=False):
            c+=int(m(tf(data[i]).unsqueeze(0).to(device)).argmax(1).item()==labels[i])
    r[fn.replace('.npy','')]=100*c/50000;print(f'{fn[:14]}: {r[fn.replace(".npy","")]:.2f}%')
torch.save(r,'/root/DOTA/cifar100_base.pt')
print(f'\nAVERAGE: {sum(r.values())/len(r):.2f}%')
