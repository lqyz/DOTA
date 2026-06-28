"""BD-TTA Level 3: ImageNet-C 1000-class, 3 hardest corruptions (Sev5)."""
import torch, torchvision, numpy as np, os, random, sys
from tqdm import tqdm
from PIL import Image
import torchvision.datasets as datasets

device='cuda'
model=torchvision.models.resnet50(weights='IMAGENET1K_V1').to(device).eval()
feats={};model.avgpool.register_forward_hook(lambda m,i,o:feats.__setitem__('x',o.flatten(1)))
tf=torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])
W,b=model.fc.weight.data,model.fc.bias.data
C,D,G=1000,2048,32

sys.path.insert(0,'/root/DOTA')
from bdtta import BDTTA

base='/root/data/picture/ImageNet-C'
corrs=['gaussian_noise','shot_noise','glass_blur'];sev='5'
imgs,labels=[],[]
for corr in corrs:
    ds=datasets.ImageFolder(f'{base}/{corr}/{sev}')
    for img_path,lbl in ds.samples:imgs.append(img_path);labels.append(lbl)
random.seed(42);combined=list(zip(imgs,labels));random.shuffle(combined);imgs,labels=zip(*combined)
print(f'ImageNet-C: {len(imgs)} images, {C} classes')

c0=0
with torch.no_grad():
    for i in tqdm(range(len(imgs)),desc='Baseline',leave=False):
        img=tf(Image.open(imgs[i]).convert('RGB')).unsqueeze(0).to(device)
        c0+=int(model(img)[0].argmax(0).item()==labels[i])
baseline=100*c0/len(imgs);print(f'Baseline: {baseline:.2f}%')

tta=BDTTA(D,C,G=G,sigma=1.0,W_init=W,device=device)
c,sw,fw=0,0,0;wv=0.25
for i in tqdm(range(len(imgs)),desc='BD-TTA',leave=False):
    img=tf(Image.open(imgs[i]).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():_=model(img);x=feats['x'];lgs=(x@W.T+b)[0]
    sc_s=tta._predict_branch(tta.slow,x);sc_f=tta._predict_branch(tta.fast,x)
    ls=lgs.cpu()+wv*sc_s.cpu().squeeze(0);lf=lgs.cpu()+wv*sc_f.cpu().squeeze(0)
    ms=ls.softmax(0).max(0).values.item();mf=lf.softmax(0).max(0).values.item()
    if ms>mf:sw+=1;wl=ls
    else:fw+=1;wl=lf
    c+=int(wl.argmax(0).item()==labels[i])
    tta._fit_branch(tta.slow,x);tta._fit_branch(tta.fast,x);tta._update_branch(tta.slow);tta._update_branch(tta.fast)
tta_acc=100*c/len(imgs);print(f'BD-TTA: {tta_acc:.2f}%  slow={100*sw/(sw+fw):.0f}%')
np.save('/root/DOTA/result_imagenet_c.npy',{'baseline':baseline,'tta':tta_acc})
