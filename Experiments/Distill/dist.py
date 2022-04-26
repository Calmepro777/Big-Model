import sys
import os
#sys.path.insert(0, '/home/wu.qife/CLIP/CLIP/clip/')
sys.path.append('/home/wu.qife/CLIP/CLIP/')
from clip import clip
import torch
from torch import nn 
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,RandomResizedCrop,RandomHorizontalFlip,ColorJitter,RandomRotation
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torchvision.models as models

model, preprocess = clip.load("RN50")
model.cuda().eval()

from torchvision.datasets import CIFAR100
cifar100 = CIFAR100("~/data", transform=preprocess, download=True)

text_descriptions = [f"This is a photo of a {label}" for label in cifar100.classes]
text_tokens = clip.tokenize(text_descriptions).cuda()

transform = transforms.Compose([ Resize(224, interpolation=BICUBIC),
                  RandomResizedCrop(224),
	          RandomHorizontalFlip(),
		  RandomRotation(30),
		  ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
                  ToTensor(),
                  Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

train_set = datasets.CIFAR100("~/data",train = True, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=32, num_workers=4, shuffle=True)

snet = models.resnet18(num_classes = 100,pretrained=False)
snet.cuda()
snet.eval()

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

with torch.no_grad():
    text_features = model.encode_text(text_tokens).float()
text_features /= text_features.norm(dim=-1, keepdim=True)

def dst_epoch(loader,student,teacher,alp,opt=None,temp=1,text_features=text_features):
  total_corr, total_loss = 0.,0.
  for x,y in loader:
    x,y = x.cuda(),y.cuda()
    with torch.no_grad():
      image_features = teacher.encode_image(x).float()
      image_features /= image_features.norm(dim=-1, keepdim=True)
    tp = (100.0 * image_features @ text_features.T)#.softmax(dim=-1)
    sp = student(x)#.softmax(dim=-1)
    s_loss = nn.CrossEntropyLoss()(sp,y) 
    dst_loss = nn.KLDivLoss(reduction="batchmean",log_target=True)(F.log_softmax(sp/temp,dim=1),F.log_softmax(tp/temp,dim=1))
    loss = alp*s_loss+(1-alp)*dst_loss
    if opt:
      opt.zero_grad()
      loss.backward()
      opt.step()
    total_corr += (sp.max(dim=1)[1] == y).sum().item()
    total_loss += loss.item() * x.shape[0]
    #avg_loss = sum(losses)/len(losses)
  avg_loss = total_loss / len(loader.dataset)
  acc = total_corr/len(loader.dataset) 
  return avg_loss, acc

opt = optim.Adam(snet.parameters(),lr=1e-4,weight_decay=1e-7)
print("Loss","Accuracy",sep="\t")
for t in range(120):
    loss, acc = dst_epoch(train_loader,snet,model,0.1,opt=opt,temp=7)
    print(*("{:.5f}".format(i)for i in (loss,acc)),sep = "\t")
    torch.save(snet,'/home/wu.qife/CLIP/snet0.pth')
torch.save(snet,'/home/wu.qife/CLIP/snet0.pth')
