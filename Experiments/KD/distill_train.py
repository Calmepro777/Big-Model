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
#import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,RandomResizedCrop,RandomHorizontalFlip,ColorJitter,RandomRotation
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
#import torchvision.models as models
#import argparse
from smilelogging import Logger
from smilelogging import argparser as parser
from resnet import resnet20

#parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type = str, default = None, help = "Path to the model")
parser.add_argument("-e", "--epoch", type = int, default = 240, help = "Number of trianing epoches")
parser.add_argument("-b", "--batch_size", type = int, default = 256, help = "Training batch size")
parser.add_argument("-w", "--weight_decay", type = float, default = 5e-4, help = "Optimizer weight decay")
parser.add_argument("-t", "--distill_temp", type = int, default = 20, help = "Distillation_temperature")
parser.add_argument("-a", "--alpha", type = float, default = 0.1, help = "Balance the influence between teacher network and class labels")
parser.add_argument("--mi", "--milestone", type = int, dest = "mi", nargs = '+', default = [60, 90, 120, 150, 180, 210], help = "LR decay milestone")
parser.add_argument("-m", "--momentum", default=0.9, type = float, help = 'Momentum')
parser.add_argument("--teacher", type = str, dest = "tnet", default = "ViT-L/14", help = "Vision model for teacher network")#ViT-L/14
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help = 'Initial learning rate', dest='lr')
args = parser.parse_args()
logger = Logger(args)

tnet, preprocess = clip.load(args.tnet)
tnet.cuda().eval()

from torchvision.datasets import CIFAR100
cifar100 = CIFAR100("~/data", download = True)

text_descriptions = [f"A photo of a {label}" for label in cifar100.classes]
text_tokens = clip.tokenize(text_descriptions).cuda()

transform = transforms.Compose([ RandomResizedCrop(224),
	                	 RandomHorizontalFlip(),
		              	 #RandomRotation(20),
		              	 #ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                  		 ToTensor(),
                  		 Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),]) 

train_set = datasets.CIFAR100("~/data",train = True, download = True, transform = transform)
train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 1, shuffle = True)

if args.path:
  snet = resnet20(num_classes = 100)
  ckpt = torch.load(args.path)
  snet.load_state_dict(ckpt['model_state_dict'])

else:
  snet = resnet20(num_classes = 100)
snet.cuda()
#snet.train()

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

with torch.no_grad():
    text_features = tnet.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim = -1, keepdim = True)

def dst_epoch(loader, student, teacher, alpha, opt = None, temp=1, text_features = text_features):
  total_corr, total_loss = 0.,0.
  for x,y in loader:
    x,y = x.cuda(), y.cuda()
    with torch.no_grad():
      image_features = teacher.encode_image(x).float()
      image_features /= image_features.norm(dim = -1, keepdim = True)
    tp = (100.0 * image_features @ text_features.T)
    sp = student(x)
    s_loss = nn.CrossEntropyLoss()(sp, y) 
    distill_loss = pow(temp,2) * nn.KLDivLoss(reduction = "batchmean", log_target = True)(F.log_softmax(sp / temp, dim = 1), F.log_softmax(tp / temp, dim = 1))
    loss = alpha * s_loss + (1-alpha)*distill_loss
    if opt:
      opt.zero_grad()
      loss.backward()
      opt.step()
    total_corr += (sp.max(dim=1)[1] == y).sum().item()
    total_loss += loss.item() * x.shape[0]
  avg_loss = total_loss / len(loader.dataset)
  acc = total_corr / len(loader.dataset) 
  return avg_loss, acc

opt = optim.SGD(snet.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
if args.path:
   opt.load_state_dict(ckpt['optimizer_state_dict'])
else:
   pass

scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones = args.mi, gamma = 0.1)
print("Epoch", "Loss","Accuracy", sep = "\t")
for t in range(args.epoch):
    snet.train()
    loss, acc = dst_epoch(train_loader, snet, tnet, args.alpha, opt = opt, temp = args.distill_temp)
    scheduler.step()
    print(t,*("{:.5f}".format(i)for i in (loss,acc)), sep = "\t")
    #torch.save(snet,'./snet_sv_1.pth')
    torch.save({
            'model_state_dict': snet.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }, './ckpt_rn20_vitl14.pth')
torch.save(snet,'./rn20_vitl14.pth')

print(args)
# https://github.com/MingSun-Tse/Regularization-Pruning/tree/a4044028edaacca4e7a063c602170b2fffad0a84 loader, batch size, weight decay, learning rate + adjust
# https://github.com/openai/CLIP/blob/main/clip/clip.py transform
# https://github.com/SforAiDl/KD_Lib/blob/master/KD_Lib/KD/vision/vanilla/vanilla_kd.py distillation temperature
# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py batch_size
# https://intellabs.github.io/distiller/knowledge_distillation.html distillation temperature
# https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb text
# https://arxiv.org/pdf/1905.04753.pdf weight_decay
