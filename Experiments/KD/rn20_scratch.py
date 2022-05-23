import sys
import torch
from torch import nn 
import torch.nn.functional as F
import torchvision
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,RandomResizedCrop,RandomHorizontalFlip,ColorJitter,RandomRotation
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import torchvision.models as models
from smilelogging import Logger
from smilelogging import argparser as parser
from resnet import resnet20
#import argparse


#parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type = str, default = None, help = "Path to the model")
parser.add_argument("-e", "--epoch", type = int, default = 240, help = "Number of trianing epoches")
parser.add_argument("-b", "--batch_size", type = int, default = 256, help = "Training batch size")
parser.add_argument("-w", "--weight_decay", type = float, default = 5e-4, help = "Optimizer weight decay")
parser.add_argument("-m", "--momentum", default=0.9, type = float, help = 'Momentum')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help = 'Initial learning rate', dest='lr')
args = parser.parse_args()
logger = Logger(args)

from torchvision.datasets import CIFAR100

transform = transforms.Compose([ RandomResizedCrop(224),
	                	 RandomHorizontalFlip(),
		              	 #RandomRotation(30),
		              	 #ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                  		 ToTensor(),
                  		 Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),]) 

train_set = CIFAR100("~/data",train = True, download = True, transform = transform)
test_set = datasets.CIFAR100("/data",train=False,download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 2, shuffle = True)
test_loader = DataLoader(test_set, batch_size=32, num_workers = 2, shuffle = False)

if args.path:
  model = resnet20(num_classes = 100)
  ckpt = torch.load(args.path)
  model.load_state_dict(ckpt['model_state_dict'])

else:
  model = resnet20(num_classes = 100)
model.cuda()
#snet.train()

torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

def epoch(loader, model, opt = None):
  total_corr, total_loss = 0.,0.
  for x,y in loader:
    x,y = x.cuda(), y.cuda()
   
    yp = model(x)
    loss = nn.CrossEntropyLoss()(yp, y) 

    if opt:
      opt.zero_grad()
      loss.backward()
      opt.step()
    total_corr += (yp.max(dim=1)[1] == y).sum().item()
    total_loss += loss.item() * x.shape[0]
  avg_loss = total_loss / len(loader.dataset)
  acc = total_corr / len(loader.dataset) 
  return avg_loss, acc

def eval_net(string, loader = test_loader, model = model):
  correct = 0
  total = 0
  with torch.no_grad():
    for img,labels in loader:
        img, labels = img.cuda(), labels.cuda()
        # calculate outputs by running images through the network
        outputs = model(img)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

  print(f'Accuracy of the network on the {string} set: {100 * correct / total} %')

opt = optim.SGD(model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = args.momentum)
if args.path:
   opt.load_state_dict(ckpt['optimizer_state_dict'])
else:
   pass

scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones = [60, 120, 180, 240], gamma = 0.1)
print("Epoch", "Loss","Accuracy", sep = "\t")
for t in range(args.epoch):
    model.train()
    loss, acc = epoch(train_loader, model = model, opt = opt)
    scheduler.step()
    print(t,*("{:.5f}".format(i)for i in (loss,acc)), sep = "\t")
    #torch.save(model,'./scrach_rn20.pth')
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }, './ckpt_rn20.pth')
torch.save(model,'./rn20_scrach.pth')

eval_net("train")

eval_net("test")

print(args)
# https://github.com/MingSun-Tse/Regularization-Pruning/tree/a4044028edaacca4e7a063c602170b2fffad0a84 loader, batch size, weight decay, learning rate + adjust
# https://github.com/openai/CLIP/blob/main/clip/clip.py transform
# https://github.com/SforAiDl/KD_Lib/blob/master/KD_Lib/KD/vision/vanilla/vanilla_kd.py distillation temperature
# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py batch_size
# https://intellabs.github.io/distiller/knowledge_distillation.html distillation temperature
# https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb text
# https://arxiv.org/pdf/1905.04753.pdf weight_decay
