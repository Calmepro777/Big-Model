import argparse
import sys 
sys.path.append('/home/wu.qife/CLIP/CLIP/')
import sam
from clip import clip
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn 
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from sam.model.wide_res_net import WideResNet
from sam.model.smooth_cross_entropy import smooth_crossentropy
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,RandomResizedCrop,RandomHorizontalFlip,ColorJitter,RandomRotation
from torchvision.transforms import InterpolationMode
from torchvision.datasets import CIFAR100
#from utility.log import Log
from sam.utility.initialize import initialize
from sam.utility.step_lr import StepLR
from sam.utility.bypass_bn import enable_running_stats, disable_running_stats

from sam import samopt



parser = argparse.ArgumentParser()
parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
parser.add_argument("-p", "--path", type = str, default = None, help = "Path to the model")
parser.add_argument("-a", "--alpha", type = float, default = 0.1, help = "Balance the influence between teacher network and class labels")
parser.add_argument("-t", "--distill_temp", type = int, default = 10, help = "Distillation_temperature")
args = parser.parse_args()

initialize(args, seed=42)

tnet, preprocess = clip.load("ViT-B/32")
tnet.cuda().eval()

cifar100 = CIFAR100("~/data",transform = preprocess, download = True)

text_descriptions = [f"A photo of a {label}" for label in cifar100.classes]
text_tokens = clip.tokenize(text_descriptions).cuda()
with torch.no_grad():
    text_features = tnet.encode_text(text_tokens).float()
    text_features /= text_features.norm(dim = -1, keepdim = True)

transform = transforms.Compose([ RandomResizedCrop(224),
	                	 RandomHorizontalFlip(),
                  		 ToTensor(),
                  		 Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

train_set = CIFAR100("~/data",train = True, download = True, transform = transform)
train_loader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 1, shuffle = True)

#log = Log(log_each=10)

snet = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=100)
if args.path:
  snet.load_state_dict(torch.load(args.path))
else:
  pass
snet.cuda()

base_optimizer = torch.optim.SGD

opt = samopt.SAM(snet.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones = [60, 120, 180], gamma = 0.1)

def dst_epoch(loader, student, teacher, alpha, opt, temp = 1, text_features = text_features):
  total_corr, total_loss = 0.,0.
  for x,y in loader:
    x,y = x.cuda(), y.cuda()
    with torch.no_grad():
      image_features = teacher.encode_image(x).float()
      image_features /= image_features.norm(dim = -1, keepdim = True)
    tp = (100.0 * image_features @ text_features.T)

    enable_running_stats(snet)
    sp = student(x)
    s_loss = smooth_crossentropy(sp, y, smoothing=args.label_smoothing)  
    distill_loss = nn.KLDivLoss(reduction = "batchmean", log_target = True)(F.log_softmax(sp / temp, dim = 1), F.log_softmax(tp / temp, dim = 1))
    loss = alpha * s_loss + (1 - alpha) * distill_loss
    loss.mean().backward()
    opt.first_step(zero_grad=True)

    disable_running_stats(snet)
    sp = student(x)
    s_loss = smooth_crossentropy(sp, y, smoothing=args.label_smoothing)  
    distill_loss = nn.KLDivLoss(reduction = "batchmean", log_target = True)(F.log_softmax(sp / temp, dim = 1), F.log_softmax(tp / temp, dim = 1))
    loss = alpha * s_loss + (1 - alpha) * distill_loss
    loss.mean().backward()
    opt.second_step(zero_grad=True)

    total_corr += (sp.max(dim=1)[1] == y).sum().item()
    total_loss += loss.item() * x.shape[0]
  avg_loss = total_loss / len(loader.dataset)
  acc = total_corr / len(loader.dataset) 
  return avg_loss, acc

print("Epoch", "Loss","Accuracy", sep = "\t")
for t in range(args.epochs):
    snet.train()
    loss, acc = dst_epoch(train_loader, snet, tnet, args.alpha, opt = opt, temp = args.distill_temp)
    scheduler.step()
    print(t,*("{:.5f}".format(i)for i in (loss,acc)), sep = "\t")
    #torch.save(snet.state_dict(),'./snet_sam.pth')
    torch.save({
            'model_state_dict': snet.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            }, './snet_sam.pth')
torch.save(snet,'./snet_sam_vitb32.pth')

