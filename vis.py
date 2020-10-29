import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import transforms
import os,sys
import torch.nn.parallel
import argparse,random
import numpy as np
from PIL import Image
from meta_model import *

num_cls = 101

transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_ft = resnet50(pretrained=True)
num_ftrs = model_ft.fc_classifier.in_features
model_ft.fc_classifier = nn.Linear(num_ftrs, num_cls)

ckpt = torch.load('model/model_best.pkl')
ckpt2 = {}
for key in ckpt:
    if not 'naive' in key:
        ckpt2[key.replace("module.", "")] = ckpt[key]
    else:
        print(key)
model_ft.load_state_dict(ckpt2)

model_ft = model_ft.cuda()
model_ft.eval()

f = open('list.txt')
names = []
for line in f:
    tmp = line.strip().split()
    names.append(tmp)

tot = len(names)
features = np.zeros((tot, 2048), dtype=np.float32)
weight = np.zeros((tot, tot, 2), dtype=np.float32)
for i in range(tot):
    for j in range(tot):
        img1 = Image.open("data/Food-101N_release/images/" + names[i][1])
        img2 = Image.open("data/Food-101N_release/images/" + names[j][1])
        img1 = transform(img1)
        img2 = transform(img2)
        img = torch.stack([img1, img2], dim=0)

        img = img.cuda()
       
        with torch.no_grad():
            feat, w = model_ft(img)
        feat = feat.cpu().numpy()
        w = w.cpu().numpy()

        features[i] = feat[0]
        features[j] = feat[1]
        weight[i, j, 0] = w[0]
        weight[i, j, 1] = w[1]
        print(i, i, tot)

np.save('features.npy', features)
np.save('weight.npy', weight)

