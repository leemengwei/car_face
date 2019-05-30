import time
import os
import copy
import argparse
import pdb
import collections
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision

import model
from anchors import Anchors
import losses
from dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader

import coco_eval
import csv_eval
import warnings
warnings.filterwarnings("ignore")
assert torch.__version__.split('.')[1] == '4'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

print('CUDA available: {}'.format(torch.cuda.is_available()))

dataset_val = CSVDataset(train_file="val.csv", class_list="classes.csv", transform=transforms.Compose([Normalizer(), Resizer()]))
retinanet = torch.load("./logs/csv_retinanet_139.pt").cuda()
retinanet.eval()
map = csv_eval.evaluate(dataset_val,retinanet)
print(map)
