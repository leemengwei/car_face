from tqdm import tqdm
import time
import os
import copy
import argparse
import pdb
import collections
import sys
import numpy as np
import torch
torch.manual_seed(44)
np.random.seed(44)
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torchvision
import object_detection_model as model
from object_detection_anchors import Anchors
import object_detection_losses as losses
from object_detection_dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from torch.utils.data import Dataset, DataLoader
import object_detection_coco_eval as coco_eval
import object_detection_csv_eval as csv_eval
import warnings
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib
from config import *
warnings.filterwarnings("ignore")
#assert torch.__version__.split('.')[1] == '4'
print('CUDA available: {}'.format(torch.cuda.is_available()))
#plt.ion()

def save_model(epoch, model, optimizer, scheduler, train_loss_history, train_mAP_history, validate_loss_history, validate_mAP_history, args):
    train_loss_history = list(train_loss_history)
    train_mAP_history = list(train_mAP_history)
    validate_loss_history = list(validate_loss_history)
    validate_mAP_history = list(validate_mAP_history)
    #Saving stuff:
    if args.parallel_mode:
        ckpt = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.module.state_dict(),
                'scheduler_state_dict': scheduler.module.state_dict(),
                'train_loss_history': train_loss_history,
                'train_mAP_history': train_mAP_history,
                'validate_loss_history': validate_loss_history,
                'validate_mAP_history': validate_mAP_history,
                }
    else:
        ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_history': train_loss_history,
                'train_mAP_history': train_mAP_history,
                'validate_loss_history': validate_loss_history,
                'validate_mAP_history': validate_mAP_history,
                }
    epoch_model_dir = './object_detection_logs_data_both_side_{}/csv_retinanet_epoch_{}.pt'.format(args.data_path.strip('./').split('_')[-1], epoch)
    last_model_dir = './object_detection_logs_data_both_side_{}/csv_retinanet_last.pt'.format(args.data_path.strip('./').split('_')[-1])
    best_model_dir = './object_detection_logs_data_both_side_{}/csv_retinanet_best.pt'.format(args.data_path.strip('./').split('_')[-1])
    torch.save(ckpt, last_model_dir)
    if args.save_model:
        print("Saving model at", epoch_model_dir)
        torch.save(ckpt, epoch_model_dir)
    else:
        pass   #not saving middle model...
    if epoch>1:
        #如果测集合上： loss最小或者 mAP最大
        if (validate_loss_history[-1] < np.array(validate_loss_history)[:-1].min()) or (validate_mAP_history[-1] > np.array(validate_mAP_history)[:-1].max()):
            print("*****************Saving current best model, epoch: %s"%epoch)
            torch.save(ckpt, best_model_dir)
        else:
            print("Not best, loss:%s>%s and mAP:%s<%s, Keep running..."%(validate_loss_history[-1], np.array(validate_loss_history[:-1]).min(), validate_mAP_history[-1], np.array(validate_mAP_history)[:-1].max()))
    else:
        pass   #Pass epoch 1
    return

def main(args=None):

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('-DP', '--data_path', required=True)
    parser.add_argument('-E', '--epochs', type=int, default=1000)
    parser.add_argument('-LR', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-RM', '--restart_model', type=str, default=None)
    parser.add_argument('-BS', '--batch_size', type=int, default=4)
    parser.add_argument('-V', '--visualization', action="store_true", default=False)
    parser.add_argument('-SM', '--save_model', action="store_true", default=False)
    parser.add_argument('-FX', '--flip_x', action="store_true", default=False)
    parser.add_argument('-TF', '--train_file', default='train.csv')
    parser.add_argument('-VF', '--val_file', default='val.csv')
    parser.add_argument('-CF', '--classes_file', help='Classes file', default="classes.csv")
    parser.add_argument('-PARA', '--parallel_mode', action="store_true", default=False)
    args = parser.parse_args(args)
    print(args)
    # Create the data loaders
    if args.flip_x:
        dataset_train = CSVDataset(train_file=args.data_path+"/%s"%args.train_file, class_list=args.data_path+"/%s"%args.classes_file, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    else:
        dataset_train = CSVDataset(train_file=args.data_path+"/%s"%args.train_file, class_list=args.data_path+"/%s"%args.classes_file, transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(train_file=args.data_path+"/"+args.val_file, class_list=args.data_path+"/%s"%args.classes_file, transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_train = AspectRatioBasedSampler(dataset_train, batch_size=args.batch_size, drop_last=False)
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler_train)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)
    # Create the model
    root_dir = "/".join(os.getcwd().split('/'))
    retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, root_dir=root_dir)
    retinanet = retinanet.cuda()
    optimizer = optim.Adam(retinanet.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)
    if args.parallel_mode:
        retinanet = torch.nn.DataParallel(retinanet)
        optimizer = torch.nn.DataParallel(optimizer)
        scheduler = torch.nn.DataParallel(scheduler)
    #Restart:
    if args.restart_model:
        print("Restart model loaded: %s"%args.restart_model)
        checkpoint = torch.load(args.restart_model)
        epoch = checkpoint['epoch']
        if args.parallel_mode:
            retinanet.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.module.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.module.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            retinanet.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_loss_history = checkpoint['train_loss_history']
        train_mAP_history = checkpoint['train_mAP_history']
        validate_loss_history = checkpoint['validate_loss_history']
        validate_mAP_history = checkpoint['validate_mAP_history']
        if args.visualization:
            plt.plot(validate_loss_history, linewidth=2, label="validate loss")
            plt.plot(validate_mAP_history, linewidth=2, label="validate mAP")
            plt.plot(train_loss_history, linewidth=2, label="train loss")
            plt.title("Training/Validating loss over epoch")
            plt.legend()
            plt.xlabel("Training epoches")
            plt.ylabel("Training/Valdation loss")
            plt.show()
            #plt.draw()
            #plt.pause(0.1)
        else:
            pass
    else:
        epoch = 0
        train_loss_history = collections.deque(maxlen=10000)
        train_mAP_history = collections.deque(maxlen=10000)
        validate_mAP_history = collections.deque(maxlen=10000)
        validate_loss_history = collections.deque(maxlen=10000)
    
    print('Num training datas: {}'.format(len(dataset_train)))
    validate_mAP = {}
    for epoch in range(epoch, args.epochs):
        #TRAIN--------------------------------------------------------------------
        retinanet.train()
        if args.parallel_mode:
            retinanet.module.freeze_bn()
        else:
            retinanet.freeze_bn() 
        iter_losses = []
        pbar_train = tqdm(dataloader_train, total=int(len(dataloader_train.dataset)/sampler_train.batch_size))
        for iter_num, (data) in enumerate(pbar_train):
            optimizer.zero_grad()
            classification_loss, regression_loss = retinanet([data['img'].float().cuda(), data['annot'].cuda()])
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            if args.parallel_mode:
                optimizer.module.step()
            else:
                optimizer.step()
            iter_losses.append(float(loss))
            pbar_train.set_description('T-Ep:{}|Iter:{}|C:{:1.2f}|R:{:1.2f}|L:{:1.2f}|M:{:1.2f}'.format(epoch, iter_num, float(classification_loss), float(regression_loss), float(loss), np.mean(iter_losses)))
        if args.parallel_mode:
            scheduler.module.step(np.mean(iter_losses))	
        else:
            scheduler.step(np.mean(iter_losses))
        print("Training Complete.")
        #VAL---------------------------------------------------------------------------
        #Validate1
        validate_mAP = {"tmp":0}
        validate_mAP = csv_eval.evaluate(dataset_val, retinanet, score_threshold=CONFIDENCE_THRESHOLD)
        validate_mAP_score = 0
        for i in validate_mAP.keys():
            validate_mAP_score += (validate_mAP[i])
        #Validate2
        train_mAP_score=0
        '''
        train_mAP = csv_eval.evaluate(dataset_train, retinanet, score_threshold=CONFIDENCE_THRESHOLD)
        for i in train_mAP.keys():
        	train_mAP_score += (train_mAP[i][0])
        print(train_mAP, train_mAP_score)
        '''
        #Validate3
        retinanet.train()   #目前trianmode保持input有标签可以eval，freeze
        if args.parallel_mode:
            retinanet.module.freeze_bn()
        else:
            retinanet.freeze_bn() 
        val_iter_losses = []
        pbar_val = tqdm(dataloader_val, total=int(len(dataloader_val.dataset)/sampler_val.batch_size))
        for val_iter_num, val_data in enumerate(pbar_val):
            with torch.no_grad():
                val_classification_loss, val_regression_loss = np.array([999]), np.array([999])
                val_classification_loss, val_regression_loss = retinanet([val_data['img'].float().cuda(), val_data['annot'].cuda()])
                val_classification_loss = val_classification_loss.mean()
                val_regression_loss = val_regression_loss.mean()
                val_loss = val_classification_loss + val_regression_loss
                val_iter_losses.append(float(val_loss))
                pbar_val.set_description('V-Ep:{}|Iter:{}|C:{:1.2f}|R:{:1.2f}|L:{:1.2f}|M:{:1.2f}'.format(epoch, val_iter_num, float(val_classification_loss), float(val_regression_loss),float(val_loss), np.mean(val_iter_losses)))
        #OTHERS--------------------------------------------------------------------------
        #Apendings:	
        train_loss_history.append(np.mean(iter_losses))	
        train_mAP_history.append(train_mAP_score)	
        validate_loss_history.append(np.mean(val_iter_losses))	
        validate_mAP_history.append(validate_mAP_score)	
        #Save model:
        save_model(epoch, retinanet, optimizer, scheduler, train_loss_history, train_mAP_history, validate_loss_history, validate_mAP_history, args)
        #Visualization:
        if args.visualization:
            plt.clf()
            train_loss_history[0] = validate_loss_history[0]*1.5
            plt.plot(train_loss_history, label="train_loss_history")
            #plt.plot(train_mAP_history, label="train_mAP_history")
            plt.plot(validate_loss_history, label="validate_loss_history")
            plt.plot(validate_mAP_history, label="validate_mAP_history")
            plt.legend()
            plt.draw()
            plt.pause(0.01)
        print("Epoch done~ \n")

if __name__ == '__main__':
    main()
