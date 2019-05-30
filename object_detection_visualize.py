import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
import sys
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from IPython import embed
from object_detection_dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
import object_detection_model as model
import object_detection_csv_eval as csv_eval
from config import *
#assert torch.__version__.split('.')[1] == '4'

def frame_detection(net, data, confidence):
    net = net.cuda()
    data = data.cuda()
    st = time.time()
    net.eval()
    with torch.no_grad():
        score, classification, transformed_anchors = net(data)
    idxs = np.where(score.cpu()>confidence)
    x1s = np.array([], dtype=int)
    y1s = np.array([], dtype=int)
    x2s = np.array([], dtype=int)
    y2s = np.array([], dtype=int)
    scores = np.array([])
    label_names = []
    for j in range(idxs[0].shape[0]):
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        label_name = int(classification[idxs[0][j]])
        x1s = np.hstack((x1s, x1))
        y1s = np.hstack((y1s, y1))
        x2s = np.hstack((x2s, x2))
        y2s = np.hstack((y2s, y2))
        scores = np.hstack((scores, score.cpu().numpy()[idxs[0][j]]))
        label_names.append(label_name)
    label_names = np.array(label_names)
    return x1s,y1s,x2s,y2s, scores,label_names, time.time()-st

def draw_caption(image, box, caption, score, args):
    classes_name_idx = ''.join(open(args.data_path+"/classes.csv").readlines()).strip('\n').split('\n')
    classes = {}
    for i in classes_name_idx:
        classes[str(i.split(',')[-1])] = i.split(',')[0]
    caption = classes[str(caption)]
    b = np.array(box).astype(int)
    text_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
    width, height = text_size[0][0], text_size[0][1]
    region = np.array([[b[0]-3, b[1]], [b[0]-3, b[1]-height-26], [b[0]+width+13, b[1]-height-26], [b[0]+width+13, b[1]]], dtype='int32')
    cv2.fillPoly(img=image, pts=[region], color=(0,0,255))                    
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(0,0,0), thickness=2)
    cv2.putText(image, str(score), (b[0], b[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1e-3 * image.shape[0], color=(255, 10, 10), thickness=2)

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('-DP', '--data_path', help='Path to data', default="./data_head/")
    parser.add_argument('-VF', '--val_file', help='Path to data', default="val.csv")
    parser.add_argument('-CF', '--classes_file', help='Classes file', default="classes.csv")
    parser.add_argument('-RM', '--restart_model', type=str, default=None)
    args = parser.parse_args(args)
    args.model = "./object_detection_logs_%s/csv_retinanet_best.pt"%args.data_path.strip('./') if args.restart_model is None else args.restart_model
    root_dir = "/".join(os.getcwd().split('/'))
    #Get data:
    dataset_val = CSVDataset(train_file=args.data_path+"/%s"%args.val_file, class_list=args.data_path+"/%s"%args.classes_file, transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    #Get model:
    checkpoint_data = torch.load(args.model)
    retinanet = model.resnet152(num_classes = dataset_val.num_classes(), pretrained=True, root_dir=root_dir)
    retinanet = retinanet.cuda()
    #retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.load_state_dict(checkpoint_data['model_state_dict'])
    retinanet = retinanet.cuda()
    retinanet.eval()
    if os.path.exists('./object_detection_log.log'):
        os.remove('./object_detection_log.log')
    for data in dataloader_val:
        with torch.no_grad():
            unnormalizer = UnNormalizer()
            img = copy.deepcopy(data['img'][0, :, :, :])
            img = np.array(255 * unnormalizer(img))
            img[img<0] = 0                                             
            img[img>255] = 255                                         
            img = np.transpose(img, (1, 2, 0))                         
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            if data['img'].float().cuda().shape!=(1,3,540,960):
                print("data shape problem, skip")
                continue
            x1s,y1s,x2s,y2s, scores,label_names, elapsed_time = frame_detection(retinanet, data['img'].float().cuda(), confidence=CONFIDENCE_THRESHOLD)
            print("On file:%s, %s object detected, Elapsed time:%s"%(data['file_name'][0], len(label_names), elapsed_time))
        for idx, label_name in enumerate(label_names):
            f = open("log.log", 'a')
            draw_caption(img, (x1s[idx], y1s[idx], x2s[idx], y2s[idx]), label_names[idx], scores[idx], args)
            cv2.rectangle(img, (x1s[idx], y1s[idx]), (x2s[idx], y2s[idx]), color=(0, 0, 255), thickness=2)
            tmp = "%s,%s,%s,%s,%s,%s\n"%(data['file_name'][0], x1s[idx], y1s[idx], x2s[idx], y2s[idx], dataloader_val.dataset.label_to_name(label_name))
            f.write(tmp)
            f.close()
        token = ''
        if 0 in label_names:
            token += "_"+dataloader_val.dataset.label_to_name(0)
        if 1 in label_names:
            token +="_"+dataloader_val.dataset.label_to_name(1)
        savename = './object_detection_output_data_both_side_%s/'%args.data_path.strip("./").split('_')[-1]+data['file_name'][0].split('/')[-1].strip('.jpg')+"%s_detected.jpg"%token
        print("save to %s"%savename)
        cv2.imwrite('%s'%savename, img)


if __name__ == '__main__':
    main()
