import cv2
import json
import mmcv
import time
import argparse
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from IPython import embed
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result,init_detector

id_to_label = {1:"angle",2:"angle_r",3:"top",4:"top_r",5:"head"}

def parse_args():
    parser = argparse.ArgumentParser(description='Predict on test images')
    parser.add_argument('--cfg', help='train config file path')
    parser.add_argument('--weights', help='the dir to load trained models', required=True)
    parser.add_argument('--draw',default=True, help='if True,draw box on image and save images')
    parser.add_argument('--filter_',default=True,help='use binary classifier to filter test images')
    parser.add_argument('--thre',default=0.3,help='threshold for predict bboxes')
    args = parser.parse_args()
    return args

def predict():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.cfg)
    cfg.model.pretrained = None
    # construct the model and load checkpoint
    #model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    model = init_detector(args.cfg, args.weights, device='cuda:0')
    #_ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
    for image in tqdm(glob("/mfs/home/wangke/data/car_face/coco/images/val2017/*.png")):
        img = mmcv.imread(image)
        filename = image.split("/")[-1]
        start = time.time()
        result = inference_detector(model, img)
        print(time.time()-start)
        labels = np.concatenate([
                               np.full(bbox.shape[0], i, dtype=np.int32)
                              for i, bbox in enumerate(result)])
        #labels = labels + 1
        bboxes = np.vstack(result)
        for label,bbox in zip(labels,bboxes):
            threshold = bbox[-1]
            if threshold < 0.8:
                continue
            x1,y1 = bbox[0],bbox[1]
            x2,y2 = bbox[2],bbox[1]
            x3,y3 = bbox[2],bbox[3]
            x4,y4 = bbox[0],bbox[3]
            text = id_to_label[label+1]
            cv2.putText(img,text,(int(x1),int(y1-2)),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255))
            cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        status = cv2.imwrite("outputs/%s"%filename,img)
        #cv2.imshow("img",img)
        #cv2.waitKey(0)
        
if __name__ == "__main__":
    predict()


