import pandas as pd
import cv2
import os.path as osp
import numpy as np
import json
import os

def save(images, annotations, name):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations
    category = [
            {'id': 1, 'name': '停车场', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 2, 'name': '停车让行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 3, 'name': '右侧行驶', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 4, 'name': '向左和向右转弯', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 5, 'name': '大客车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 6, 'name': '左侧行驶', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 7, 'name': '慢行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 8, 'name': '机动车直行和右转弯', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 9, 'name': '注意行人', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 10, 'name': '环岛行驶', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 11, 'name': '直行和右转弯', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 12, 'name': '禁止大客车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 13, 'name': '禁止摩托车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 14, 'name': '禁止机动车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 15, 'name': '禁止非机动车通行', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 16, 'name': '禁止鸣喇叭', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 17, 'name': '立交直行和转弯行驶', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 18, 'name': '限制速度40公里每小时', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 19, 'name': '限速30公里每小时', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 20, 'name': '鸣喇叭', 'supercategory': 'jiaotongbiaozhi'},
          ]
    ann['categories'] = category
    json.dump(ann, open(os.path.join(data_dir, 'jiaotongbiaozhi_448_448_{}.json'.format(name)), 'w'), indent=2)

train_csv = "../train_label_fix.csv"
img_root = '../images/'
data_dir = './'
patch_img_root = 'train_patch_448_448/'
if not os.path.exists(patch_img_root):
  os.makedirs(patch_img_root)  
df = pd.read_csv(train_csv)

img_names = df['filename'].tolist()
xmin_list = df['X1'].tolist()
xmax_list = df['X3'].tolist()
ymin_list = df['Y1'].tolist()
ymax_list = df['Y3'].tolist()
class_label = df['type'].tolist()


json_images = []
json_annos = []
image_id = -1
idx = 1
for i, img_name in enumerate(img_names):
    print('%.2f%%' % (i/20258 * 100))
    path = img_root + img_name
    img = cv2.imread(path)
    assert img is not None
    height, width, _ = img.shape
    # height, width = 1800, 3000
    bbox_xmin = xmin_list[i]
    bbox_xmax = xmax_list[i]
    bbox_ymin = ymin_list[i]
    bbox_ymax = ymax_list[i]
    bbox_w = bbox_xmax - bbox_xmin
    bbox_h = bbox_ymax - bbox_ymin
    # 256*256
    crop_size = 224
    if bbox_xmin + int( bbox_w / 2)> crop_size and (3200-bbox_xmax) + int(bbox_w / 2) > crop_size:
      x_start = bbox_xmin - (crop_size - int(bbox_w / 2) - 1)
    elif bbox_xmin < crop_size - int(bbox_w / 2) - 1 and (3200-bbox_xmax) + int(bbox_w / 2) > crop_size:
      x_start = 0
    elif bbox_xmin > crop_size - int(bbox_w / 2) - 1 and (3200-bbox_xmax) + int(bbox_w / 2) < crop_size:
      x_start = 3200-crop_size * 2
    else:
      assert "fuck sample!!!"
    
    if bbox_ymin > crop_size - int(bbox_h / 2) - 1 and (1800-bbox_ymax) + int(bbox_h / 2)>crop_size:
      y_start = bbox_ymin - (crop_size - int(bbox_h / 2) - 1)
    elif bbox_ymin < crop_size - int(bbox_h / 2) - 1 and (1800-bbox_ymax) + int(bbox_h / 2)>crop_size:
      y_start = 0
    elif bbox_ymin > crop_size - int(bbox_h / 2) - 1 and (1800-bbox_ymax) + int(bbox_h / 2)<crop_size:
      y_start = 1800-crop_size * 2
    else:
      assert "fuck sample!!!"

    
    # crop_need_w = np.round((666 - 2 * bbox_w) / 2)
    # crop_need_h = np.round((400 - 2 * bbox_h) / 2)
    # x_crop_min = int(max(0, xmin_list[i] - crop_need_w))
    # x_crop_max = int(min(xmax_list[i] + crop_need_w, width))
    # y_crop_min = int(max(0, ymin_list[i] - crop_need_h))
    # y_crop_max = int(min(ymax_list[i] + crop_need_h, height))

    # print(x_crop_min, y_crop_min, x_crop_max, y_crop_max)
    img_patch = img[y_start: y_start+crop_size*2, x_start: x_start+crop_size*2, :]
    # print(img_patch.shape)
    assert img_patch.shape == (crop_size*2, crop_size*2, 3)
    
    # img_patch = np.zeros([400, 666, 3])
    target_path = osp.join(patch_img_root, img_name)
    cv2.imwrite(target_path, img_patch)
    image_id += 1
    image = {'file_name': img_name, 'width': img_patch.shape[1], 'height': img_patch.shape[0], 'id': image_id}
    json_images.append(image)
    ann = {'segmentation': [[]], 'area': bbox_w * bbox_h, 'iscrowd': 0, 'image_id': image_id,
           'bbox': [bbox_xmin - x_start, bbox_ymin - y_start, bbox_w, bbox_h], 'category_id': class_label[i], 'id': idx, 'ignore': 0}
    idx += 1
    json_annos.append(ann)

save(json_images, json_annos, "train")