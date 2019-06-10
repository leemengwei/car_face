import os
import json
import numpy as np
import glob
import shutil
from tqdm import tqdm
from IPython import embed
from sklearn.model_selection import train_test_split
np.random.seed(41)
#0为背景
classname_to_id = {"angle": 1,"angle_r":2, "top":3, "top_r":4, "head":5}

class Lableme2CoCo:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 由json文件构建COCO
    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 构建类别
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        image = {}
        from labelme import utils
        img_x = utils.img_b64_to_arr(obj['imageData'])
        h, w = img_x.shape
        #h, w = img_x.shape[:-1]
        image['height'] = h
        image['width'] = w
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".png")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
#        embed()
        if 'head' in label:
            label = 'head'
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]


if __name__ == '__main__':
    labelme_train_path = "../train/"
    labelme_test_path = "../test/"
    saved_coco_path = "./cocos_here/"
    # 创建文件
    if not os.path.exists("%s/annotations/"%saved_coco_path):
        os.makedirs("%s/annotations/"%saved_coco_path)
    if not os.path.exists("%s/images/train2017/"%saved_coco_path):
        os.makedirs("%s/images/train2017"%saved_coco_path)
    if not os.path.exists("%s/images/test2017/"%saved_coco_path):
        os.makedirs("%s/images/test2017"%saved_coco_path)
    # 获取images目录下所有的joson文件列表
    train_jsons = glob.glob(labelme_train_path + "/*/*.json")
    test_jsons = glob.glob(labelme_test_path + "/*/*.json")
    train_path = train_jsons
    test_path = test_jsons
    print("train_n:", len(train_path), 'test_n:', len(test_path))

    l2c_train = Lableme2CoCo()
    l2c_test = Lableme2CoCo()
    train_instance = l2c_train.to_coco(train_path)
    test_instance = l2c_test.to_coco(test_path)
    l2c_train.save_coco_json(train_instance, '%s/annotations/instances_train2017.json'%saved_coco_path)
    l2c_test.save_coco_json(test_instance, '%s/annotations/instances_test2017.json'%saved_coco_path)
    print("copying...")
    for file in tqdm(train_path):
        shutil.copy(file.replace("json","png"),"%s/images/train2017/"%saved_coco_path)
    for file in tqdm(test_path):
        shutil.copy(file.replace("json","png"),"%s/images/test2017/"%saved_coco_path)
