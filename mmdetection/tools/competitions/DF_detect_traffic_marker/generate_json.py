import json
import os
from glob import glob
# import cv2
from tqdm import tqdm


data_dir = './test_patch_448_448/'

def save(images, annotations, name):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    category = [
            {'id': 1, 'name': 'Park', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 2, 'name': 'Stop to give way', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 3, 'name': 'Keep Right', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 4, 'name': 'Left and right turns', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 5, 'name': 'Bus passage', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 6, 'name': 'left driving', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 7, 'name': 'slow down', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 8, 'name': 'Motor vehicles go straight and turn right', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 9, 'name': 'Watch For Pedestrians', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 10, 'name': 'roundabout', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 11, 'name': 'Go straight and turn right', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 12, 'name': 'No buses allowed', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 13, 'name': 'No motorcycles allowed', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 14, 'name': 'No Motor Vehicles', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 15, 'name': 'No non-motor vehicles allowed', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 16, 'name': 'No Honking', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 17, 'name': 'Go straight and turn at the overpass', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 18, 'name': '40 kilometers', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 19, 'name': '30 kilometers', 'supercategory': 'jiaotongbiaozhi'},
            {'id': 20, 'name': 'honking', 'supercategory': 'jiaotongbiaozhi'},

    ]
    ann['categories'] = category
    json.dump(ann, open('jiaotongbiaozhi_448_448{}.json'.format(name), 'w'))

# noobject_img = ["76ebb78be6534a70a974c684b28b5c64.jpg",
#                 "7234a04b9395469fbb1080a7aa7b6768.jpg",
#                 "f59b9c04f7f2488a9155fd1c2dba9bdd.jpg",
#                 "ded57aafa4784aa68f836d7319110960.jpg",
#                 "86b483028d33472f89d6c6cecc19a809.jpg",
#                 "187c579c789949a889caaa7832f0facc.jpg",
#                 "a468fca03af7419189de25d17ae083ec.jpg",
#                 "974f4483761a4ae79636e56e57e8ba11.jpg",
#                 "c62937d3dfbe4e18ab7995317bafd0a9.jpg"]
noobject_img = []
def test_dataset(im_dir):
    im_list = glob(os.path.join(im_dir, '*.jpg'))
    print(len(im_list))
    idx = 1
    image_id = 20190000000
    images = []
    annotations = []
    for im_path in tqdm(im_list):
        if os.path.split(im_path)[-1] in noobject_img:
            print(im_path)
            continue
        h, w, = 448, 448
        image_id += 1
        image = {'file_name': os.path.split(im_path)[-1], 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[10, 10, 20, 20]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            # bbox[] is x,y,w,h
            # left_top
            # seg.append(bbox[0])
            # seg.append(bbox[1])
            # # left_bottom
            # seg.append(bbox[0])
            # seg.append(bbox[1] + bbox[3])
            # # right_bottom
            # seg.append(bbox[0] + bbox[2])
            # seg.append(bbox[1] + bbox[3])
            # # right_top
            # seg.append(bbox[0] + bbox[2])
            # seg.append(bbox[1])
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    save(images, annotations, 'test')
    
if __name__ == '__main__':
    test_dir = data_dir
    test_dataset(test_dir)