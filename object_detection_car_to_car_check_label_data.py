from IPython import embed
import numpy as np
import argparse
import sys, os
import glob
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import json
import collections

def check_label_names(side, jsons_side):
    #Check specific label:
    for idx,json_side in enumerate(jsons_side):
        try:
            data = json.load(open(json_side,"r",encoding="utf-8"))
        except FileNotFoundError:
            #print("No json, no %s"%json_side)
            return
        labels_this_json = [data['shapes'][i]['label'] for i in range(len(data['shapes']))]
        side_target_labels = left_target_labels if side=="left" else right_target_labels
        wrong_angle_name = "angle_r" if side =="left" else "angle"
        right_angle_name = "angle" if side =="left" else "angle_r"
        wrong_top_name = "top_r" if side =="left" else "top"
        right_top_name = "top" if side =="left" else "top_r"
        wrong_head_name = "head4" if side =="left" else "head3"
        right_head_name = "head5"
        #重复标签警告：
        if len(list(set(labels_this_json))) != len(labels_this_json):
            print("Duplicate labels in %s"%json_side)
            pass
        #无头警告：
        if 'head' not in str(labels_this_json):
            print("No head! Suspicious! %s"%json_side)
            pass
        #无标识物警告：
        if ('angle' not in str(labels_this_json) and 'top' not in str(labels_this_json)):
            print("No REFS! Suspicious! %s"%json_side)
            pass
        #标签侧名称*错误*:
        if len(set(side_target_labels+labels_this_json)) != len(side_target_labels):
            print("***Error! Wrong labels appeared in %s"%json_side, side)
            os.popen("sed -i 's/\"{wrong_angle_name}\"/\"{right_angle_name}\"/g' {json_name}".format(json_name=json_side.replace(" ","\ "), wrong_angle_name=wrong_angle_name, right_angle_name=right_angle_name, overwrite=json_side.replace(" ","\ ")))
            os.popen("sed -i 's/\"{wrong_top_name}\"/\"{right_top_name}\"/g' {json_name}".format(json_name=json_side.replace(" ","\ "), wrong_top_name=wrong_top_name, right_top_name=right_top_name, overwrite=json_side.replace(" ","\ ")))
            os.popen("sed -i 's/\"{wrong_head_name}\"/\"{right_head_name}\"/g' {json_name}".format(json_name=json_side.replace(" ","\ "), wrong_head_name=wrong_head_name, right_head_name=right_head_name, overwrite=json_side.replace(" ","\ ")))
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Double side...")
    args = parser.parse_args()

    #car_to_car_dir = "/home/user/Data/good/"
    car_to_car_dir = "/home/user/storage/data/car_head/20190429/"
    cars = glob.glob(car_to_car_dir+"/*")
    cars.sort()
    left_target_labels = ['top', 'angle', 'head1', 'head2', 'head3', 'head5']  #左边不可能看到4 
    right_target_labels = ['top_r', 'angle_r', 'head1', 'head2', 'head4', 'head5']  #右边不可能看到3
    image_format = "png"
    for idx, car in enumerate(cars[:]):
        #Images:
        images_both_side = glob.glob(car+"/*.%s"%image_format)
        images_both_side.sort()
        assert len(images_both_side)==6, "Check dir:%s"%car
        #Jsons:
        jsons_both_side = glob.glob(car+"/*.json")
        jsons_both_side = 'UQ'.join(images_both_side).replace('.%s'%image_format, '.json').split('UQ')
        jsons_both_side.sort()
        #Check general:
        names_of_images = " ".join(images_both_side).replace(".%s"%image_format,'').split(' ')
        names_of_jsons = " ".join(jsons_both_side).replace(".json",'').split(' ')
        #Split left and right:
        images_left = images_both_side[:int(len(images_both_side)/2)]
        images_right = images_both_side[int(len(images_both_side)/2):]
        jsons_left = 'UQ'.join(images_left).replace(".%s"%image_format, ".json").split("UQ")
        jsons_right = 'UQ'.join(images_right).replace(".%s"%image_format, ".json").split("UQ")
        #embed()

        #Check json labels:
        check_label_names("left", jsons_left)
        check_label_names("right", jsons_right)

