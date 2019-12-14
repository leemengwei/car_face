import config

from IPython import embed
import numpy as np
import argparse
import sys, os
from camera import camera
import front_position_algorithm_A as A
import threading
import glob
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import seat_merge

def car_merge(cars, A_program, B_program, C_program, D_program):
    #删除old_days索引
    cars.sort()
    #cars.pop()
    position_score_by_vote = 0
    position_score_by_union = 0
    number_score_by_vote = 0
    number_score_by_union = 0
    all_num = 0
    for idx, car in enumerate(cars[54:]):
        print("\nNEW PIC" ,idx, "of", len(cars), car)
        images_both_side = glob.glob(car+"/*.png")
        images_both_side.sort()
        try:
            images_left = images_both_side[:3]
            images_right = images_both_side[3:6]
            images_back = images_both_side[6:]
            if len(images_back)==4:
                images_back = images_back[::2]
        except:
            print("Not enough image in %s"%car)
        print("Lefts:%s\n Rights:%s\n Back:%s\n"%(images_left, images_right, images_back))
        pos1 = pos2 = pos3 = pos4 = pos5 = pos6 = pos7 = pos8 = []
        try:
            assert len(images_left)==len(images_right), "left have %s, right have %s"%(len(images_left), len(images_right))
            assert len(images_back)==2, "back have %s"%len(images_back)
        except Exception as e:
            print(e)
            continue
        #LEFTS:
        preds = []
        if config.VISUALIZATION:
            plt.figure()
        A_image_data = camera.get_image_data(images_left[0])
        pos1, A_plt = A_program.self_logic(A_image_data, CONFIDENCE_THRESHOLD)
        if config.VISUALIZATION:
            plt.figure()
        A_image_data = camera.get_image_data(images_left[1])
        pos2, A_plt = A_program.self_logic(A_image_data, CONFIDENCE_THRESHOLD)
        if config.VISUALIZATION:
            plt.figure()
        A_image_data = camera.get_image_data(images_left[2])
        pos3, A_plt = A_program.self_logic(A_image_data, CONFIDENCE_THRESHOLD)
        #RIGHTS:
        if config.VISUALIZATION:
            plt.figure()
        A_image_data = camera.get_image_data(images_right[0])
        pos4, A_plt = B_program.self_logic(A_image_data, CONFIDENCE_THRESHOLD)
        if config.VISUALIZATION:
            plt.figure()
        A_image_data = camera.get_image_data(images_right[1])
        pos5, A_plt = B_program.self_logic(A_image_data, CONFIDENCE_THRESHOLD)
        if config.VISUALIZATION:
            plt.figure()
        A_image_data = camera.get_image_data(images_right[2])
        pos6, A_plt = B_program.self_logic(A_image_data, CONFIDENCE_THRESHOLD)
        #BACKS:
        if config.VISUALIZATION:
            plt.figure()
        A_image_data = camera.get_image_data(images_back[0])
        pos7, A_plt = C_program.self_logic(A_image_data, config.BACK_CONFIDENCE_THRESHOLD)
        if config.VISUALIZATION:
            plt.figure()
        A_image_data = camera.get_image_data(images_back[1])
        pos8, A_plt = D_program.self_logic(A_image_data, config.BACK_CONFIDENCE_THRESHOLD)

        car_result_union = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, method = "union")
        car_result_vote = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, method = "vote")
        car_result_front_and_back = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, method = "front_and_back")
        #Offline yield txt result file.
        with open(car+"/python_txt.txt", 'w') as f:
            f.write("front:%s\n"% car_result_vote)
            f.write("front and back:%s\n"%car_result_front_and_back)
            f.write("backleft:%s\n"%pos7)
            f.write("backright:%s\n"%pos8)


        car_result_label = list(set(np.array(os.popen("cat %s/*.json|grep head|grep label|cut -c 21|sort|uniq"%car.replace(' ','\ ')).read().split()).astype(int)))
        print("Label this car:", car_result_label)
        if len(car_result_label)==0:
            print("No label file for this, pass")
        else:
            car_result_label = set(car_result_label+[1])
            all_num += 1
            #判断不同融合方法的好坏：
            if set(car_result_union) == car_result_label:
                position_score_by_union += 1
            if set(car_result_vote) == car_result_label:
                position_score_by_vote += 1
            if len(set(car_result_union)) == len(car_result_label):
                number_score_by_union += 1
            if len(set(car_result_vote)) == len(car_result_label):
                number_score_by_vote += 1
            print("PbyU:", position_score_by_union, "PbyV:", position_score_by_vote, "NbyU:", number_score_by_union, "NbyV:", number_score_by_vote, "All:", all_num)
        input()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        plt.close()
        #embed()
        sys.stdout.flush()
    try:
        print("Position score by union:", position_score_by_union/all_num)
        print("Position score by vote:", position_score_by_vote/all_num)
        print("Number score by union:", number_score_by_union/all_num)
        print("Number score by vote:", number_score_by_vote/all_num)
    except:
        print("Prohaps too few to print")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Double side...")
    args = parser.parse_args()
     
    CONFIDENCE_THRESHOLD = config.get_confidence()
    print("Using confidence:%s"%CONFIDENCE_THRESHOLD)
    #Initialization:
    #A:
    print("Initializing front camera A...")
    root_dir = "/".join(os.getcwd().split('/'))
    A_program = A.A(root_dir, "left")
    #B:
    print("Initializing front camera B...")
    B_program = A.A(root_dir, "right")
    #C:
    print("Initializing front camera C...")
    C_program = A.A(root_dir, "backleft")
    #D:
    print("Initializing front camera D...")
    D_program = A.A(root_dir, "backright")

    car_to_car_dir = config.CAR_TO_CAR_DIR
    cars = glob.glob(car_to_car_dir+"/*")
    car_merge(cars, A_program, B_program, C_program, D_program)

