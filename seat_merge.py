#-*-coding:utf-8-*-
import config

from IPython import embed
import numpy as np
import argparse
import sys, os
from camera import camera
import front_position_algorithm_A as A
#import side_position_algorithm_B as B  #will depracate in next version
import threading
import glob
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns

CONFIDENCE_THRESHOLD = config.get_confidence()

#注意，这里的所有融合函数唯一可能要考虑的问题是对0的剔除，至于诸如“有车就一定有人”的策略已经在单帧就实现完毕了。
def seat_merge(A_prediction, B_prediction):
    predictions_merged = set(A_prediction)|set(B_prediction) 
    print("Should not be here!")
    sys.exit()
    return list(predictions_merged)

def seat_merge_spatial(A_prediction, B_prediction):
    predictions_merged = set(A_prediction)|set(B_prediction)
    sys.stdout.flush()
    predictions_merged = list(predictions_merged)
    if predictions_merged != [0] and 0 in predictions_merged:
        predictions_merged.remove(0)
    print("This is in spatial merge", predictions_merged)
    return predictions_merged

def seat_merge_temporal(A_prediction, B_prediction, C_prediction):
    predictions_merged = set(A_prediction)|set(B_prediction)|set(C_prediction)
    sys.stdout.flush()
    predictions_merged = list(predictions_merged)
    if predictions_merged != [0] and 0 in predictions_merged:
        predictions_merged.remove(0)
    print("This is in temporal merge", predictions_merged)
    return predictions_merged

def seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, method="union"):
    #For now, we just ignore backs in merge.
    #preds = pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + pos7 + pos8
    preds = pos1 + pos2 + pos3 + pos4 + pos5 + pos6 
    predictions_merged = []
    if method == "union":
        for pred in preds:
            predictions_merged.append(pred)
        predictions_merged = set(predictions_merged)
    elif method == "vote":
        for pred in preds:
            predictions_merged.append(pred)
        seat_count = np.zeros(config.NUM_OF_SEATS_PEER_CAR)
        for i in range(1, config.NUM_OF_SEATS_PEER_CAR+1):
            seat_count[i-1] = predictions_merged.count(i)
        predictions_merged = set(np.where(seat_count>=config.VOTE_THRESHOLD)[0]+1)
    elif method == "front_and_back":
        predictions_merged = set(preds)
    else:
        print("Wrong method given. [union, vote]")
        sys.exit()
    if (0 in predictions_merged):
        predictions_merged.remove(0)
    predictions_merged = list(predictions_merged)
    print("Merge:", predictions_merged, method, pos7, pos8)
    return predictions_merged

class Thread_A(threading.Thread):
    def run(self):
        filelist = glob.glob("/home/user/left/*.jpg")
        for idx, filename in enumerate(filelist):
            A_image_data = camera.get_image_data(filename)
            A_pos = A_program.self_logic(A_image_data, CONFIDENCE_THRESHOLD)
            print(idx, A_pos)

class Thread_B(threading.Thread):
    def run(self):
        filelist = glob.glob("/home/user/right/*.jpg")
        for idx, filename in enumerate(filelist):
            B_image_data = camera.get_image_data(filename)
            B_pos = B_program.self_logic(B_image_data, CONFIDENCE_THRESHOLD)
            print(idx, B_pos)

if __name__ == "__main__":

    #Initialization:
    #A:
    print("Initializing front camera A...")
    root_dir = os.getcwd()
    A_program = A.A(root_dir)
    #B:
    print("Initializing front camera B...")
    B_program = B.B(root_dir)
    #创建线程
    #thread_A = Thread_A()
    #thread_B = Thread_B()
    #开启线程
    #thread_A.start()
    #thread_B.start()

    A_filelist = glob.glob("./left/*.jpg")
    B_filelist = glob.glob("./right/*.jpg")
    for idx, (A_filename, B_filename) in enumerate(zip(A_filelist[:], B_filelist[:])):
        #A side:
        plt.figure()
        A_image_data = camera.get_image_data(A_filename)
        A_pos, A_plt = A_program.self_logic(A_image_data, CONFIDENCE_THRESHOLD)
        #B side:
        plt.figure()
        B_image_data = camera.get_image_data(B_filename)
        B_pos, B_plt = B_program.self_logic(B_image_data, CONFIDENCE_THRESHOLD)
        #Both side:
        plt.figure()
        positions_peer_car = seat_merge_all(A_pos+B_pos) 
        seats = np.zeros(shape=(1,5))
        if len(positions_peer_car)>0:
            seats[0, np.array(positions_peer_car)-1] = 1
        sns.heatmap(seats, linewidths=0.1, vmin=0, vmax=1, cmap='Blues', square=True, linecolor='white', annot=True, xticklabels=['1','2','3','4','5'])
        plt.title("Number of person this car: %s"%seats.sum())
        plt.pause(0.01)

        input()
        plt.close()
        plt.close()
        plt.close()
        print(idx, A_pos, B_pos)
        sys.exit()
