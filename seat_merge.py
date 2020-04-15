#-*-coding:utf-8-*-
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

def seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, method="union"):
    #For now, we just ignore backs in merge.
    preds_front = pos1 + pos2 + pos3 + pos4 + pos5 + pos6 
    preds_back = pos7 + pos8 + pos9 + pos10 
    predictions_merged = []
    if method == "union":
        predictions_merged = set(preds_front+preds_back)
    elif method == "vote":
        seat_count = np.zeros(config.NUM_OF_SEATS_PEER_CAR)
        for i in range(1, config.NUM_OF_SEATS_PEER_CAR+1):
            seat_count[i-1] = preds_front.count(i)
        predictions_merged = set(np.where(seat_count>=config.VOTE_THRESHOLD)[0]+1)
    elif method == "front_and_back":
        #Front is still vote.
        seat_count = np.zeros(config.NUM_OF_SEATS_PEER_CAR)
        for i in range(1, config.NUM_OF_SEATS_PEER_CAR+1):
            seat_count[i-1] = preds_front.count(i)
        front = set(np.where(seat_count>=config.VOTE_THRESHOLD)[0]+1)
        #Add up back 
        back = set(pos7 + pos8 + pos9 +pos10) - {1} - {2}
        predictions_merged = front.union(back)
        predictions_merged = predictions_merged - {0}
        #最终汇总时没2号，and 3,4,5 all in back, 则把后排修改为34两人
        if 2 not in predictions_merged and 3 in predictions_merged and 4 in predictions_merged and 5 in predictions_merged:
            predictions_merged = predictions_merged-{5}
    else:
        print("Wrong method given. [union, vote, front_and_back]")
        sys.exit()
    if (0 in predictions_merged):
        predictions_merged.remove(0)
    predictions_merged = list(predictions_merged)
    print("Merge:", predictions_merged, method, "back pos given:", (pos7+pos8), (pos9+pos10))
    return predictions_merged

