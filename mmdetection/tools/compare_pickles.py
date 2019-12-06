import pickle
import numpy as np
from IPython import embed
from config import *
import os, sys

score_thr = CONFIDENCE_THRESHOLD

def solve_with_one_file(filename):
    f = open(filename,"rb")
    results = pickle.load(f)
    angles_list, tops_list, heads_list = np.array([]), np.array([]), np.array([])
    for result in results:
        num_of_angles = np.where(result[0][:, -1]>score_thr)[0].shape[0]
        num_of_tops = np.where(result[1][:, -1]>score_thr)[0].shape[0]
        num_of_heads = np.where(result[2][:, -1]>score_thr)[0].shape[0]
        angles_list = np.append(angles_list, num_of_angles)
        tops_list = np.append(tops_list, num_of_tops)
        heads_list = np.append(heads_list, num_of_heads)
    return angles_list, tops_list, heads_list

def compare_and_report(which_part):
    parts_accross_pkls = list(which_part.values())
    num_of_files = len(parts_accross_pkls)
    num_of_images_peer_pkl = len(parts_accross_pkls[0])
    bool_status = np.tile(np.array([True]), num_of_images_peer_pkl)
    for i in range(num_of_files-1):
        tmp_bool_status = (parts_accross_pkls[i]==parts_accross_pkls[i+1])
        bool_status = bool_status&tmp_bool_status
    for idx, statu in enumerate(bool_status):
        if statu == False:
            #embed()
            print("Plz visually Check:", idx)


if __name__ == "__main__":
    filenames = [sys.argv[1], sys.argv[2]]
    assert len(filenames)>=2, "At least two files!"
    anlges, tops, heads = {}, {}, {}
    for idx,filename in enumerate(filenames):
        anlges[filename], tops[filename], heads[filename] = solve_with_one_file(filename)
    #对比：
    compare_and_report(heads)
