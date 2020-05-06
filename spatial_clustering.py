from IPython import embed
import argparse
import matplotlib
import numpy as np
import copy
from time import sleep
import time
import os
import sys
from tqdm import tqdm
import plot_utils
import spatial_model
import spatial_get_data as get_data
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt
from itertools import cycle

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = "Car Space Net...")
    parser.add_argument('-V', '--visualization', action="store_true", default=False)
    parser.add_argument('-R', '--restart', action="store_true", default=False)
    parser.add_argument('-SM', '--save_model', action="store_true", default=False)
    parser.add_argument('-RM', '--restart_model', type=str, default = None)
    parser.add_argument('-C', '--cuda_number', type=int, default = 0)
    parser.add_argument('-HD', '--hidden_depth', type=int, default = 3)
    parser.add_argument('-HW', '--hidden_width', type=int, default = 100)
    parser.add_argument('-EX', '--expander', type=int, default=1)
    parser.add_argument('-LR', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-E', '--epochs', type=int, default=24)
    parser.add_argument('-TR', '--test_ratio', type=float, default = 0.2)
    parser.add_argument('-BS', '--batch_size', type=int, default=100)
    parser.add_argument('-DP', '--data_path', type=str, required=True)
    args = parser.parse_args()


    inputs, paths = get_data.get_ref_and_heads(args.data_path, args)
    inputs, paths = get_data.mannual_feature(inputs, paths, args, plot=False)

    #do clustering:
    X = inputs[['x_ratio', 'y_ratio']].values
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=None)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    
    print("number of estimated clusters : %d" % n_clusters_)
    
    #plt.figure(1)
    #plt.clf()
    #colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    #for k, col in zip(range(n_clusters_), colors):
    #    my_members = labels == k
    #    cluster_center = cluster_centers[k]
    #    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    #    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    #plt.title('Estimated number of clusters: %d' % n_clusters_)
    #plt.show()
    
    
    
