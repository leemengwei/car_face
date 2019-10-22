'''
from front_position_algorithm.A import A
class B():
    def __init__(self, tmp):
        pass
    pass
'''
import argparse
from IPython import embed
import matplotlib
import matplotlib.pyplot as plt
import sys,os 
import config
sys.path.append("../")
from camera import camera
from torch import load
import collections
import warnings
warnings.filterwarnings("ignore")
import glob
import object_detection_model as model
import spatial_model
import front_position_algorithm_A as A
A = A.A

class B(A):
    def __init__(self, root_dir):
        super(B, self).__init__(root_dir)
        #super(A, self).__init__()
        self.root_dir = root_dir
        if type(self.root_dir) is bytes:
            self.root_dir = root_dir.decode("utf-8")
        self.netname = "B"
        self.side = "right"
        print("Program B-Initialized.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Side B Net...")
    parser.add_argument('-V', '--visualization', action="store_true", default=False)
    args = parser.parse_args()

    #Initializing:
    print("Initializing front camera B...")
    root_dir = "/".join(os.getcwd().split('/')[:-1])
    B_program = B(root_dir)
    print("Real_time running...")
    filelist = glob.glob("/home/user/storage/data/car_head/couple-data/right/*.jpg")    #TODO : Feed B.py right data to get correct results
    for idx, filename in enumerate(filelist):
        image_data = camera.get_image_data(filename)
        #global_signal = B_program.get_global_signal()
        B_program.self_logic(image_data)

