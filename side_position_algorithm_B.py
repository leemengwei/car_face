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
        #super().__init__(self.camera_number)
        #Init model:
        self.objs_net_checkpoint_dict = load('%s/%s'%(self.root_dir, config.OBJECT_DETECTION_MODEL))
        self.spatial_net_checkpoint_dict = load('%s/%s'%(self.root_dir, config.SPATIAL_IN_SEAT_MODEL))
        #load model structure
        self.net_to_detect_objs = model.resnet152(num_classes = 5, pretrained=True, root_dir=self.root_dir)
        #self.net_to_detect_objs = model.resnet50(num_classes = 5, pretrained=True, root_dir=self.root_dir)
        self.net_spatial = spatial_model.NeuralNet(input_size=12, hidden_size= 20, hidden_depth=5, output_size=5)
        #load model params
        self.net_to_detect_objs.load_state_dict(self.objs_net_checkpoint_dict['model_state_dict'])
        self.net_spatial.load_state_dict(self.spatial_net_checkpoint_dict['model_state_dict'])
        #switch mode
        self.net_to_detect_objs.cuda().eval()
        self.net_spatial.cuda().eval()
        #Init internal parameters:
        self.seq_ground_signal = collections.deque(maxlen=200)
        self.seq_threshold_signal = collections.deque(maxlen=200)
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

