#STuffs: imports for Pyinstaller distribution
#Before all fix pyinstaller:
#import multiprocessing
#multiprocessing.freeze_support()
#import pywt._extensions._cwt
#import sklearn.utils._cython_blas
#import skimage.io
#import skimage.io._plugins.matplotlib_plugin
#skimage.io.use_plugin('pil', 'imread')   #这些都是编译exe所需，这里须显示调用pil的imread，而不是matplotlib的，因为我训练的时候dataloader里默认用的是pil的imread，和matplotlib的imread读取的数据分布不一样。

def night_cast():
    try:
        with open("./daylight", 'r') as f:
            light = int(f.readline())
    except:
        light = 10
    if light>_LIGHT_THRESHOLD:
        NIGHT_CAST = True
    else:
        NIGHT_CAST = False
    return NIGHT_CAST

def get_confidence(): 
    #CONFIDENCE_THRESHOLD = 0.63   #retrain with conf 0.53 to get 2.87 with old data.
    CONFIDENCE_THRESHOLD = 0.5
    if night_cast():
        #CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD/2   #晚上的置信度是白天的一半  #will depracate in future
        pass
    return CONFIDENCE_THRESHOLD

#Just a workaround:
################Options for object detection:
CLASSES_4 = ['angle', 'top', 'head']    #加入晚上数据的模型已经不区分左右了

################Options for A and B:
UNVEIL               = False
UNVEIL               = True

_LIGHT_THRESHOLD = 20  #光线曝光时间阈值，实际值大于阈值则说明是晚上
CONFIDENCE_THRESHOLD = get_confidence()   
BACK_CONFIDENCE_THRESHOLD = 0.6
#定位模型
#SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_danger_full_5_pos/model_best_old12345.pt"   
#SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_danger_full_5_adjust/model_best.pt"   
SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_danger_full_5_but_for_1234/model_best_for_1234.pt"   
#检测模型
MMD_CONFIG = "mmdetection/configs/car_face/cascade_rcnn_hrnetv2p_w32_20e_4_more_neg.py"
MMD_WEIGHTS = "object_detection_logs_data_both_side_finetunes/hrnet_epoch_7_head944_conf049.pth"

BACK_HEAD_TOO_SMALL = 50 #40
HEAD_TOO_SMALL = 60  #45
TOP_TOO_SMALL = 52
ANGLE_TOO_SMALL = 53
WINDOW_WIDTH = 650*0.85
WINDOW_HEIGHT = 200*0.85
#################Options for threads_start:
PARALLEL_MODE = False    #单线程的threads_starts会有bug！只会调用左侧的 测试的话 请注意！  单 car_to_car_merge应该不受影响
#PARALLEL_MODE = True
if PARALLEL_MODE:
    VISUALIZATION = False
else:
    VISUALIZATION = True

##################Options for Seat merge:
NUM_OF_SEATS_PEER_CAR = 5
MERGE_METHOD = "vote"
VOTE_THRESHOLD = 2  #where >= count
#CAR_TO_CAR_DIR = "/home/user/experiments/"
CAR_TO_CAR_DIR = "./units_experiments/"

IGNORE_5 = True
IGNORE_5 = False

######################MMD:

