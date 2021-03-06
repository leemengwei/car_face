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

def get_confidences(): 
    #CONFIDENCE_THRESHOLD = 0.63   #retrain with conf 0.53 to get 2.87 with old data.
    CONFIDENCE_THRESHOLDS = [0.5, 0.5, 0.5]
    if night_cast():
        #CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD/2   #晚上的置信度是白天的一半  #will depracate in future
        pass
    return CONFIDENCE_THRESHOLDS

#Just a workaround:
################Options for object detection:
CLASSES_4 = ['angle', 'top', 'head']    #加入晚上数据的模型已经不区分左右了

################Options for A and B:
UNVEIL               = False
UNVEIL               = True

_LIGHT_THRESHOLD = 20  #光线曝光时间阈值，实际值大于阈值则说明是晚上
#CLASSES: angle top head
FRONT_CONFIDENCE_THRESHOLDS = [0.3, 0.3, 0.86]
#FRONT_CONFIDENCE_THRESHOLDS = [0.95, 0.95, 0.95]
BACK_CONFIDENCE_THRESHOLDS = [1.0, 0.86, 0.86]
#BACK_CONFIDENCE_THRESHOLDS = [0.95, 0.95, 0.95]
#定位模型
#SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_danger_full_5_pos/model_best_old12345.pt"   
#SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_danger_full_5_adjust/model_best.pt"   
SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_danger_full_5_but_for_1234/model_best_for_1234.pt_bak"   
#检测模型
#FRONT:
MMD_FRONT_CONFIG = "./mmdetection/configs/car_face/cascade_rcnn_hrnetv2p_w32_20e_4.py"
#MMD_FRONT_WEIGHTS = "./object_detection_logs_data_both_side_finetunes/epoch_18_front_FP3684.pth"
MMD_FRONT_WEIGHTS = "./object_detection_logs_data_both_side_finetunes/epoch_14_front_FP3870.pth"
#BACK:
MMD_BACK_CONFIG = "./mmdetection/configs/car_face/cascade_rcnn_hrnetv2p_w32_20e_4_back.py"
#MMD_BACK_WEIGHTS = "./object_detection_logs_data_both_side_finetunes/epoch_14_back_FP2978.pth"
MMD_BACK_WEIGHTS = "./object_detection_logs_data_both_side_finetunes/epoch_14_back_FP2769.pth"

BACK_HEAD_TOO_SMALL = 1500 #3200 #~=57 #50 #40
BACK_HEAD_TOO_BIG = 99999999 #40
HEAD_TOO_SMALL = 1500 #5000 #~=70  #60  #45
TOP_TOO_SMALL = 35
ANGLE_TOO_SMALL = 35
WINDOW_WIDTH = 650*0.85
WINDOW_HEIGHT = 200*0.85
LEFT_FILL = 3 #for left multi1
RIGHT_FILL = 4  #for right multi2
SLIT = 20   #to judge if too far

#BACKLEFTGUESS = [4,3,5,5,5]
BACKLEFTGUESS = [4,5,5,5,5]
#BACKRIGHTGUESS = [3,4,5,5,5]
BACKRIGHTGUESS = [3,5,5,5,5]

#################Options for threads_start:
PARALLEL_MODE = False    #单线程的threads_starts会有bug！只会调用左侧的 测试的话 请注意！  单 car_to_car_merge应该不受影响
#PARALLEL_MODE = True
if PARALLEL_MODE:
    VISUALIZATION = False
else:
    VISUALIZATION = True
VISUALIZATION = False   #set False when only want to check txt or debug

##################Options for Seat merge:
NUM_OF_SEATS_PEER_CAR = 5
MERGE_METHOD = "vote"
VOTE_THRESHOLD = 1  #where >= count
CAR_TO_CAR_DIR = "/home/user/345/"
#CAR_TO_CAR_DIR = "./units_experiments/"

IGNORE_5 = True
IGNORE_5 = False

######################MMD:

