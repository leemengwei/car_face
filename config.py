#STuffs: imports for Pyinstaller distribution
#Before all fix pyinstaller:
#import multiprocessing
#multiprocessing.freeze_support()
#import pywt._extensions._cwt
#import sklearn.utils._cython_blas
#import skimage.io
#import skimage.io._plugins.matplotlib_plugin
#skimage.io.use_plugin('pil', 'imread')   #这些都是编译exe所需，这里须显示调用pil的imread，而不是matplotlib的，因为我训练的时候dataloader里默认用的是pil的imread，和matplotlib的imread读取的数据分布不一样。

_LIGHT_THRESHOLD = 20

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
    CONFIDENCE_THRESHOLD = 0.595
    if night_cast():
        CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD/2   #晚上的置信度是白天的一半
    return CONFIDENCE_THRESHOLD

CONFIDENCE_THRESHOLD = get_confidence()

#Just a workaround:
six_or_four = 4

################Options for object detection:
if six_or_four == 6:
    CLASSES = ['angle', 'angle_r', 'top', 'top_r', 'head']
elif six_or_four == 4:
    CLASSES = ['angle', 'top', 'head']
else:
    raise KeyboardInterrupt

################Options for A and B:
VISUALIZATION        = False
VISUALIZATION        = True
UNVEIL               = False
UNVEIL               = True
OBJECT_DETECTION_MODEL = "object_detection_logs_data_both_side_finetunes/csv_retinanet_full_data_465.pt"    #微调后
SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_finetunes/model_best.pt"   #微调后
if six_or_four == 6:
    MMD_CONFIG = "mmdetection/configs/car_face/cascade_rcnn_hrnetv2p_w32_20e.py"
    MMD_WEIGHTS = "mmdetection/trained_models/car_face/hrnet_latest.pth"
else:
    MMD_CONFIG = "mmdetection/configs/car_face/cascade_rcnn_hrnetv2p_w32_20e_4.py"
    MMD_WEIGHTS = "object_detection_logs_data_both_side_finetunes/hrnet_night_and_day.pth"
    MMD_CONFIG_NIGHT = "mmdetection/configs/car_face/cascade_rcnn_hrnetv2p_w32_20e_4.py"
    MMD_WEIGHTS_NIGHT = "object_detection_logs_data_both_side_finetunes/hrnet_night_and_day.pth"

#################Options for threads_start:
PARALLEL_MODE = False    #单线程的threads_starts会有bug！只会调用左侧的 测试的话 请注意！  单 car_to_car_merge应该不受影响
PARALLEL_MODE = True
if PARALLEL_MODE:
    VISUALIZATION = False


##################Options for Seat merge:
NUM_OF_SEATS_PEER_CAR = 5
MERGE_METHOD = "vote"
VOTE_THRESHOLD = 2  #where >= count
#CAR_TO_CAR_DIR = "../shanghai_data/2019-06-17-statics/"
CAR_TO_CAR_DIR = "/home/user/list/"




######################MMD:
