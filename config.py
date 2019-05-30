#STuffs: imports for Pyinstaller distribution
#Before all fix pyinstaller:
#import multiprocessing
#multiprocessing.freeze_support()
#import pywt._extensions._cwt
#import sklearn.utils._cython_blas
#import skimage.io
#import skimage.io._plugins.matplotlib_plugin
#skimage.io.use_plugin('pil', 'imread')   #这些都是编译exe所需，这里须显示调用pil的imread，而不是matplotlib的，因为我训练的时候dataloader里默认用的是pil的imread，和matplotlib的imread读取的数据分布不一样。


################Options for object detection:
CLASSES = ['angle', 'angle_r', 'top', 'top_r', 'head']
CONFIDENCE_THRESHOLD = 0.8


################Options for A and B:
VISUALIZATION        = False
VISUALIZATION        = True
UNVEIL               = False
UNVEIL               = True
#OBJECT_DETECTION_MODEL = "object_detection_logs_data_both_side_all_for_deploy/csv_retinanet_best.pt"    #微调前
OBJECT_DETECTION_MODEL = "object_detection_logs_data_both_side_finetunes/morning_update/csv_retinanet_finetune_51.pt"    #微调后
#SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_all_for_deploy/model_best.pt"   #微调前
SPATIAL_IN_SEAT_MODEL = "spatial_model_both_side_finetunes/morning_update/model_best.pt"   #微调后


#################Options for threads_start:
PARALLEL_MODE = False    #单线程有bug！只会调用左侧的 测试的话 请注意！
PARALLEL_MODE = True
if PARALLEL_MODE:
    VISUALIZATION = False


##################Options for Seat merge:
NUM_OF_SEATS_PEER_CAR = 5
MERGE_METHOD = "vote"
VOTE_THRESHOLD = 2  #where >= count
CAR_TO_CAR_DIR = "/home/user/list"
