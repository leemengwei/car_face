import config

import threading
import time,os,sys
import glob
import front_position_algorithm_A as A
from camera import camera
import seat_merge
from IPython import embed
import numpy as np

class Worker():
    def __init__(self, side, time_num=None):
        self.root_dir = os.getcwd()
        self.program = A.A(self.root_dir, side, time_num)    #With side annoted here.
    def do(self, image_data, CONFIDENCE_THRESHOLDS):
        positions, plt = self.program.self_logic(image_data, CONFIDENCE_THRESHOLDS)
        return positions

class thread_manager(threading.Thread):
    def __init__(self, worker, CONFIDENCE_THRESHOLDS):
        threading.Thread.__init__(self)
        self.worker = worker
        self.image_data = None
        self.positions = None
        self.CONFIDENCE_THRESHOLDS = CONFIDENCE_THRESHOLDS
    def set_values(self, image_data):
        self.image_data = image_data
    def run(self):
        if self.image_data is None:
            raise NotImplementedError
        self.positions = self.worker.do(self.image_data, self.CONFIDENCE_THRESHOLDS)

def python_get_images(image_idx, filelist):
    image_name1 = filelist[image_idx]
    image_data1 = camera.get_image_data(image_name1)
    image_idx += 1
    image_name2 = filelist[image_idx]
    image_data2 = camera.get_image_data(image_name2)
    image_idx += 1
    image_name3 = filelist[image_idx]
    image_data3 = camera.get_image_data(image_name3)
    image_idx += 1
    image_name4 = filelist[image_idx]
    image_data4 = camera.get_image_data(image_name4)
    image_idx += 1
    image_name5 = filelist[image_idx]
    image_data5 = camera.get_image_data(image_name5)
    image_idx += 1
    image_name6 = filelist[image_idx]
    image_data6 = camera.get_image_data(image_name6)
    image_idx += 1
    image_name7 = filelist[image_idx]
    image_data7 = camera.get_image_data(image_name7)
    image_idx += 1
    image_name8 = filelist[image_idx]
    image_data8 = camera.get_image_data(image_name8)
    image_idx += 1
    image_name9 = filelist[image_idx]
    image_data9 = camera.get_image_data(image_name9)
    image_idx += 1
    image_name10 = filelist[image_idx]
    image_data10 = camera.get_image_data(image_name10)
    image_idx += 1
    return image_data1, image_data2, image_data3, image_data4, image_data5, image_data6, image_data7, image_data8, image_data9, image_data10, image_idx

class workers_cluster():
    # 初始化新线程，加载模型X3
    def __init__(self):
        self.worker1 = Worker("left", 1)
        self.worker2 = Worker("left", 2)
        self.worker3 = Worker("left", 3)
        self.worker4 = Worker("right", 1)
        self.worker5 = Worker("right", 2)
        self.worker6 = Worker("right", 3)
        self.worker7 = Worker("backleft", 1)
        self.worker8 = Worker("backleft", 2)
        self.worker9 = Worker("backright", 1)
        self.worker10 = Worker("backright", 2)
        sys.stdout.flush()
        self.workers_list = [ \
                self.worker1, \
                self.worker2, \
                self.worker3, \
                self.worker4, \
                self.worker5, \
                self.worker6, \
                self.worker7, \
                self.worker8, \
                self.worker9, \
                self.worker10, \
                ]

def algorithm_detection_and_merge(workers, \
                     image_data1, image_data2, image_data3, image_data4, image_data5, image_data6, image_data7, image_data8, image_data9, image_data10):
    #INTERFACE WITH C
    #Will deprecate in next version
    if os.path.exists("./history_refs_right"):
        os.remove('./history_refs_right')
    if os.path.exists("./history_refs_left"):
        os.remove('./history_refs_left')
    workers_list = workers.workers_list
    sys.stdout.flush()
    if config.PARALLEL_MODE:
        print("Running on parallel mode...")
        start_time = time.time()
        pos = []
        #GPU 够，10个一起并行吧, 顺序间workers实例化
        #Lefts:
        _thread1 = thread_manager(workers_list[0], config.FRONT_CONFIDENCE_THRESHOLDS)
        _thread1.set_values(image_data1)
        _thread1.start()
        _thread2 = thread_manager(workers_list[1], config.FRONT_CONFIDENCE_THRESHOLDS)
        _thread2.set_values(image_data2)
        _thread2.start()
        _thread3 = thread_manager(workers_list[2], config.FRONT_CONFIDENCE_THRESHOLDS)
        _thread3.set_values(image_data3)
        _thread3.start()
        #Rights:
        _thread4 = thread_manager(workers_list[3], config.FRONT_CONFIDENCE_THRESHOLDS)
        _thread4.set_values(image_data4)
        _thread4.start()
        _thread5 = thread_manager(workers_list[4], config.FRONT_CONFIDENCE_THRESHOLDS)
        _thread5.set_values(image_data5)
        _thread5.start()
        _thread6 = thread_manager(workers_list[5], config.FRONT_CONFIDENCE_THRESHOLDS)
        _thread6.set_values(image_data6)
        _thread6.start()
        #Backs:
        _thread7 = thread_manager(workers_list[6], config.BACK_CONFIDENCE_THRESHOLDS)
        _thread7.set_values(image_data7)
        _thread7.start()
        _thread8 = thread_manager(workers_list[7], config.BACK_CONFIDENCE_THRESHOLDS)
        _thread8.set_values(image_data8)
        _thread8.start()
        _thread9 = thread_manager(workers_list[8], config.BACK_CONFIDENCE_THRESHOLDS)
        _thread9.set_values(image_data9)
        _thread9.start()
        _thread10 = thread_manager(workers_list[9], config.BACK_CONFIDENCE_THRESHOLDS)
        _thread10.set_values(image_data10)
        _thread10.start()

        #print("Fronts finished, now back...")
        #sys.exit()

        _thread1.join()
        _thread2.join()
        _thread3.join()
        _thread4.join()
        _thread5.join()
        _thread6.join()
        _thread7.join()
        _thread8.join()
        _thread9.join()
        _thread10.join()
        
        pos1 = _thread1.positions
        pos2 = _thread2.positions
        pos3 = _thread3.positions
        pos4 = _thread4.positions
        pos5 = _thread5.positions
        pos6 = _thread6.positions
        pos7 = _thread7.positions
        pos8 = _thread8.positions
        pos9 = _thread9.positions
        pos10 = _thread10.positions
        #融合:
        predictions_merged = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, method="union") 
        predictions_merged = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, method="vote") 
        predictions_merged = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, method="front_and_back")
    else:
        print("Running in serial mode with bug...")
        start_time = time.time()
        i = 1
        for worker in workers_list:
            pos1 = worker.do(image_data1, config.FRONT_CONFIDENCE_THRESHOLDS)
            i += 0.5
        pos2 = pos3 = pos4 = pos5 = pos6 = pos7 = pos8 = pos9 = pos10 = pos1
        predictions_merged = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, method="union") 
        predictions_merged = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, method="vote") 
        predictions_merged = seat_merge.seat_merge_all(pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, pos9, pos10, method="front_and_back")
    #目前分别返回，后侧加在最后，并以“人数个数”显示（而不是位置）
    num7 = len(np.where(np.array(pos7)!=0)[0])
    num8 = len(np.where(np.array(pos8)!=0)[0])
    num_backleft = max(num7, num8)
    num9 = len(np.where(np.array(pos9)!=0)[0])
    num10 = len(np.where(np.array(pos10)!=0)[0])
    num_backright = max(num9, num10)
    predictions_merged = predictions_merged + [num_backleft] + [num_backright]
    print("predictions_merged", predictions_merged)


    print("-"*50, "TIME", time.time()-start_time, '-'*50)
    sys.stdout.flush()
    return predictions_merged

if __name__ == "__main__":
    #初始化：
    workers = workers_cluster() 
    #input("Enter to start")
    filelist = glob.glob(config.CAR_TO_CAR_DIR+"/*/*.png")
    image_idx = 0
    while True:
        try:
            #数据：
            image_data1, image_data2, image_data3, image_data4, image_data5, image_data6, image_data7, image_data8, image_data9, image_data10, image_idx = python_get_images(image_idx, filelist)
        except IndexError:
            print ("图片完， 退出主线程")
            break
        final_result = algorithm_detection_and_merge(workers, image_data1, image_data2, image_data3, image_data4, image_data5, image_data6, image_data7, image_data8, image_data9, image_data10) 
            #input()   

