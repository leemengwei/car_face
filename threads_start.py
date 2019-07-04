import config

import threading
import time,os,sys
import glob
import front_position_algorithm_A as A
import side_position_algorithm_B as B
from camera import camera
import seat_merge
from IPython import embed
class Worker():
    def __init__(self, name, side):
        self.name = name
        self.root_dir = os.getcwd()
        if side is "left":
            self.program = A.A(self.root_dir)
        else:  #side is "right"
            self.program = B.B(self.root_dir)
    def do(self, image_data, CONFIDENCE_THRESHOLD):
        positions, plt = self.program.self_logic(image_data, CONFIDENCE_THRESHOLD)
        return positions
class thread_manager(threading.Thread):
    def __init__(self, worker, CONFIDENCE_THRESHOLD):
        threading.Thread.__init__(self)
        self.worker = worker
        self.image_data = None
        self.positions = None
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
    def set_values(self, image_data):
        self.image_data = image_data
    def run(self):
        if self.image_data is None:
            raise NotImplementedError
        print("DO", self.CONFIDENCE_THRESHOLD)
        self.positions = self.worker.do(self.image_data, self.CONFIDENCE_THRESHOLD)

def get_images(image_idx, filelist):
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
    return image_data1, image_data2, image_data3, image_data4, image_data5, image_data6, image_idx

class workers_cluster():
    # 初始化新线程，加载模型X3
    def __init__(self):
        self.worker1 = Worker("Left1", "left")
#        self.worker2 = Worker("Left2", "left")
#        self.worker3 = Worker("Left3", "left")
        self.worker4 = Worker("Right1", "right")
#        self.worker5 = Worker("Right2", "right")
#        self.worker6 = Worker("Rihgt3", "right")
        sys.stdout.flush()
        self.workers_list = [ \
                self.worker1, \
#                self.worker2, \
#                self.worker3, \
                self.worker4, \
#                self.worker5, \
#                self.worker6, \
                ]

def algorithm_detection_and_merge(workers, \
                     image_data1, image_data2, image_data3, image_data4, image_data5, image_data6):
    #embed()
    #image_data1 = image_data2 = image_data3 = image_data4 = image_data5 = image_data6 = camera.get_image_data("/home/user/Data1/2019-05-12 16-53-48/2019-05-12 16-53-49-520.png")
    workers_list = workers.workers_list
    sys.stdout.flush()
    CONFIDENCE_THRESHOLD = config.get_confidence() 
    if config.PARALLEL_MODE:
        print("Running on parallel mode...")
        start_time = time.time()
        pos = []
        #GPU 显存不够，2个2个并行
        _thread1 = thread_manager(workers_list[0], CONFIDENCE_THRESHOLD)
        _thread1.set_values(image_data1)
        _thread1.start()
        _thread1.join()
        pos1 = _thread1.positions
        _thread4 = thread_manager(workers_list[1], CONFIDENCE_THRESHOLD)
        _thread4.set_values(image_data4)
        _thread4.start()
        _thread4.join()
        pos4 = _thread4.positions
        #第二组
        _thread2 = thread_manager(workers_list[0], CONFIDENCE_THRESHOLD)
        _thread2.set_values(image_data2)
        _thread2.start()
        _thread2.join()
        pos2 = _thread2.positions
        _thread5 = thread_manager(workers_list[1], CONFIDENCE_THRESHOLD)
        _thread5.set_values(image_data5)
        _thread5.start()
        _thread5.join()
        pos5 = _thread5.positions
        #第二组
        _thread3 = thread_manager(workers_list[0], CONFIDENCE_THRESHOLD)
        _thread3.set_values(image_data3)
        _thread3.start()
        _thread3.join()
        pos3 = _thread3.positions
        _thread6 = thread_manager(workers_list[1], CONFIDENCE_THRESHOLD)
        _thread6.set_values(image_data6)
        _thread6.start()
        _thread6.join()
        pos6 = _thread6.positions
        #融合:
        pos = pos1 + pos2 + pos3 + pos4 + pos5 + pos6
        predictions_merged = seat_merge.seat_merge_all(pos, method=config.MERGE_METHOD) 
    else:
        print("Running on serial mode...")
        start_time = time.time()
        pos = []
        for worker in workers_list:
            pos += workers.worker1.do(image_data5, CONFIDENCE_THRESHOLD)
        predictions_merged = seat_merge.seat_merge_all(pos, method=config.MERGE_METHOD)
    print("TIME", time.time()-start_time, predictions_merged)
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
            image_data1, image_data2, image_data3, image_data4, image_data5, image_data6, image_idx = get_images(image_idx, filelist)
        except IndexError:
            print ("图片完， 退出主线程")
            break
        final_result = algorithm_detection_and_merge(workers, \
                                    image_data1, image_data2, image_data3, image_data4, image_data5, image_data6) 
            #input()   

