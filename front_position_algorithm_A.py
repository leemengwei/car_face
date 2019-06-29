'''
class A():
    def __init__(self, tmp):
        pass
    pass

'''
import copy
import numpy as np
#import argparse
from IPython import embed
import sys,os
import config
import matplotlib
import matplotlib.pyplot as plt
sys.path.append("../")
from camera import camera
from torch import load, no_grad, FloatTensor
#import cv2
import object_detection_visualize as visualize
import object_detection_model as model
import spatial_model
import spatial_in_seat_train_and_test
import time
import skimage.io
import glob
plt.ion()
import collections
import warnings
warnings.filterwarnings("ignore")
from config import *
import seaborn as sns

import mmcv
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
import mmdetection.tools.test_fix as test_fix
from mmdet.apis import init_detector


class A(camera):
    def __init__(self, root_dir):
        super(A, self).__init__()
        self.root_dir = root_dir
        if type(self.root_dir) is bytes:
            self.root_dir = self.root_dir.decode("utf-8")
        self.netname = "A"
        self.side = "left"
        #super().__init__(self.camera_number)
        #Init model:
        self.angles_net_checkpoint_dict = load('%s/%s'%(self.root_dir, config.OBJECT_DETECTION_MODEL))
        self.spatial_net_checkpoint_dict = load('%s/%s'%(self.root_dir, config.SPATIAL_IN_SEAT_MODEL))
        #load model structure
        self.net_to_detect_angles = model.resnet152(num_classes = len(config.CLASSES), pretrained=True, root_dir=self.root_dir)
        self.net_spatial = spatial_model.NeuralNet(input_size=12, hidden_size= 20, hidden_depth=5, output_size=config.NUM_OF_SEATS_PEER_CAR)
        #load model params
        #use mmd now:
        self.net_to_detect_angles.load_state_dict(self.angles_net_checkpoint_dict['model_state_dict'])
        self.net_to_detect_objs= self.get_mmd_model_and_template()
        self.net_spatial.load_state_dict(self.spatial_net_checkpoint_dict['model_state_dict'])
        #switch mode
        self.net_to_detect_angles.cuda().eval()
        self.net_to_detect_objs.cuda().eval()
        self.net_spatial.cuda().eval()
        #Init internal parameters:
        self.seq_ground_signal = collections.deque(maxlen=200)
        self.seq_threshold_signal = collections.deque(maxlen=200)
        print("Program %s-Initialized."%self.netname)
    def get_mmd_model_and_template(self):
        cfg = mmcv.Config.fromfile(MMD_CONFIG)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        model = init_detector(MMD_CONFIG, MMD_WEIGHTS)
        return model

    def get_objs_position(self, angle_cam_frame, net_cam_frame):
        with no_grad():
            #Compensation
            #angles_x1s, angles_y1s, angles_x2s, angles_y2s, angles_scores, angles_indexes, angles_elapsed_time = visualize.frame_detection(self.net_to_detect_angles, angle_cam_frame, confidence=0.8)
            ##Don't use this head
            #angles_x1s = angles_x1s[np.where(angles_indexes!=4)]
            #angles_y1s = angles_y1s[np.where(angles_indexes!=4)]
            #angles_x2s = angles_x2s[np.where(angles_indexes!=4)]
            #angles_y2s = angles_y2s[np.where(angles_indexes!=4)]
            #angles_scores = angles_scores[np.where(angles_indexes!=4)]
            #angles_indexes = angles_indexes[np.where(angles_indexes!=4)]
            objs_x1s, objs_y1s, objs_x2s, objs_y2s, objs_scores, objs_indexes, objs_elapsed_time = test_fix.single_gpu_frame_detection(self.net_to_detect_objs, net_cam_frame, show=False)
            #But use its refs
            try:
                objs_x1s = np.append(objs_x1s, angles_x1s)
                objs_y1s = np.append(objs_y1s, angles_y1s)
                objs_x2s = np.append(objs_x2s, angles_x2s)
                objs_y2s = np.append(objs_y2s, angles_y2s)
                objs_scores = np.append(objs_scores, angles_scores)
                objs_indexes = np.append(objs_indexes, angles_indexes).astype(int)
            except:
                print("COMPENSATING ERROR")
                pass
            print("This:", "index",objs_indexes, "score",objs_scores)
            classes = config.CLASSES
            objs_names = np.array([classes[i] for i in objs_indexes])
            if len(objs_names)>0:
                heads_x1s = objs_x1s[np.where(objs_names=='head')]
                heads_y1s = objs_y1s[np.where(objs_names=='head')]
                heads_x2s = objs_x2s[np.where(objs_names=='head')]
                heads_y2s = objs_y2s[np.where(objs_names=='head')]
                heads_scores = objs_scores[np.where(objs_names=='head')]
                heads_names = objs_names[np.where(objs_names=='head')]
                refs_x1s = objs_x1s[np.where(objs_names!='head')]
                refs_y1s = objs_y1s[np.where(objs_names!='head')]
                refs_x2s = objs_x2s[np.where(objs_names!='head')]
                refs_y2s = objs_y2s[np.where(objs_names!='head')]
                refs_scores = objs_scores[np.where(objs_names!='head')]
                refs_names = objs_names[np.where(objs_names!='head')]
            else:
                heads_x1s = np.array([])
                heads_y1s = np.array([])
                heads_x2s = np.array([])
                heads_y2s = np.array([])
                heads_scores = np.array([])
                heads_names = np.array([])
                refs_x1s = np.array([])
                refs_y1s = np.array([])
                refs_x2s = np.array([])
                refs_y2s = np.array([])
                refs_scores = np.array([])
                refs_names = np.array([]) 
        return refs_x1s, refs_y1s, \
               refs_x2s, refs_y2s, \
               refs_scores, refs_names, \
               heads_x1s, heads_y1s, \
               heads_x2s, heads_y2s, \
               heads_scores, heads_names
    def check_refs_outputs(self, refs_x1s, refs_y1s, refs_x2s, refs_y2s, refs_scores, refs_label_names):
        frame_status = "ok"
        #这里考虑单侧的相机，无论是否看到了另一侧的angle或top，都不管，只处理本侧的angle和top
        #统一到单侧
        angle_name = "angle"
        top_name = "top"
        angles_x1s = refs_x1s[np.where(refs_label_names==angle_name)]
        angles_y1s = refs_y1s[np.where(refs_label_names==angle_name)]
        angles_x2s = refs_x2s[np.where(refs_label_names==angle_name)] 
        angles_y2s = refs_y2s[np.where(refs_label_names==angle_name)]
        angles_scores = refs_scores[np.where(refs_label_names==angle_name)]
        tops_x1s = refs_x1s[np.where(refs_label_names==top_name)] 
        tops_y1s = refs_y1s[np.where(refs_label_names==top_name)]
        tops_x2s = refs_x2s[np.where(refs_label_names==top_name)]
        tops_y2s = refs_y2s[np.where(refs_label_names==top_name)]
        tops_scores = refs_scores[np.where(refs_label_names==top_name)]
        angle_score = np.array([])
        top_score = np.array([])
        #标识物检测angle不能超过一个
        if len(angles_scores)>1:
            which_more_sure = np.argmax(angles_scores)
            angle_x1 = np.array(angles_x1s[which_more_sure]).reshape(-1,1)
            angle_y1 = np.array(angles_y1s[which_more_sure]).reshape(-1,1)
            angle_x2 = np.array(angles_x2s[which_more_sure]).reshape(-1,1)
            angle_y2 = np.array(angles_y2s[which_more_sure]).reshape(-1,1)
            angle_score = np.array(angles_scores[which_more_sure]).reshape(-1,1)
        else:
            angle_x1 = np.array(angles_x1s).reshape(-1,1)
            angle_y1 = np.array(angles_y1s).reshape(-1,1)
            angle_x2 = np.array(angles_x2s).reshape(-1,1)
            angle_y2 = np.array(angles_y2s).reshape(-1,1)
            angle_score = np.array(angles_scores).reshape(-1,1)
        #标识物检测top不能超过一个
        if len(tops_scores)>1:
            which_more_sure = np.argmax(tops_scores)
            top_x1 = np.array(tops_x1s[which_more_sure]).reshape(-1,1)
            top_y1 = np.array(tops_y1s[which_more_sure]).reshape(-1,1)
            top_x2 = np.array(tops_x2s[which_more_sure]).reshape(-1,1)
            top_y2 = np.array(tops_y2s[which_more_sure]).reshape(-1,1)
            top_score = np.array(tops_scores[which_more_sure]).reshape(-1,1)
        else:
            top_x1 = np.array(tops_x1s).reshape(-1,1)
            top_y1 = np.array(tops_y1s).reshape(-1,1)
            top_x2 = np.array(tops_x2s).reshape(-1,1)
            top_y2 = np.array(tops_y2s).reshape(-1,1)
            top_score = np.array(tops_scores).reshape(-1,1)
        #强制就绪两个标识物,(强制就绪意在解决明明有但没检测到的情况，包括车超出画面的情况,也会被强制就绪)
        if self.side is "left":
            if (len(angle_score)==1 and len(top_score)==0):
                top_x1 = angle_x1+685.0948918547668
                top_y1 = angle_y1-125.42961902952528
                top_x2 = angle_x2+660.4268173950429
                top_y2 = angle_y2-159.48261207951592
                top_score = np.array([0]).reshape(-1,1)
            elif (len(angle_score)==0 and len(top_score)==1):
                angle_x1 = top_x1-685.0948918547668
                angle_y1 = top_y1+125.42961902952528
                angle_x2 = top_x2-660.4268173950429
                angle_y2 = top_y2+159.48261207951592
                angle_score = np.array([0]).reshape(-1,1)
            elif (len(angle_score)==0 and len(top_score)==0):   #对于两个标识物一个都没有的情况，则状态变为无效帧
                frame_status = "NoRefs"
            else:   #或者都刚好有一个的情况,
                #But twisted refs
                if ((abs(top_x1+top_x2)/2 <= abs(angle_x1+angle_x2)/2) or (abs(top_y1+top_y2)/2 >= abs(angle_y1+angle_y2)/2)):
                    frame_status = "LeftSideMixedRefs"
                #Or all okay
                else:
                    pass    #或者都刚好有一个的情况，不处理。
        else: #self.side is "right":
            if (len(angle_score)==1 and len(top_score)==0):
                top_x1 = angle_x1-579.6467673760624
                top_y1 = angle_y1-98.41134448029959
                top_x2 = angle_x2-442.5618252549455
                top_y2 = angle_y2-124.53897522533538
                top_score = np.array([0]).reshape(-1,1)
            elif (len(angle_score)==0 and len(top_score)==1):
                angle_x1 = top_x1+579.6467673760624
                angle_y1 = top_y1+98.41134448029959
                angle_x2 = top_x2+442.5618252549455
                angle_y2 = top_y2+124.53897522533538
                #data.loc[right]['ref1_y2'].mean()-data.loc[right]['ref2_y2'].mean()
                angle_score = np.array([0]).reshape(-1,1)
            elif (len(angle_score)==0 and len(top_score)==0):  #对于两个标识物一个都没有的情况，则状态变为无效帧
                frame_status = "NoRefs"
            else:   #或者都刚好有一个的情况,
                #But twisted refs
                if ((abs(top_x1+top_x2)/2 >= abs(angle_x1+angle_x2)/2) or (abs(top_y1+top_y2)/2 >= abs(angle_y1+angle_y2)/2)):
                    frame_status = "RightSideTwistedRefs"
                #Or all okay
                else:
                    pass 
        return angle_x1, angle_y1, angle_x2, angle_y2, \
               top_x1, top_y1, top_x2, top_y2, \
               angle_score, top_score, angle_name, top_name, frame_status
    def check_heads_outputs(self, heads_x1s, heads_y1s, heads_x2s, heads_y2s, angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2):
        angle_xc = ((angle_x1+angle_x2)/2).reshape(-1)
        angle_yc = ((angle_y1+angle_y2)/2).reshape(-1)
        top_xc = ((top_x1+top_x2)/2).reshape(-1)
        top_yc = ((top_y1+top_y2)/2).reshape(-1)
        heads_keep_idx = []
        for idx in range(len(heads_x1s)):
            head_xc = ((heads_x1s[idx]+heads_x2s[idx])/2).reshape(-1)
            head_yc = ((heads_y1s[idx]+heads_y2s[idx])/2).reshape(-1)
            #判断是否在窗内:
            if (head_xc>min(angle_xc, top_xc) and head_xc<max(angle_xc, top_xc) and head_yc>min(angle_yc, top_yc) and head_yc<max(angle_yc, top_yc)):
                heads_keep_idx.append(idx)
            else:
                print("Outside head dropped...")
                pass    #车窗外舍弃
        heads_x1s = heads_x1s[heads_keep_idx]
        heads_y1s = heads_y1s[heads_keep_idx]
        heads_x2s = heads_x2s[heads_keep_idx]
        heads_y2s = heads_y2s[heads_keep_idx]
        return heads_x1s, heads_y1s, heads_x2s, heads_y2s
    def get_spatial_in_seat_position(self, angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2, heads_x1s, heads_y1s, heads_x2s, heads_y2s):
        if len(heads_x1s)==0:   #如果：没人头
            if len(angle_x1)!=0 or len(top_x1)!=0:   #但出现了标识物
                status = "Empirical, %s head, %s refs, case 1.[1,]"%(len(heads_x1s), len(angle_x1)+len(top_x1))
                return [1,], status    #肯定有车，所以肯定有司机
            else:    #连标识物都没有
                status = "Empirical, %s head, %s refs, case 2.[0,]"%(len(heads_x1s), len(angle_x1)+len(top_x1))
                return [0,], status    #那就算了,什么都没有
        else:                   #如果:有人头
            if len(angle_x1)==1 and len(top_x1)==1:  #并且刚好标识物各有一个,运行神经网络
                #生成输入矩阵 n×12维度， n是人头个数
                inputs_tmp = np.tile(np.array([angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2]).reshape(-1), (len(heads_x1s),1))
                positions_peer_car = {}
                with no_grad():
                    for i in range(len(heads_x1s)):
                        head_pos = np.array([heads_x1s[i], heads_y1s[i], heads_x2s[i], heads_y2s[i]])
                        inputs = FloatTensor(np.hstack((inputs_tmp[i], head_pos))).view(-1,12)
                        position_probabilities, position = spatial_in_seat_train_and_test.frame_in_seat(self.net_spatial, inputs)
                        i = 0
                        #如果该位置已经坐人了：
                        while position in positions_peer_car.keys():
                            i += 1
                            if positions_peer_car[position][0,position-1] >= position_probabilities[0,position-1]:   #如果先前的结果比较确定
                                position = (-position_probabilities).argsort()[0][i]+1   #则取一个新的信任位置给目前的结果。
                            else:          #如果目前的结果更为确定
                                tmp_probabilities = positions_peer_car[position]   #把先前的概率序列取出
                                tmp_position = (-tmp_probabilities).argsort()[0][i]+1   #给一个第二信任的位置给之前的结果。
                                positions_peer_car[position] = position_probabilities   #改掉之前的结果
                                position = tmp_position     #把之前的给成现在的，就好像恰好后得到这个量一样
                                position_probabilities = tmp_probabilities   #把之前的给成现在的，就好像恰好后得到这个量一样
                                #positions_peer_car[tmp_position] = tmp_probabilities   #更改先前的入座结果
                        positions_peer_car[position] = position_probabilities   #只有当position不再出现过，才可以赋值进去
                #Add in driver.
                positions_peer_car[1]="666" if 1 not in positions_peer_car.keys() else positions_peer_car[1]
                _result_ = list(positions_peer_car.keys())
                status = "Predicted, %s head, %s refs, case 3. %s"%(len(heads_x1s), len(angle_x1)+len(top_x1), _result_)
                return _result_, status
            elif len(angle_x1)==0 and len(top_x1)==0:   #没有任何标识物
                possible_seat = [1,2,3,4,5]  #将从经验位置顺序里直接取
                _result_ = possible_seat[:len(heads_x1s)]
                status = "Empirical, %s head, %s refs, case 4. %s"%(len(heads_x1s), len(angle_x1)+len(top_x1), _result_)
                return _result_, status 
            else:
                print("?????")   #之前已经强制就绪了两个标识物，要么都有要么都没有，不该出现这种情况。
                sys.exit()
    def ignore_5(self, positions_peer_car):
        move_to_seat = 3 if self.side == 'left' else 4
        try:
            positions_peer_car[np.where(np.array(positions_peer_car)==5)[0][0]]=move_to_seat
        except:
            pass
        return positions_peer_car
    def self_logic(self, image_data):
        if image_data.dtype == np.uint8:   #If come from C
            print("Must be C running...")
            image_data = image_data.astype(float)/255
        global_signal = 1
        #print(image_data.min(), image_data.mean(), image_data.max())
        cam_frame = image_data
        #判定当前全局信号，GPU是否开始检测
        #Preprocess cam data:
        angle_cam_frame = self.preprocess_cam_frame(cam_frame)
        net_cam_frame = image_data*255
        #print(image_data.min(), image_data.mean(), image_data.max())
        start_time = time.time()
        #检测标识物和人头:
        refs_x1s, refs_y1s, refs_x2s, refs_y2s, refs_scores, refs_label_names, \
            heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores, heads_names \
                =  self.get_objs_position(angle_cam_frame, net_cam_frame)
        #对标识物的经验审查与修补：
        angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2, angle_score, top_score, angle_name, top_name, frame_status = self.check_refs_outputs(refs_x1s, refs_y1s, refs_x2s, refs_y2s, refs_scores, refs_label_names)
        if frame_status == "ok":
            #对人头的经验审查与修补：
            heads_x1s, heads_y1s, heads_x2s, heads_y2s = self.check_heads_outputs(heads_x1s, heads_y1s, heads_x2s, heads_y2s, angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2)
            #位置判别网络：
            positions_peer_car, status = self.get_spatial_in_seat_position(angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2, heads_x1s, heads_y1s, heads_x2s, heads_y2s)
            #注意：暂时忽略5位置！！！！！！！！！！！！！！！！！！！！！！！
            #positions_peer_car = self.ignore_5(positions_peer_car)
            time_used = time.time() - start_time
        else:
            status = "Frame Skipped since (%s)"%frame_status
            positions_peer_car = [0,]
            time_used = time.time() - start_time
        print("Net:", self.netname, "Time_used:", np.round(time_used, 4), "Judge_stauts:", status, "Positions_peer_frame_peer_side:************", positions_peer_car, "********")
        #并且启动旁侧程序：
        #pass, 两侧可能等价，同时启动。
        #Record current status:
        self.seq_ground_signal.append(global_signal)
        self.seq_threshold_signal.append(0.5)
        #VISUALIZATION:
        if VISUALIZATION:
            #Plot1:
            plt.clf()
            ax1 = plt.subplot(211)
            ax1.imshow(cam_frame)
            ax1.set_title("View of Camera %s, always keep running"%self.netname)
            ax1.axis('off')
            if global_signal is 0:    #没有全局信号，则plot1 直接留空
                network_status = "Reference point detection: Standby\n\nHead detection: Standby\n\nNo signal received, not running"
            else:     #有全局信号， 开始plot1
                network_status = "Reference point detection: Running\nHead detection: Running\n\n%s cars detected, %s heads detected"%(len(refs_x1s), len(heads_x1s))
                #plot1中画标识物：
                if UNVEIL:
                    currentAxis = plt.gca()
                    angle_color = 'r' if angle_score != 0 else 'blue'
                    rect_1 = matplotlib.patches.Rectangle((angle_x1, angle_y1), angle_x2-angle_x1, angle_y2-angle_y1, linewidth=1, edgecolor=angle_color, facecolor=angle_color, alpha=0.4)
                    top_color = 'r' if top_score != 0 else 'blue'
                    rect_2 = matplotlib.patches.Rectangle((top_x1, top_y1), top_x2-top_x1, top_y2-top_y1, linewidth=1, edgecolor=top_color, facecolor=top_color, alpha=0.4)
                    if len(angle_score)>0:
                        ax1.text(angle_x1, angle_y1,"class:%s, score:%s"%(angle_name, np.round(angle_score,2)), color=angle_color)
                        currentAxis.add_patch(rect_1)
                    if len(top_score)>0: 
                        ax1.text(top_x1, top_y1,"class:%s, score:%s"%(top_name, np.round(top_score,2)), color=top_color)
                        currentAxis.add_patch(rect_2)
            #plot1中画人头：
            for heads_idx, heads_i in enumerate(heads_x1s):
                currentAxis = plt.gca()
                heads_rect = matplotlib.patches.Rectangle((heads_x1s[heads_idx], heads_y1s[heads_idx]), heads_x2s[heads_idx]-heads_x1s[heads_idx], heads_y2s[heads_idx]-heads_y1s[heads_idx], linewidth=1, edgecolor='green', facecolor='green', alpha=0.3)
                ax1.text(heads_x1s[heads_idx], heads_y1s[heads_idx],"class:%s, score:%s"%(heads_names[heads_idx], np.round(heads_scores[heads_idx],2)), color=[0,1,0])
                currentAxis.add_patch(heads_rect)
            ax1.set_title("View of Camera %s, always keep running"%self.netname)
            ax1.axis('off')
            seats = np.zeros(shape=(1,5))
            if positions_peer_car==[0]:
                pass
            else:
                seats[0, np.array(positions_peer_car)-1] = 1
            if UNVEIL:
                #Plot2:
                ax2 = plt.subplot(223)
                sns.heatmap(seats, linewidths=0.1, vmin=0, vmax=2, cmap='Reds', square=True, linecolor='white', annot=True, ax=ax2, xticklabels=['1','2','3','4','5'])
                ax2.set_xlabel("Seat number")
                ax2.set_ylabel("Seat occupied")
                ax2.set_title(network_status)
            #Plot3:
            if UNVEIL:
                ax3 = plt.subplot(224)
            else:
                ax3 = plt.subplot(212)
            if np.where(seats[0]!=0)[0].shape[0]!=0:
                view_name = self.root_dir+'/views/'+''.join(list((np.where(seats[0]!=0)[0]+1).astype(str)))+".png"
            else:
                view_name = self.root_dir+'/views/0.png'
            #embed()
            ax3.imshow(plt.imread(view_name))
            plt.draw()
            plt.pause(0.001)
            #input()
        return [positions_peer_car, plt]

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description = "Front A Net...")
    #parser.add_argument('-V', '--visualization', action="store_true", default=False)
    #args = parser.parse_args()
    #embed()
    #Initializing:
    print("Initializing front camera A...")
    root_dir = "/".join(os.getcwd().split('/')[:-1])
    A_program = A(root_dir)
    print("Real_time running...")
    filelist = glob.glob("../left/*.jpg")    #TODO : Feed A.py left data to get correct results
    for idx, filename in enumerate(filelist[:]):
        image_data = camera.get_image_data(filename)
        A_program.self_logic(image_data)

