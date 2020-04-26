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
#plt.ion()
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
    def __init__(self, root_dir, side, time_num=1):
        super(A, self).__init__()
        self.root_dir = root_dir
        if type(self.root_dir) is bytes:
            self.root_dir = self.root_dir.decode("utf-8")
        self.side = side
        self.time_num = time_num
        #Init model:
        #Sptial net 载入结构：
        self.net_spatial = spatial_model.NeuralNet(input_size=12, hidden_size= 100, hidden_depth=3, output_size=config.NUM_OF_SEATS_PEER_CAR)
        self.net_spatial_param = load('%s/%s'%(self.root_dir, config.SPATIAL_IN_SEAT_MODEL))
        self.net_spatial.load_state_dict(self.net_spatial_param['model_state_dict'])
        #Cascade net 载入结构:
        if not 'back' in self.side:
            self.net_to_detect_objs = self.get_mmd_model_and_template(MMD_FRONT_CONFIG, MMD_FRONT_WEIGHTS)
        else:   #side is back
            self.net_to_detect_objs = self.get_mmd_model_and_template(MMD_BACK_CONFIG, MMD_BACK_WEIGHTS)
        #On gpu:
        self.net_spatial.cuda().eval()
        self.net_to_detect_objs.cuda().eval()
        #self.net_to_detect_objs_night.cuda().eval()
        print("Side %s-%s Initialized."%(self.side, self.time_num))
    def get_mmd_model_and_template(self, _mmd_config, _mmd_weights):
        cfg = mmcv.Config.fromfile(_mmd_config)
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        cfg.data.test.test_mode = True
        model = init_detector(_mmd_config, _mmd_weights)
        return model
    def drop_small_heads(self, heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores):
        #Drop if too small:
        head_heights = heads_y2s-heads_y1s 
        head_widths = heads_x2s-heads_x1s
        if 'back' in self.side:
            where_too_small = head_heights*head_widths<config.BACK_HEAD_TOO_SMALL   #for back, neither side too small is wrong
            #where_too_small = (head_heights<config.BACK_HEAD_TOO_SMALL)|(head_widths<config.BACK_HEAD_TOO_SMALL)   #for back, neither side too small is wrong
        else:   #front too smal
            where_too_small = head_heights*head_widths<config.HEAD_TOO_SMALL    #for front, we can accept one side small scenario
            #where_too_small = (head_heights<config.HEAD_TOO_SMALL)&(head_widths<config.HEAD_TOO_SMALL)    #for front, we can accept one side small scenario
        heads_x1s = heads_x1s[~where_too_small]
        heads_x2s = heads_x2s[~where_too_small]
        heads_y1s = heads_y1s[~where_too_small]
        heads_y2s = heads_y2s[~where_too_small]
        heads_scores = heads_scores[~where_too_small]
        if len(np.where(where_too_small==True)[0])>=1:
            print("There are %s small head dropped..."%len(np.where(where_too_small==True)))
        #Drop if too big:
        #head_heights = heads_y2s-heads_y1s 
        #head_widths = heads_x2s-heads_x1s
        #if 'back' in self.side:
        #    where_too_big = (head_heights>config.BACK_HEAD_TOO_BIG)|(head_widths>config.BACK_HEAD_TOO_BIG)   #for back, neither side too big is wrong
        #else:
        #    where_too_big = (head_heights>999)|(head_widths>999)   #temporary no limit for front side
        #heads_x1s = heads_x1s[~where_too_big]
        #heads_x2s = heads_x2s[~where_too_big]
        #heads_y1s = heads_y1s[~where_too_big]
        #heads_y2s = heads_y2s[~where_too_big]
        #heads_scores = heads_scores[~where_too_big]
        #if len(np.where(where_too_big==True)[0])>=1:
        #    print("There are %s big head dropped..."%len(np.where(where_too_big==True)))
        return heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores
    def get_objs_position(self, net_cam_frame, CONFIDENCE_THRESHOLDS):
        classes = config.CLASSES_4
        with no_grad():
            objs_x1s, objs_y1s, objs_x2s, objs_y2s, objs_scores, objs_indexes, objs_elapsed_time = test_fix.single_gpu_frame_detection(self.net_to_detect_objs, net_cam_frame, CONFIDENCE_THRESHOLDS, show=False)
            print("Detection(%s-%s):"%(self.side, self.time_num), objs_indexes, objs_scores)
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
        #Drop small refs, and then make pair of refs
        top_widths = tops_x2s-tops_x1s
        top_heights = tops_y2s-tops_y1s
        where_too_small = ~np.add(~(top_heights<config.TOP_TOO_SMALL) , ~(top_widths<config.TOP_TOO_SMALL))
        tops_x1s = tops_x1s[~where_too_small]
        tops_x2s = tops_x2s[~where_too_small]
        tops_y1s = tops_y1s[~where_too_small]
        tops_y2s = tops_y2s[~where_too_small]
        tops_scores = tops_scores[~where_too_small]
        angle_widths = angles_x2s-angles_x1s
        angle_heights = angles_y2s-angles_y1s
        where_too_small = ~np.add(~(angle_heights<config.ANGLE_TOO_SMALL) , ~(angle_widths<config.ANGLE_TOO_SMALL))
        angles_x1s = angles_x1s[~where_too_small]
        angles_x2s = angles_x2s[~where_too_small]
        angles_y1s = angles_y1s[~where_too_small]
        angles_y2s = angles_y2s[~where_too_small]
        angles_scores = angles_scores[~where_too_small]
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
        #====================SIDE JUDGE===================:
        #强制就绪两个标识物,(强制就绪意在解决明明有但没检测到的情况，包括车超出画面的情况,也会被强制就绪)
        if self.side is "left":
            if (len(angle_score)==1 and len(top_score)==0):
                #top_x1 = angle_x1+685.0948918547668
                #top_y1 = angle_y1-125.42961902952528
                #top_x2 = angle_x2+660.4268173950429
                #top_y2 = angle_y2-159.48261207951592
                top_x1 = angle_x1+config.WINDOW_WIDTH
                top_y1 = angle_y1-config.WINDOW_HEIGHT
                top_x2 = top_x1+100
                top_y2 = top_y1-100
                top_score = np.array([0]).reshape(-1,1)
                frame_status = "Fixed"
            elif (len(angle_score)==0 and len(top_score)==1):
                #angle_x1 = top_x1-685.0948918547668
                #angle_y1 = top_y1+125.42961902952528
                #angle_x2 = top_x2-660.4268173950429
                #angle_y2 = top_y2+159.48261207951592
                angle_x1 = top_x1-config.WINDOW_WIDTH
                angle_y1 = top_y1+config.WINDOW_HEIGHT
                angle_x2 = angle_x1-100
                angle_y2 = angle_y1+100
                angle_score = np.array([0]).reshape(-1,1)
                frame_status = "Fixed"
            elif (len(angle_score)==0 and len(top_score)==0):   #对于两个标识物一个都没有的情况，则状态变为无框帧
                frame_status = "NoRefs"
            else:   #或者都刚好有一个的情况,
                if ((abs(top_x1+top_x2)/2 <= abs(angle_x1+angle_x2)/2) or (abs(top_y1+top_y2)/2 >= abs(angle_y1+angle_y2)/2)):
                    frame_status = "LeftSideMixedRefs"
                else:
                    pass    #或者都刚好有一个的情况，不处理。
        elif self.side is "right":
            if (len(angle_score)==1 and len(top_score)==0):
                #top_x1 = angle_x1-579.6467673760624
                #top_y1 = angle_y1-98.41134448029959
                #top_x2 = angle_x2-442.5618252549455
                #top_y2 = angle_y2-124.53897522533538
                top_x1 = angle_x1-config.WINDOW_WIDTH
                top_y1 = angle_y1-config.WINDOW_HEIGHT
                top_x2 = top_x1-100
                top_y2 = top_y1-100
                top_score = np.array([0]).reshape(-1,1)
                frame_status = "Fixed"
            elif (len(angle_score)==0 and len(top_score)==1):
                #angle_x1 = top_x1+579.6467673760624
                #angle_y1 = top_y1+98.41134448029959
                #angle_x2 = top_x2+442.5618252549455
                #angle_y2 = top_y2+124.53897522533538
                angle_x1 = top_x1+config.WINDOW_WIDTH
                angle_y1 = top_y1+config.WINDOW_HEIGHT
                angle_x2 = angle_x1+100
                angle_y2 = angle_y1+100
                #data.loc[right]['ref1_y2'].mean()-data.loc[right]['ref2_y2'].mean()
                angle_score = np.array([0]).reshape(-1,1)
                frame_status = "Fixed"
            elif (len(angle_score)==0 and len(top_score)==0):  #对于两个标识物一个都没有的情况，则状态变为无框帧
                frame_status = "NoRefs"
            else:   #或者都刚好有一个的情况,
                if ((abs(top_x1+top_x2)/2 >= abs(angle_x1+angle_x2)/2) or (abs(top_y1+top_y2)/2 >= abs(angle_y1+angle_y2)/2)):
                    frame_status = "RightSideTwistedRefs"
                else:
                    pass
        else:  #self.side is "backleft or backright"
           frame_status = "ThisIsBackSide"
           pass
        return angle_x1, angle_y1, angle_x2, angle_y2, \
               top_x1, top_y1, top_x2, top_y2, \
               angle_score, top_score, angle_name, top_name, frame_status

    def check_heads_outputs_front(self, heads_x1s, heads_y1s, heads_x2s, heads_y2s, angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2, heads_scores):
        #Judge if head big enough
        heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores = self.drop_small_heads(heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores)
        #Drop if outside:
        angle_xc = ((angle_x1+angle_x2)/2).reshape(-1)
        angle_yc = ((angle_y1+angle_y2)/2).reshape(-1)
        top_xc = ((top_x1+top_x2)/2).reshape(-1)
        top_yc = ((top_y1+top_y2)/2).reshape(-1)
        heads_x_center = ((heads_x1s+heads_x2s)/2).reshape(-1)
        heads_y_center = ((heads_y1s+heads_y2s)/2).reshape(-1)
        top_width = abs(top_x1-top_x2).reshape(-1)
        if self.side is "left":
            left_right_within = (heads_x_center>angle_xc)&(heads_x_center<top_xc+0.6*top_width)
        elif self.side is "right":
            left_right_within = (heads_x_center<angle_xc)&(heads_x_center>top_xc-0.6*top_width)
        else:   #side is back
            print("This check function should not be called in backside")
            sys.exit()
        up_down_within = (heads_y_center<angle_yc)&(heads_y_center>top_yc)  #image pixel y is opposite
        where_within = up_down_within&left_right_within
        heads_keep_idx = np.where(where_within==True)[0]
        if len(heads_keep_idx)-len(heads_x_center)!=0:
            print("There are %s outside head dropped..."%(len(heads_x_center)-len(heads_keep_idx)))
        heads_x1s = heads_x1s[heads_keep_idx]
        heads_y1s = heads_y1s[heads_keep_idx]
        heads_x2s = heads_x2s[heads_keep_idx]
        heads_y2s = heads_y2s[heads_keep_idx]
        heads_scores = heads_scores[heads_keep_idx]
        return heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores
    def get_spatial_in_seat_position(self, angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2, heads_x1s, heads_y1s, heads_x2s, heads_y2s):
        if len(heads_x1s)==0:   #如果：没人头
            status = "No head, but must have car,"
            return [1,], status    #肯定有车，所以肯定有司机
        else:                   #如果:有人头
            #生成输入矩阵 n×12维度， n是人头个数
            if "back" in self.side:
                #仅以区分前后映射空间的不同
                #inputs_tmp = np.tile(np.array([-angle_x1, -angle_y1, -angle_x2, -angle_y2, -top_x1, -top_y1, -top_x2, -top_y2]).reshape(-1), (len(heads_x1s),1))
                print("Should not be here")
                sys.exit()
            else:  #fronts
                inputs_tmp = np.tile(np.array([angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2]).reshape(-1), (len(heads_x1s),1))
            positions_peer_side = {}
            position_probabilities_list = []
            position_list = []
            with no_grad():
                for i in range(len(heads_x1s)):
                    head_pos = np.array([heads_x1s[i], heads_y1s[i], heads_x2s[i], heads_y2s[i]])
                    inputs = FloatTensor(np.hstack((inputs_tmp[i], head_pos))).view(-1,12)
                    position_probabilities, position = spatial_in_seat_train_and_test.frame_in_seat(self.net_spatial, inputs)
                    position_probabilities_list.append(position_probabilities)
                    position_list.append(position)
            pos_raw = list(np.argmax(position_probabilities_list, axis=1)+1)
            #Several special treats for loc error:
            #sp 0) left NO 4->1, right NO 3->2, but first to adjust obvious to 4,3
            heads_center_x = ((heads_x1s+heads_x2s)/2).reshape(-1)
            heads_width = np.abs(heads_x1s-heads_x2s).reshape(-1)
            top_center_x = ((top_x1+top_x2)/2).reshape(-1)
            if self.side is 'left':
                if len(np.where(heads_center_x>top_center_x)[0])>0:
                    pos_raw[np.where(heads_center_x>top_center_x)[0][0]] = 4
                if 4 in pos_raw and len(top_center_x)==1:
                    if float(heads_center_x[pos_raw.index(4)])<float(top_center_x):
                        print("Correcting obvious left 4 to 1")
                        pos_raw[pos_raw.index(4)] = 1
            if self.side is 'right':
                if len(np.where(heads_center_x<top_center_x)[0])>0:
                    pos_raw[np.where(heads_center_x<top_center_x)[0][0]] = 3
                if 3 in pos_raw and len(top_center_x)==1:
                    if float(heads_center_x[pos_raw.index(3)])>float(top_center_x):
                        print("Correcting right 3 to 2")
                        pos_raw[pos_raw.index(3)] = 2
            #1) right front只看到单4的情况，几乎不可能发生，但又可能因为司机后仰而导致location错误，故这种情况4修正为1
            if self.side is 'right':
                if pos_raw == [4]:
                    pos_raw = [1]
                if pos_raw == [4,4]:
                    pos_raw = [1,4]
            #2) spectial treat for right side pos 2: if pos 2 is too far away from its top, then it's 5 标准：绝大多数贴着，给一个最小2号的宽度冗余：
            if self.side is 'right' and 2 in pos_raw:
                if pos_raw.count(2)>=1:
                    check_at = np.where(np.array(pos_raw)==2)[0]
                    criterion = (np.abs(heads_x1s-heads_x2s)/2)[check_at].min()
                    for check_this in check_at:
                        if min(heads_x1s[check_this], heads_x2s[check_this])-max(top_x1,top_x2)>criterion:
                            width_this = np.abs(heads_x1s[check_this]-heads_x2s[check_this]).min()
                            if pos_raw.count(1)>0 and max(heads_x1s[check_this], heads_x2s[check_this])+1.1*width_this>min(heads_x1s[np.where(np.array(pos_raw)==1)],heads_x2s[np.where(np.array(pos_raw)==1)]):
                                print("A right 2 is Toofaraway! but close to 1, Current alter to 4")   #It depends:
                                pos_raw[check_this] = 4
                            else:
                                print("A right 2 is Toofaraway!!! Current alter to 5")   #It depends:
                                pos_raw[check_this] = 5
            #sp 3) from left, if multi back seats but no 2, then must be a 2 and a back
            if self.side is 'left' and 2 not in pos_raw:
                if pos_raw.count(3)==2:
                    pos_raw = [1,2,3]
                if pos_raw.count(3)==3:
                    pos_raw = [1,2,3,3]
            #sp 4) from right, if multi back seats but no 1, then must be a 1 and a back
            if self.side is 'right' and 1 not in pos_raw:
                if pos_raw.count(4)>=1:
                    pos_raw.remove(4)  #very likely to be 1
                    pos_raw+=[1]
            status = "Predicted"
            counter = np.array([pos_raw.count(i+1) for i in range(config.NUM_OF_SEATS_PEER_CAR)])
            #print(counter)
            for where_multi,i in enumerate(counter):
                where_multi+=1
                if i<=1:
                    continue
                print("raw taken:", pos_raw)
                status = "Modified"
                #all else:
                if self.side is "left":
                    if where_multi==1:  #It depends!
                        ones_at = np.where(np.array(pos_raw)==1)
                        if heads_width[ones_at].sum()+config.SLIT<np.array([heads_x1s[ones_at],heads_x2s[ones_at]]).max()-np.array([heads_x1s[ones_at],heads_x2s[ones_at]]).min():  #when not Overlap (there's a slit between)
                            pos_raw += [config.LEFT_FILL]
                            print("left multi1 as %s"%config.LEFT_FILL)    
                        else:
                            pos_raw += [5]
                            print("left multi1 as 5") 
                    if where_multi==2: 
                        pos_raw += [3]
                        print("a 2 as 3")
                    if where_multi==3:   
                        print("multi3pass")
                        pass
                    if where_multi==4:
                        print("multi4pass") 
                    if where_multi==5:   
                        print("multi5pass")
                else:  # self.side is "right":
                    if where_multi==1:
                        print("right multi1 as 4") 
                        pos_raw += [4]
                    if where_multi==2:  #It depends!
                        twos_at = np.where(np.array(pos_raw)==2)
                        if heads_width[twos_at].sum()+config.SLIT<np.array([heads_x1s[twos_at],heads_x2s[twos_at]]).max()-np.array([heads_x1s[twos_at],heads_x2s[twos_at]]).min():  #when not Overlap, far as 4, there's a slit between
                            pos_raw += [config.RIGHT_FILL]
                            print("right multi2 far to 2 as %s"%config.RIGHT_FILL)    
                        else:
                            pos_raw += [5]
                            print("right multi2 close to 2 as 5") 
                    if where_multi==3:
                        print("multi3pass") 
                    if where_multi==4:
                        print("multi4pass")
                        pass
                    if where_multi==5:  
                        print("multi5pass")
            #帧内：对前侧相机来说只有单张图上看到33或44才算345
            #Consider: if right no 1, then 4 must be 1, if left multi 3 and no 2, then must be 2.
            #if (3 in pos_raw and 5 in pos_raw) or (4 in pos_raw and 5 in pos_raw):
            if self.side is "left":
                if pos_raw.count(3)==2:
                    if 2 not in pos_raw:
                        pos_raw.remove(3)
                        pos_raw += [2]
                    else:
                        pos_raw = pos_raw+[3,4,5]
                elif 3 in pos_raw and 5 in pos_raw:
                    pos_raw += [3,4,5]
                else:
                    pass
            else:   #side is right
                if pos_raw.count(4)==2:
                    if 1 not in pos_raw:
                        pos_raw.remove(4)
                        pos_raw += [1]
                    else:
                        pos_raw = pos_raw+[3,4,5]
                elif 4 in pos_raw and 5 in pos_raw:
                    pos_raw += [3,4,5]
                else:
                    pass
            pos_taken = list(set(pos_raw+[1]))
            #print(pos_taken)
            _result_ = list(pos_taken)
            #embed()
            return _result_, status

    def ignore_5(self, positions_peer_side):
        move_to_seat = 3 if self.side == 'left' else 4
        try:
            positions_peer_side[np.where(np.array(positions_peer_side)==5)[0][0]]=move_to_seat
            print("Moving 5 to %s"%move_to_seat)
        except:
            pass
        return positions_peer_side

    def check_heads_outputs_back(self, image_data, heads_x1s, heads_y1s, heads_x2s, heads_y2s, top_x1, top_y1, top_x2, top_y2, heads_scores):
        image_shape = image_data.shape
        _old_len = len(heads_scores)
        heads_center_x = ((heads_x1s+heads_x2s)/2).reshape(-1)
        center_top_x = ((top_x1+top_x2)/2).reshape(-1)
        if self.side == 'backleft':
            validate_line = image_shape[1]*(3/5) #*99
            keeps = np.where(heads_center_x<validate_line)
            if len(center_top_x)>0:
                keeps = np.where(heads_center_x>center_top_x.min())
        elif self.side == 'backright':
            validate_line = image_shape[1]*(2/5)  #*0
            keeps = np.where(heads_center_x>validate_line)
            if len(center_top_x)>0:
                keeps = np.where(heads_center_x<center_top_x.max())
        else:
            print("Should not be here!")
            sys.exit()
        heads_x1s = heads_x1s[keeps]
        heads_x2s = heads_x2s[keeps]
        heads_y1s = heads_y1s[keeps]
        heads_y2s = heads_y2s[keeps]
        heads_scores = heads_scores[keeps]
        if len(keeps[0])<_old_len:
            print("Back report: heads are droped due to region position!")
        heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores = self.drop_small_heads(heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores)
        return heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores

    def self_logic(self, image_data, CONFIDENCE_THRESHOLDS):
        if image_data.dtype == np.uint8:   #If come from C
            #print("Must be C running...")
            image_data = image_data.astype(float)/255
        #TODO: 当后侧相机传来图像，该图像的分辨率和前侧不一样。目前前后模型一起训练的，也就是说后侧数据在训练和测试的时候都会被压扁然后经过网络。
        #print(image_data.min(), image_data.mean(), image_data.max())
        #判定当前全局信号，GPU是否开始检测
        #Preprocess cam data:
        net_cam_frame = image_data*255
        #print(image_data.min(), image_data.mean(), image_data.max())
        start_time = time.time()
        #检测标识物和人头:
        refs_x1s, refs_y1s, refs_x2s, refs_y2s, refs_scores, refs_label_names, heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores, heads_names = self.get_objs_position(net_cam_frame, CONFIDENCE_THRESHOLDS)
        #对标识物的经验审查与修补：
        angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2, angle_score, top_score, angle_name, top_name, frame_status = self.check_refs_outputs(refs_x1s, refs_y1s, refs_x2s, refs_y2s, refs_scores, refs_label_names)
        #前侧则：
        if self.side == "left" or self.side =="right":
            if frame_status == "ok":    #识别或补充到两个标识物才是ok
                #对人头的经验审查与修补：
                heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores = self.check_heads_outputs_front(heads_x1s, heads_y1s, heads_x2s, heads_y2s, angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2, heads_scores) 
                #位置判别：
                positions_peer_side, status = self.get_spatial_in_seat_position(angle_x1, angle_y1, angle_x2, angle_y2, top_x1, top_y1, top_x2, top_y2, heads_x1s, heads_y1s, heads_x2s, heads_y2s)
                #暂时忽略5位置
                if config.IGNORE_5:
                    positions_peer_side = self.ignore_5(positions_peer_side)
            else:
                status = "Frame Skipped since (%s)"%frame_status
                positions_peer_side = [0,]
        #是后侧
        else:
            heads_x1s, heads_y1s, heads_x2s, heads_y2s, heads_scores = self.check_heads_outputs_back(image_data, heads_x1s, heads_y1s, heads_x2s, heads_y2s, top_x1, top_y1, top_x2, top_y2, heads_scores) 
            #后侧直接猜，不使用任何定位网络，也不在乎 frame_status。
            status = "Direct Guess (%s)"%self.side
            heads_center_x = ((heads_x1s+heads_x2s)/2).reshape(-1)
            top_center_x = ((top_x1+top_x2)/2).reshape(-1)
            heads_width = np.abs(heads_x1s-heads_x2s).reshape(-1)
            if self.side == "backleft":
                positions_peer_side = [4,5,5,5,5][:min(len(heads_x1s),5)]  #[1, 4, 5, 3, 2][:min(len(heads_x1s),5)]
            else:   # self.side == "backright"
                positions_peer_side = [3,5,5,5,5][:min(len(heads_x1s),5)] #[2, 3, 5, 4, 1][:min(len(heads_x1s),5)]
            #LOOK Through Correction:
            #if len(heads_center_x)==2:
            #    if abs(heads_center_x[0] - heads_center_x[-1])>1.0*heads_width.min():
            #        print("Warning, two heads far away, might look through, drop one")
            #        positions_peer_side.remove(5)
            if len(heads_center_x)==1 and len(top_center_x)==1:
                if self.side == 'backleft':
                    #more strict
                    #if heads_center_x[0]<max(top_x1[0], top_x2[0]):
                    if min(heads_x1s, heads_x2s)<max(top_x1[0], top_x2[0]):
                        print("Warning, too close to corner, might look through, drop")
                        positions_peer_side.remove(4)
                if self.side == 'backright':
                    #if heads_center_x[0]>min(top_x1[0], top_x2[0]):
                    #more strict
                    if max(heads_x1s, heads_x2s)>min(top_x1[0], top_x2[0]):
                        print("Warning, too close to corner, might look through, drop")
                        positions_peer_side.remove(3)
            positions_peer_side = [3,4,5] if 5 in positions_peer_side else positions_peer_side
            positions_peer_side = [0] if len(positions_peer_side)==0 else positions_peer_side
        time_used = time.time() - start_time
        print("Localization(%s-%s):"%(self.side, self.time_num), status, "***", positions_peer_side, "***")

        #VISUALIZATION:
        if VISUALIZATION:
            #Plot1:
            plt.clf()
            ax1 = plt.subplot(211)
            ax1.imshow(image_data)
            ax1.set_title("View of Camera %s%s, always keep running"%(self.side, self.time_num))
            ax1.axis('off')
            #plot1中画标识物：
            if UNVEIL:
                currentAxis = plt.gca()
                angle_color = 'r' if angle_score != 0 else 'blue'
                rect_1 = matplotlib.patches.Rectangle((angle_x1, angle_y1), angle_x2-angle_x1, angle_y2-angle_y1, linewidth=2, edgecolor=angle_color, fill=False)
                top_color = 'r' if top_score != 0 else 'blue'
                rect_2 = matplotlib.patches.Rectangle((top_x1, top_y1), top_x2-top_x1, top_y2-top_y1, linewidth=2, edgecolor=top_color, fill=False)
                if len(angle_score)>0:
                    ax1.text(angle_x1, angle_y1,"%s,%s"%(angle_name, np.round(angle_score,2)), color=angle_color)
                    currentAxis.add_patch(rect_1)
                if len(top_score)>0: 
                    ax1.text(top_x1, top_y1,"%s,%s"%(top_name, np.round(top_score,2)), color=top_color)
                    currentAxis.add_patch(rect_2)
            #plot1中画人头：
            for heads_idx, heads_i in enumerate(heads_x1s):
                currentAxis = plt.gca()
                heads_rect = matplotlib.patches.Rectangle((heads_x1s[heads_idx], heads_y1s[heads_idx]), heads_x2s[heads_idx]-heads_x1s[heads_idx], heads_y2s[heads_idx]-heads_y1s[heads_idx], linewidth=2, edgecolor='lightgreen', fill=False)
                ax1.text(heads_x1s[heads_idx], heads_y1s[heads_idx],"%s,%s"%(heads_names[heads_idx], np.round(heads_scores[heads_idx],2)), color=[0,1,0])
                currentAxis.add_patch(heads_rect)
            ax1.set_title("View of Camera %s, always keep running"%self.side)
            ax1.axis('off')
            seats = np.zeros(shape=(1,5))
            if positions_peer_side==[0]:
                pass
            else:
                seats[0, np.array(positions_peer_side)-1] = 1
            if UNVEIL:
                #Plot2:
                ax2 = plt.subplot(223)
                sns.heatmap(seats, linewidths=0.1, vmin=0, vmax=2, cmap='Reds', square=True, linecolor='white', annot=True, ax=ax2, xticklabels=['1','2','3','4','5'])
                ax2.set_xlabel("Seat number")
                ax2.set_ylabel("Seat occupied")
            #Plot3:
            if UNVEIL:
                ax3 = plt.subplot(224)
            else:
                ax3 = plt.subplot(212)
            if np.where(seats[0]!=0)[0].shape[0]!=0:
                view_name = self.root_dir+'/views/'+''.join(list((np.where(seats[0]!=0)[0]+1).astype(str)))+".png"
            else:
                view_name = self.root_dir+'/views/0.png'
            if os.path.exists(view_name):
                ax3.imshow(plt.imread(view_name))
            #plt.draw()
            #plt.pause(0.001)
            #input()
            #plt.close()
        return [positions_peer_side, plt]

#if __name__ == "__main__":
#    #parser = argparse.ArgumentParser(description = "Front A Net...")
#    #parser.add_argument('-V', '--visualization', action="store_true", default=False)
#    #args = parser.parse_args()
#    #embed()
#    #Initializing:
#    print("Initializing front camera A...")
#    root_dir = "/".join(os.getcwd().split('/')[:-1])
#    A_program = A(root_dir, "left")
#    print("Real_time running...")
#    filelist = glob.glob("../left/*.jpg")   
#    for idx, filename in enumerate(filelist[:]):
#        image_data = camera.get_image_data(filename)
#        CONFIDENCE_THRESHOLDS = config.get_confidence()
#        A_program.self_logic(image_data, CONFIDENCE_THRESHOLDS)
#
#
