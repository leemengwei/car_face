import numpy as np
import object_detection_dataloader as dataloader
from torch import transpose, FloatTensor
import skimage.io

class camera(object):
    def __init__(self, capnum=0):
        #Init camera:
        w = 1920
        h = 1080
        self.resizer = dataloader.Resizer()
        self.normalizer = dataloader.Normalizer()
    def preprocess_cam_frame(self, cam_frame):
        tmp = self.resizer({'img':cam_frame,'annot':np.array([[0]], dtype=float)})
        resized_cam_frame = tmp['img']
        normalized_cam_frame = self.normalizer({'img':resized_cam_frame.numpy(),'annot':np.array([[0]])})['img']
        net_cam_frame = transpose(transpose(FloatTensor(normalized_cam_frame),2,1),1,0).unsqueeze(0)
        return net_cam_frame

    def get_image_data(filename=None):
        #获取相机图像
        #cam_frame = self.get_camera_view()
        #手动读图片，不走相机：
        if not filename:
            filename = ("/mfs/home/limengwei/car_face/car_face/object_detection_data_both_side/JPEGImages/2019-03-22-11-19-11-216.jpg")
        else:
            pass
        cam_frame = skimage.io.imread(filename)
        positions_peer_car = [0,]
        if len(cam_frame.shape) == 2:
            cam_frame = skimage.color.gray2rgb(cam_frame).astype(np.float32)/255.0
        return cam_frame
       
        
        
        
