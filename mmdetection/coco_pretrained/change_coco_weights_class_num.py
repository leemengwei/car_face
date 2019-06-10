# for cascade rcnn
import torch
num_classes = 6
model_coco = torch.load("cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth")
#model_coco = torch.load("cascade_rcnn_hrnetv2_w32_fpn_20e_20190522-55bec4ee.pth")

# weight
model_coco["state_dict"]["bbox_head.0.fc_cls.weight"].resize_(num_classes,1024)
model_coco["state_dict"]["bbox_head.1.fc_cls.weight"].resize_(num_classes,1024)
model_coco["state_dict"]["bbox_head.2.fc_cls.weight"].resize_(num_classes,1024)
# bias
model_coco["state_dict"]["bbox_head.0.fc_cls.bias"].resize_(num_classes)
model_coco["state_dict"]["bbox_head.1.fc_cls.bias"].resize_(num_classes)
model_coco["state_dict"]["bbox_head.2.fc_cls.bias"].resize_(num_classes)
#save new model
torch.save(model_coco,"rcnn_for_car_face_%d.pth"%num_classes)
