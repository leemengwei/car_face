# for cascade rcnn
from IPython import embed
import torch
num_classes = 4

#model_name = "cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth"
#model_name = "cascade_mask_rcnn_r50_fpn_20e_20181123-6e0c9713.pth"

#model_name = "cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth"
model_name = "cascade_rcnn_hrnetv2_w32_fpn_20e_20190522-55bec4ee.pth"

model_coco = torch.load("%s"%model_name)
#embed()

# weight
model_coco["state_dict"]["bbox_head.0.fc_cls.weight"].resize_(num_classes,1024)
model_coco["state_dict"]["bbox_head.1.fc_cls.weight"].resize_(num_classes,1024)
model_coco["state_dict"]["bbox_head.2.fc_cls.weight"].resize_(num_classes,1024)

# bias
model_coco["state_dict"]["bbox_head.0.fc_cls.bias"].resize_(num_classes)
model_coco["state_dict"]["bbox_head.1.fc_cls.bias"].resize_(num_classes)
model_coco["state_dict"]["bbox_head.2.fc_cls.bias"].resize_(num_classes)

#mask weight
#model_coco["state_dict"]["mask_head.0.conv_logits.weight"].resize_(num_classes,256,1,1)
#model_coco["state_dict"]["mask_head.1.conv_logits.weight"].resize_(num_classes,256,1,1)
#model_coco["state_dict"]["mask_head.2.conv_logits.weight"].resize_(num_classes,256,1,1)

#mask bias
#model_coco["state_dict"]["mask_head.0.conv_logits.bias"].resize_(num_classes)
#model_coco["state_dict"]["mask_head.1.conv_logits.bias"].resize_(num_classes)
#model_coco["state_dict"]["mask_head.2.conv_logits.bias"].resize_(num_classes)

#save new model
save_name = "_".join(model_name.split("_")[:-1]+["class_%s.pth"%num_classes])
print("Saving to %s"%save_name)
torch.save(model_coco, save_name)
