class DefaultConfigs(object):
    image_files = "/mfs/home/limengwei/car_face/car_face/object_detection_data_both_side_finetunes/train/*"    # your data file
    raw_data_format = "csv"            # "voc","coco","csv","txt","labelme"
    save_format = "csv"                # annotation saved format                 
    csv_anno_saved = "/mfs/home/limengwei/car_face/car_face/object_detection_data_both_side_finetunes/"    # your  data root
    image_format = "png"               # original image file format
    csv_annotations = "/mfs/home/limengwei/car_face/car_face/object_detection_data_both_side_finetunes/train.csv"               # original annotation file  for csv format
    image_saved_path = "/mfs/home/limengwei/car_face/car_face/object_detection_data_both_side_finetunes/train_aug/"              # image save path for augumented images
    #classes = ["angle,","angle_r","top","top_r","head"]               # all classes

config = DefaultConfigs()
