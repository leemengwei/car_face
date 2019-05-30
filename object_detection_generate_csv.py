import xml.etree.ElementTree as ET
from os import getcwd
import os
from tqdm import tqdm
classes = {1:"face"}

def convert():
    path = "/home/user/storage/zhuchaojie/facedetection/Annotations/"
    file_names = os.listdir(path)
    #file_names = [path+"MAG_train_02_0000209_B.xml"]
    with open("train.csv","w") as f:
        for file_name in tqdm(file_names):
            in_file = open(path + file_name)
            #in_file = open(file_name)
            tree = ET.parse(in_file)
            root = tree.getroot()
            if root.find("object"):
                
                for obj in root.iter("object"):
                    class_name = obj.find("name").text
                    xmlbox = obj.find("bndbox")
                    # print(xmlbox.find('xmin').text + " " + 
                    #         xmlbox.find('ymin').text + " " +
                    #         xmlbox.find('xmax').text + " " +
                    #         xmlbox.find('ymax').text + " " +
                    #         class_name + "\n")
                    f.write("/home/user/storage/zhuchaojie/facedetection/JPEGImages/"+file_name[:-3] + "jpg,")
                    f.write(xmlbox.find('xmin').text + "," + 
                            xmlbox.find('ymin').text + "," +
                            xmlbox.find('xmax').text + "," +
                            xmlbox.find('ymax').text + "," +
                            class_name+"\n")
                    
                #f.write("\n")
            else:
                # f.write("/data2/dockspace_zcj/magnetic/detection_data/train/"+file_name[:-3] + "jpg,,,,,")
                # f.write("\n")
                pass
     
convert()
