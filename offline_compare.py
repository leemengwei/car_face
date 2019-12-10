import numpy as np
import os, sys
from IPython import embed
import glob






files=glob.glob("./car_dir/*")


for i in files:
    try:
        content = open(i+"/python_txt.txt", 'r').read().split("\n") 
        front = content[0].split(":")[1].strip('[').strip(']')
        front_and_back = content[1].split(":")[1].strip('[').strip(']')
        if len(front)!=len(front_and_back):
            print(i, front, 'vs', front_and_back)
            #print("cp %s /home/user/front_and_back_compare/ -r"%i.replace(" ","\ "))
            os.system("cp %s /home/user/front_and_back_compare/ -r"%i.replace(" ","\ "))
    except FileNotFoundError:
        print("Skip")


#embed()



