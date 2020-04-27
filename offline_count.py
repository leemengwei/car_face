import config
import os,sys,time
import numpy as np
from IPython import embed
import glob
import shutil

out_dir = '/'.join(sys.argv[1].split('/')[:-2])+"/wrongs/"
num_token = int(sys.argv[1].split('/')[-2])
txts = glob.glob(sys.argv[1]+"*/python_txt.txt")
how_many = []
for this_txt in txts:
    with open(this_txt, 'r') as f:
        content = f.readlines()
        num_of_people = len(content[1].strip('\n').split(":")[-1].strip("[]").split(", "))
        how_many.append(num_of_people)
        if num_of_people!= num_token:
            wrong_dir = '/'.join(this_txt.split('/')[:-1])
            des_dir = out_dir+'/'.join(wrong_dir.split('/')[-2:])
            print("Copying: %s to %s"%(wrong_dir, des_dir))
            try:
                shutil.copytree(wrong_dir, des_dir)
            except FileExistsError:
                pass
for i in range(1,6):
    print("Num:%s"%i, how_many.count(i), np.round(how_many.count(i)/len(how_many)*100,1),"%")
