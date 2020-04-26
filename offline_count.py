import config
import os,sys,time
import numpy as np
from IPython import embed
import glob

txts = glob.glob(sys.argv[1]+"*/python_txt.txt")
how_many = []
for this_txt in txts:
    with open(this_txt, 'r') as f:
        content = f.readlines()
        num_of_people = len(content[1].strip('\n').split(":")[-1].strip("[]").split(", "))
        how_many.append(num_of_people)
for i in range(1,6):
    print("Num:%s"%i, how_many.count(i), np.round(how_many.count(i)/len(how_many)*100,1),"%")
