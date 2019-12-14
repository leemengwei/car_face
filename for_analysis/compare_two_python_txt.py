import numpy as np
import sys,os
import glob
from IPython import embed


olds = glob.glob("/home/user/python_txts_with_old_pos/*")
news = glob.glob("/home/user/python_txts_with_new_pos/*")
olds.sort()
news.sort()

assert len(olds)==len(news)
count = 0
for i in range(len(olds)):
    if os.path.exists(news[i]+"/python_txt.txt") and os.path.exists(olds[i]+"/python_txt.txt"):
        print(i, 'okay')
        pass
    else:
        continue
    if open(news[i]+"/python_txt.txt",'r').read() != open(olds[i]+"/python_txt.txt",'r').read():
        print("Different:%s"%i, open(news[i]+"/python_txt.txt",'r').read(), open(olds[i]+"/python_txt.txt",'r').read())
        print("cp %s /home/user/differences_over_dir/new/ -r"%news[i].replace(" ", "\ "))
        print("cp %s /home/user/differences_over_dir/old/ -r"%olds[i].replace(" ", "\ "))
        os.system("cp %s /home/user/differences_over_dir/new/ -r"%news[i].replace(" ", "\ "))
        os.system("cp %s /home/user/differences_over_dir/old/ -r"%olds[i].replace(" ", "\ "))
        count += 1

print(count)


