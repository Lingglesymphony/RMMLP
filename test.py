import imp
import torch
import os
from glob import glob

with open('/home/amax/repository/jc/MySCI/dataset/all.list','r') as f:
    img_ids = f.readlines()
# file = open("dataset/test1.list",'w')
# for item in img_ids:
#     item = item.split(".")[0]
#     file.write(str(item))


print(len(img_ids))
