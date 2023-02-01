
import os
import random
import numpy as np
from skimage import io
from PIL import Image

root_dir = '/home/amax/repository/jc/isic/train'                # change it in your saved original data path
save_dir = '/home/amax/repository/jc/MySCI/dataset/fold5'


if __name__ == '__main__':
    imgfile = os.path.join(root_dir, 'images')
    labfile = os.path.join(root_dir, 'masks')
    f = sorted([os.path.join(imgfile, x) for x in os.listdir(imgfile) if x.endswith('.jpg')])
    random.shuffle(f)
    filename = []
    for file in f:
        temp = os.path.basename(file)
        filename.append(temp.split(".")[0])
    
    train = open(save_dir+"/train.list","w")
    val = open(save_dir+"/val.list","w")
    test =open(save_dir+"/test.list","w")
    for i in range(0,1816):
        train.write(filename[i]+"\n")
    for i in range(1816,2076):
        val.write(filename[i]+"\n")
    for i in range(2076,2594):
        test.write(filename[i]+"\n")

    
    