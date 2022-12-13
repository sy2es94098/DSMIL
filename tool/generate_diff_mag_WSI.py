from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
from scipy import ndimage as nd
import openslide

img_path = "/data1/ian/C16_training_small/WSI/1/"
mask_path = "/data3/ian/dsmil-wsi/test-c16/rotate_attention"

store_path = "/data3/ian/dsmil-wsi/test-c16/small_attention"
Image.MAX_IMAGE_PIXELS=10000000000000

files = os.listdir(img_path)
for f in files:
    try:
        file = os.path.join(img_path,f)
        img = Image.open(file)
        w, h = img.size
                    
        mask_file = os.path.join(mask_path,f)
        mask = Image.open(mask_file)
        print(f)
        print(mask.size)
        mask = mask.resize((w, h))
        print(w,h)
        print(mask.size)
                                                    
        store_file = os.path.join(store_path,f)
        mask.save(store_file)
    except:
        print('Error',f)
