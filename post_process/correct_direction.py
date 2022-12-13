from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
from scipy import ndimage as nd

Image.MAX_IMAGE_PIXELS=10000000000000

path = "/data3/ian/dsmil-wsi/dsmil-wsi/test-c16/output"
store_path = "/data3/ian/dsmil-wsi/dsmil-wsi/test-c16/output"
files = os.listdir(path)
for f in files:
    file = os.path.join(path,f)
    img = Image.open(file)
    img = img.transpose(Image.ROTATE_270)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    store_file = os.path.join(store_path,f)
    img.save(store_file)
    
