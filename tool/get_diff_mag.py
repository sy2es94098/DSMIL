from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
from scipy import ndimage as nd
import openslide

path = "/data1/ian/C16_test_mask/"
store_path = "/data1/ian/C16_training_index_3/C16_testing_index_3/mask"
files = os.listdir(path)

zoom_level = 3

for f in files:
    file = os.path.join(path,f)
    slide = openslide.open_slide(file)
    w, h = slide.level_dimensions[zoom_level]
    print(w,h)
    slide = slide.read_region((0, 0),zoom_level, (w, h))
    
    store_file = os.path.join(store_path,f[:-3]+'png')
    slide.save(store_file)
    print(store_file)
