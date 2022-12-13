from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
from scipy import ndimage as nd
import sys

Image.MAX_IMAGE_PIXELS=10000000000000

#path = "/data3/ian/dsmil-wsi/test-c16/binary_attention_mask60"
#store_path = "/data3/ian/dsmil-wsi/test-c16/binary_attention_mask60"
path = sys.argv[1]
store_path = path

files = os.listdir(path)
for f in files:
    file = os.path.join(path,f)
    img = Image.open(file).convert('L')
    img = np.array(img)
    binary = nd.morphology.binary_fill_holes(img)
    binary = Image.fromarray(binary)
    
    print(f)
    store_file = os.path.join(store_path,f)
    binary.save(store_file)
    
