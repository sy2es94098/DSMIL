from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
import sys
Image.MAX_IMAGE_PIXELS=None

#path = "/data3/ian/dsmil-wsi/test-c16/instance_random_11092022_4"
path =  '/data3/ian/dsmil-wsi/test-c16/attention_mask'
#store_path = "/data3/ian/dsmil-wsi/test-c16/binary_attention_mask60"
store_path = sys.argv[1]
files = os.listdir(path)
print(sys.argv[2])
for f in files:
    file = os.path.join(path,f)
    print(file)
    img = Image.open(file).convert('L')
    img = np.array(img)
    num_arr = img[img>35]
    #print(np.percentile(img,99.5))
    print(np.unique(num_arr))
    p = np.percentile(img,float(sys.argv[2]))
    binary = np.where(img < p, 0, 255)
    binary = np.uint8(binary)
    binary = Image.fromarray(binary)
#    plt.imshow(binary, cmap='gray', vmin=0, vmax=255) 
#    plt.show()
    
    print(f)
    store_file = os.path.join(store_path,f)
    binary.save(store_file)
    
