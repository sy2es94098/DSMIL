from PIL import Image
import numpy as np
from  matplotlib import pyplot as plt
import os
from scipy import ndimage as nd
import openslide
import sys, argparse, os, copy, itertools, glob, datetime
Image.MAX_IMAGE_PIXELS=None

parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
parser.add_argument('--dir', default=None, type=str, help='Image directory')
parser.add_argument('--sample_dir', default=None, type=str, help='Reference directory')
parser.add_argument('--store_dir', default=None, type=str, help='Result store directory')
parser.add_argument('--level', default=0, type=int, help='WSI retrieve  level')
parser.add_argument('--ext', default=0, type=int, help='file extend')
args = parser.parse_args()

img_path = args.dir

files = os.listdir(img_path)
file_handler = None

if args.ext == 'tif' or args.ext == 'tiff':
    file_handler=Tiff_handler(args)
elif args.ext == 'png':
    file_handler=Png_handler(args)

for f in files:
    file_handler.get_item(f)
    file_handler.resize(f)
    file_handler.store_file(f)
