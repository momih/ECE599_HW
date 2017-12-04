import pandas as pd
from shutil import copy

bbox = pd.read_csv('../data/bbox.csv')
lung_shadow = ['Atelectasis', 'Infiltrate', 'Pneumonia', 'Mass', 'Nodule']
bbox = bbox[bbox['bbox'].isin(lung_shadow)]
image_path = '/home/momi/Documents/599/Final/Code/data/uncompressed/images/'
dest_path = '/home/momi/Documents/599/Final/Code/data/xrays/bound/full/'
bbox.apply(lambda y: copy(image_path + y['filename'], dest_path) , 1)

