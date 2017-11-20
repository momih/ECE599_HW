from scipy.misc import imread, imsave
import pandas as pd

bbox = pd.read_csv('data/bbox.csv')

def save_bbox(s):
    img = imread('data/tar_files/images/' + s.loc['filename'])
    x,y,w,h = s.values[2:]
    deform = img[int(y):int(y+h), int(x):int(x+w)]
    name_ = 'data/xrays/bbox/' + s.loc['bbox'] + '_' + s.loc['filename']
    imsave(name_, deform)
    
bbox.apply(lambda y: save_bbox(y), 1)

