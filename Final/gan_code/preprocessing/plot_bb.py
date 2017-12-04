from tqdm import tqdm
from cv2 import imread, imwrite
import pandas as pd
from PIL import Image, ImageDraw
import cv2 

DATADIR = '/home/momi/Documents/599/Final/gan_code/data/xrays/bound/full/'
DEST = '/home/momi/Documents/599/Final/gan_code/data/xrays/bound/mod/'

df = pd.read_csv('../data/xrays/bound/shadow.csv')

bbox = df.drop_duplicates('filename')
for filename in tqdm(os.listdir(DATADIR)):
    img =  imread(DATADIR + filename,0)
    x,y,w,h = bbox[bbox['filename'] == filename].values[0][2:]
    
    cx, cy = (y+h/2, x+w/2)
    size = w if w>h else h
    new_x, new_y = (cy-size/2, cx-size/2)
    if new_x <0 or new_y <0 or new_x+size>1023 or new_y+size>1023:
        pass
    else:    
        # numpy
        up = [(y, var) for var in range(x, x+w)] 
        down = [(y+h, var) for var in range(x, x+w)]
        left = [(var, x) for var in range(y, y+h)] 
        right = [(var, x+w) for var in range(y, y+h)] 
        
        b_1 = up+down+left+right
        for pt in b_1:
            img[pt] = 0
        
        
        up = [(new_y, var) for var in range(new_x, new_x+size)] 
        down = [(new_y+size, var) for var in range(new_x, new_x+size)]
        left = [(var, new_x) for var in range(new_y, new_y+size)] 
        right = [(var, new_x+size) for var in range(new_y, new_y+size)] 
        
        b_1 = up+down+left+right
        for pt in b_1:
            img[pt] = 255
        
        imwrite(DEST+filename,img)

deform = img[y:y+h,x:x+w]


