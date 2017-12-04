import cv2, os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import random
from shutil import copy

DATADIR = '/home/momi/Documents/599/Final/gan_code/data/xrays/bound/full/'
DEST = '/home/momi/Documents/599/Final/gan_code/gan/data/masked/full/'

bbox = pd.read_csv('../data/xrays/bound/shadow.csv')
filename = '00013118_008.png' #'00014716_007.png'
tqdm.pandas(desc="my bar!")  

# =============================================================================
# normal - 10k train, 1k test, 1k val 
# abnormal - 6k train
# infect - 587 images, 10k masks tr, 1k masks test, 1k masks val
# =============================================================================
# %%
def multiple_square_boxes(coordinates, offset=15, final_offset=200):
    """
    Takes an existing bounding box and returns squared bounding box 
    """
    x, y, w, h = coordinates
    
    #Find center and size of box
    cy, cx = (y+h/2, x+w/2)
    size = max(w, h)

    # Create sequence of sizes to be produced
    final_size = size + final_offset
    size_seq = np.arange(size, final_size, offset)
        
    # Find new co-ordinates of top left
    new = []
    for given_size in size_seq:      
        new_x = cx - (given_size/2)
        new_y = cy - (given_size/2)
        
        # Checking out of image bound error
        new_box_left = [new_x, new_y]
        new_box_right = [new_x+given_size, new_y+given_size]
        if any(x < 0 for x in new_box_left):
            # bounding box too close to left, squaring it makes it go out of bounds
            # return non-square bbox increased by 50 pixels
            pass
        elif any(x > 1022 for x in new_box_right):
            # bounding box too close to right, squaring it makes it go out of bounds
            pass
        else:    
            new.append((new_x, new_y, given_size, given_size))           
    return new

def get_boxes_diff_inc(coordinates):
    squares = []
    #default
    t_list = multiple_square_boxes(coordinates)
    squares.extend(t_list)
    i = 20
    f = 400
    for _ in range(25):
        if len(squares) < 25 and len(t_list) > 1:
            t_list = multiple_square_boxes(squares[-1], offset=i, final_offset=f)
            squares.extend(t_list)
            i, f = i + 10, f
        else:
            pass
    return list(set(squares))

def create_mask_image(row):
    cords = row.values[2:]
    filename = row['filename'][3:12]
    bboxes = get_boxes(cords)
    bboxes.append(tuple(cords.tolist()))
    for item in bboxes:
        x,y,w,h = item
        mask = np.zeros((1024,1024), dtype='uint8')
        mask[y:y+h,x:x+w] = 255
        mask = cv2.resize(mask, (256, 256))
        save_file = DEST + filename + '_mask_' + str(w) + '_' + str(h) +'.jpg'
        if not os.path.exists(save_file):
            cv2.imwrite(save_file, mask)
        else:
            save_file = DEST + filename + '_mask1_' + str(w) + '_' + str(h) +'.jpg'
            cv2.imwrite(save_file, mask)

files = os.listdir('./')
train, test = train_test_split(files, test_size=11000)
for f in test:
    move(f, )            

