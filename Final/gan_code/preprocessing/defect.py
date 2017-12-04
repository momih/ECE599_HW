import cv2, os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import random
import math

DATADIR = '/home/momi/Documents/599/Final/gan_code/data/xrays/bound/full/'
DEST = '/home/momi/Documents/599/Final/gan_code/data/xrays/bound/crop/'

bbox = pd.read_csv('../data/xrays/bound/shadow.csv')
filename = '00013118_008.png' #'00014716_007.png'
tqdm.pandas(desc="my bar!")  

# =============================================================================
# blur
# =============================================================================
img = cv2.imread(DATADIR+filename,0)
#blurred = cv2.GaussianBlur(original, (49,49), 0)
#
#blurred[y:y+h,x:x+w] = original[y:y+h,x:x+w]
#cv2.imwrite('cvBlurredOutput.jpg', blurred)


# =============================================================================
# crop
# =============================================================================

def get_square_box(coordinates):
    x, y, w, h = coordinates
    
    #Find center and size of box
    cy, cx = (y+h/2, x+w/2)
    size = w if w > h else h
    new_x = cx - (size/2)
    new_y = cy - (size/2)
    return (new_x, new_y, size, size)

def multiple_square_boxes(coordinates, increment=5, final_increment=200):
    """
    Takes an existing bounding box and returns squared bounding box 
    """
    x, y, w, h = coordinates
    
    #Find center and size of box
    cy, cx = (y+h/2, x+w/2)
    size = w if w > h else h

    # Create sequence of sizes to be produced
    final_size = size + final_increment
    size_seq = np.arange(size, final_size, increment)
        
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
            pass
        elif any(x > 1022 for x in new_box_right):
            # bounding box too close to right, squaring it makes it go out of bounds
            pass
        else:    
            new.append((new_x, new_y, given_size, given_size))           
    return new
   
def draw_bbox(pic, coordinates, bbox_c=None, ret=True):
    x,y,w,h = coordinates
    draw = ImageDraw.Draw(pic)
    draw.rectangle([(x, y), (x+w, y+h)], outline=225)
    
    if bbox_c is not None:
        x,y,w,h = bbox_c
        draw.rectangle([(x, y), (x+w, y+h)], outline=0)
    
    if ret == True:
        return np.array(pic)
    else:
        pic.show()

def draw_multiple_square_boxes(row): 
    x,y,w,h = row.values[2:]
    filename = row['filename']
    
    dest_name = DEST + 'BBOX_' + str(random.randint(1,5)) + '_' + filename
    #os.mkdir(dest_dir)
    
    # draw initial bbox in white
    im = Image.open(DATADIR + filename)
    img = draw_bbox(im, (x,y,w,h))
    
    # get new bboxes
    list_of_boxes = multiple_square_boxes((x,y,w,h))   
    
    # draw new bboxes 
    for item in list_of_boxes:
        new_x, new_y, size = item    
        up = [(new_y, var) for var in range(new_x, new_x+size)] 
        down = [(new_y+size, var) for var in range(new_x, new_x+size)]
        left = [(var, new_x) for var in range(new_y, new_y+size)] 
        right = [(var, new_x+size) for var in range(new_y, new_y+size)] 
        
        b_1 = up+down+left+right
        for pt in b_1:
            img[pt] = 0
            
    cv2.imwrite(dest_name, img)

def get_boxes_diff_inc(coordinates):
    squares = []
    #default
    t_list = multiple_square_boxes(coordinates)
    squares.extend(t_list)
    i = 10
    f = 25
    for _ in range(25):
        if len(t_list) > 1:
            t_list = multiple_square_boxes(squares[-1], increment=i, final_increment=100)
            squares.extend(t_list)
            i, f = i + 10, f + 10
        else:
            pass
    return list(set(squares))
    
def save_box(row):
    a,b,c,d = row.values[2:]
    filename = row['filename']
    bboxes = get_boxes_diff_inc((a,b,c,d))

    #open image
    xray = cv2.imread(DATADIR + filename)

    # extract boxes
    for box in bboxes:
        x, y, w, h = box
        deform = xray[y:y+h,x:x+w]
        deform = cv2.resize(deform, (32,32))
        write_name = DEST + filename[:-4] + '_' + str(h) + '_' + str(x) + '.png'
        cv2.imwrite(write_name, deform)
        

# =============================================================================
# blur
# =============================================================================


def apply_black_gradient(path_in, gradient=1., initial_opacity=1.):
    """
    Applies a black gradient to the image, going from left to right.

    Arguments:
    ---------
        path_in: string
            path to image to apply gradient to
        path_out: string (default 'out.png')
            path to save result to
        gradient: float (default 1.)
            gradient of the gradient; should be non-negative;
            if gradient = 0., the image is black;
            if gradient = 1., the gradient smoothly varies over the full width;
            if gradient > 1., the gradient terminates before the end of the width;
        initial_opacity: float (default 1.)
            scales the initial opacity of the gradient (i.e. on the far left of the image);
            should be between 0. and 1.; values between 0.9-1. give good results
    """

    # get image to operate on
    input_im = path_in
    width, height = input_im.size

    # create a gradient that
    # starts at full opacity * initial_value
    # decrements opacity by gradient * x / width
    imgsize = (250, 250) 
    innerColor = 0  #Color at the center
    outerColor = 255 #Color at the corners

    alpha_gradient = Image.new('L', imgsize, color=0xFF)
    for y in range(imgsize[1]):
        for x in range(width):
                  #Find the distance to the center
            distanceToCenter = math.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2)
    
            #Make it on a scale from 0 to 1
            distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * imgsize[0]/2)
    
            #Calculate r, g, and b values
            r = outerColor * distanceToCenter + innerColor * (1 - distanceToCenter)
            
            alpha_gradient.putpixel((x, y), int(r))
        # print '{}, {:.2f}, {}'.format(x, float(x) / width, a)
    alpha = alpha_gradient.resize(input_im.size)

    # create black image, apply gradient
    black_im = Image.new('L', (width, height), color=0) # i.e. black
    black_im.putalpha(alpha)

    # make composite with original image
    output_im = Image.alpha_composite(input_im, black_im)
    
    return output_im




imgsize = (250, 250) #The size of the image

image = Image.new('L', (250,250), color=0) #Create the image


for y in range(imgsize[1]):
    for x in range(imgsize[0]):

        #Find the distance to the center
        distanceToCenter = math.sqrt((x - imgsize[0]/2) ** 2 + (y - imgsize[1]/2) ** 2)

        #Make it on a scale from 0 to 1
        distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * imgsize[0]/2)

        #Calculate r, g, and b values
        r = outerColor * distanceToCenter + innerColor * (1 - distanceToCenter)
       

        #Place the pixel        
        image.putpixel((x, y), int(r))

