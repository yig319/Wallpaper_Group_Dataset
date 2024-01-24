import os, glob, random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.ndimage import rotate # anti-clockwise
from PIL import Image # anti-clockwise
from manage_data import verify_image_vector 

def process_unit_cell(result, ts, VA, VB, out_size):
    unit_cell = np.copy(result)
    polygon = Polygon([ts[::-1], (ts+VA)[::-1], (ts+VA+VB)[::-1], (ts+VB)[::-1]])
    for i in range(unit_cell.shape[0]):
        for j in range(unit_cell.shape[1]):
            if not polygon.contains(Point(j,i)):
                unit_cell[i,j] = 0
    return unit_cell

def rotate_xy(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    oy, ox = origin
    py, px = point
    
    theta = np.radians(angle)
    qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
    qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
    return int(qy), int(qx)

def blank_detection(region):
    return np.sum(region) == np.mean(region)

def remove_black_spot(img):
    region = np.copy(img).astype(np.int32)
    h, w = region.shape[:2]
    for k in range(region.shape[2]):
        if region[0,0,k] == 0:  region[0,0,k] = (region[1,0,k] + region[0,1,k])//2
        if region[h-1,0,k] == 0:  region[h-1,0,k] = (region[h-2,0,k] + region[h-1,1,k])//2
        if region[0,w-1,k] == 0:  region[0,w-1,k] = (region[0,w-2,k] + region[1,w-1,k])//2
        if region[h-1,w-1,k] == 0:  region[h-1,w-1,k] = (region[h-1,w-2,k] + region[h-2,w-1,k])//2

    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            for k in range(region.shape[2]):
                l = len(region[i-1:i+2, j-1:j+2, k].reshape(-1))
                avg = np.abs((np.sum(region[i-1:i+2, j-1:j+2, k])-region[i,j,k])/(l-1))
                if region[i,j,k] == 0:
                    region[i,j,k] = avg
    return region


def crop_area(source_image, shape, symmetry):
    
    in_size = source_image.shape
    area = int(round(np.random.normal(loc=960, scale=10))) # normal distribution near 960 in area
    if shape == 'square':
        angle = 90
        unit_h = int(round(np.sqrt(area)))
        unit_w = unit_h
        
        va = np.array((0, unit_w))
        vb = np.array((unit_h, 0))
        
    if shape == 'rectangular':
        angle = 90
        unit_w = int(round(np.random.normal(loc=31, scale=3)))
        unit_h = int(round(area/unit_w))
        
        va = np.array((0, unit_w))
        vb = np.array((unit_h, 0))
        
    if shape == 'oblique':
        
        angle = np.random.randint(45,85)
        unit_w = int(round(np.random.normal(loc=31, scale=3)))
        unit_h = int(round(area/unit_w))
        extra_w = int(round(np.abs(unit_h/np.tan(angle))))

        va = np.array((0, unit_w))
        # assume vb[0] is negative
        vb = np.array((unit_h, -extra_w))
        unit_w = unit_w + extra_w
        
        while 2 * np.abs(vb[1]) >= unit_w: # make sure oblique shape can be seperated to 3 parts
            angle = np.random.randint(45,85)

            unit_w = int(round(np.random.normal(loc=50, scale=3)))
            unit_h = int(round(area/unit_w))
            extra_w = int(round(np.abs(unit_h/np.tan(angle))))

            va = np.array((0, unit_w))
            # assume vb[0] is negative
            vb = np.array((unit_h, -extra_w))
            unit_w = unit_w + extra_w


    if shape == 'hexagonal':
        angle = 60
        area = area*2/3
        unit_h = int(np.sqrt(area/2*np.sqrt(3)))
        unit_w = int(round(3/2*unit_h*2/np.sqrt(3)))
        
        va = np.array((0, int(round(unit_w*2/3))))
        # assume vb[0] is negative
        vb = np.array((unit_h, -int(round(unit_h/np.tan(np.deg2rad(angle))))))
        
        
    if shape == 'rhombic':
        unit_w = int(round(np.random.normal(loc=31, scale=3)))
        unit_h = int(round(area/unit_w))
        # area of rhombic is half of the rectangular or oblique
        unit_h, unit_w = int(round(np.sqrt(2)*unit_h)), int(round(np.sqrt(2)*unit_w))
        
        # make sure even number for translation operation
        if unit_h % 2 == 1: unit_h += 1
        if unit_w % 2 == 1: unit_w += 1

        va = np.array((-int(round(unit_h/2)), int(round(unit_w/2))))
        vb = np.array((-va[0], va[1]))  
        
    if shape == 'triangle':
        area = area*2
        
        if symmetry in ['p4m', 'p4g']:
            unit_h = int(round(np.sqrt(area)))
            unit_w = unit_h
            
            va = np.array((-unit_h, unit_w))
            vb = np.array((0, unit_w))
            
        elif symmetry in ['p3']:
            angle = 60
            unit_w = int(np.sqrt(area/2*np.sqrt(3)))
            unit_h = int(round(3/2*unit_w*2/np.sqrt(3)))
            
            while unit_h % 3 != 0: unit_h+=1

            va = np.array((int(1/3*unit_h), unit_w))
            vb = np.array((int(2/3*unit_h), 0))
            
        elif symmetry in ['p3m1']:            
            # vertical triangle as primitive unit cell
            unit_w = int(np.sqrt(area * 2 // np.sqrt(3))) 
            while unit_w % 6 != 0: unit_w+=1
            unit_h = int(unit_w * np.sqrt(3) // 2)

            # horizontal triangles and vectors - reversed w,h position
            va = np.array((-int(1/2*unit_w), unit_h))
            vb = np.array((int(1/2*unit_w), unit_h))   
            
        elif symmetry in ['p31m', 'p6']:            
            unit_h = int(np.sqrt(area * 2 / np.sqrt(3))//2) 
            unit_w = int(round(unit_h * 2 * np.sqrt(3)))

            # horizontal triangles and vectors
            va = np.array((-unit_h, int(round(unit_w/2))))
            vb = np.array((0, unit_w))  
            
            
        elif symmetry in ['p6m']:            
            unit_h = int(np.sqrt(area / np.sqrt(3))) 
            unit_w = int(round(unit_h * np.sqrt(3)))
            while unit_w % 4 != 0: 
                unit_w+=1
                unit_h = int(round(unit_w / np.sqrt(3)))

            # horizontal triangles and vectors
            va = np.array((-unit_h, unit_w))
            vb = np.array((0, unit_w)) 
            
        else:
            unit_w = int(round(np.random.normal(loc=44, scale=3)))
            unit_h = int(round(area/unit_w))
            # area of rhombic is half of the rectangular or oblique
            unit_h, unit_w = int(round(np.sqrt(2)*unit_h)), int(round(np.sqrt(2)*unit_w))
        
            if (symmetry in ['p2']) and (unit_w % 2 == 1): unit_w+=1
        
            va = np.array((-unit_h, int(round(unit_w/2))))
            vb = np.array((0, unit_w))

            if symmetry in ['cmm']:
                va = np.array((-unit_h, unit_w))
                vb = np.array((0, unit_w))
                

    
    if 0>=(in_size[0]-unit_h) or 0>=(in_size[1]-unit_w):
        print(symmetry, shape, in_size, unit_h, unit_w)
        print(0, in_size[0]-unit_h, 0, in_size[1]-unit_w)
        source_start =  np.array((0,0)) 
        translate_start = np.array(( np.random.randint(0, unit_h), np.random.randint(0, unit_w) ))
        crop_region = np.zeros((unit_h, unit_w)).astype(np.uint8)

    else:
        source_start = np.array(( np.random.randint(0, in_size[0]-unit_h), np.random.randint(0, in_size[1]-unit_w)) ) # crop starting point
        translate_start = np.array(( np.random.randint(0, unit_h), np.random.randint(0, unit_w) ))
        crop_region = vector_crop(source_image, source_start, unit_h, unit_w)
        # print(crop_region.dtype)

    return crop_region, source_start, translate_start, va, vb



def vector_crop(source_image, ss, unit_h, unit_w):

    # convert to x, y coordinates from h, w
    ss_ = np.array((ss[1], ss[0]))
    a_ = np.array((ss[1], ss[0] + unit_h))
    b_ = np.array((ss[1] + unit_w, ss[0]))
    c_ = np.array((ss[1] + unit_w, ss[0] + unit_h))

    mask = np.zeros(source_image.shape, dtype=np.uint8)
    roi_corners = np.array([[ss_, a_, c_, b_]], dtype=np.int32)

    # compromise for triangle
    if unit_h < 0 and unit_w == 0:
        ss_ = ss_ + np.array((0,-1))
        roi_corners = np.array([[ss_, a_, b_]], dtype=np.int32)
    
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = source_image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    # from Masterfool: use cv2.fillConvexPoly if you know it's convex
    
    masked_image = cv2.bitwise_and(source_image, mask)
    
    # crop rectangular region with x, y coordinates
    x_max, x_min = np.max((ss_[0], a_[0], b_[0], c_[0])), np.min((ss_[0], a_[0], b_[0], c_[0]))
    y_max, y_min = np.max((ss_[1], a_[1], b_[1], c_[1])), np.min((ss_[1], a_[1], b_[1], c_[1]))
    
    # compromise for triangle
    if unit_h and unit_w == 0:
        x_max, x_min = np.max((ss_[0], a_[0], b_[0])), np.min((ss_[0], a_[0], b_[0]))
        y_max, y_min = np.max((ss_[1], a_[1], b_[1])), np.min((ss_[1], a_[1], b_[1]))
    
    return np.roll(masked_image[y_min:y_max, x_min:x_max], np.abs(unit_h), axis=0)




def set_parameters(symmetry, shape, ss, ts, va, vb):
    '''
    va, vb: primitive unit cell vectors
    VA, VB: translation unit cell vectors
    VA_, VB_: translation unit cell vectors to fulfill the requirement in translation operation
    '''
    if symmetry =='p1':
        VA, VB = va, vb
        VA_, VB_ = va, vb
        
    if symmetry =='p2' :
        if shape == 'triangle':
            VA, VB = va, va*np.array((-1,1))
            VA_, VB_ = VA+VB, VB*np.array((2,0))
        else:
            VA, VB = va, 2*vb
            VA_, VB_ = VA, VB

    if symmetry in ['pm', 'pg']:
        VA, VB = va, 2*vb
        VA_, VB_ = va, 2*vb
        
    if symmetry in ['cm', 'pmm', 'pmg', 'pgg', 'p4']:
        VA, VB = 2*va, 2*vb
        VA_, VB_ = 2*va, 2*vb

    if symmetry == 'cmm':
        VA, VB = va, va*np.array((-1,1))
        VA_, VB_ = VA+VB, VB*np.array((2,0))

    if symmetry == 'p4m':
        VA, VB = va+va*np.array((-1,1)), va*np.array((-1,1))*np.array((2,0))
        VA_, VB_ = VA, VB

    if symmetry == 'p4g':
        ts = ts + vb
        VA, VB = va+va*np.array((-1,1)), va*np.array((-1,1))*np.array((2,0))
        VA_, VB_ = VA, VB
        
    if symmetry == 'p3':
        VA, VB = va*np.array((0,2)), va*np.array((3,-1))
        VA_, VB_ = VA, VB

    if symmetry == 'p3m1':
        VA, VB = (va*np.array((0,2))).astype(np.int32), vb*np.array((3,-1))
        VA_, VB_ = VA, VB
    
    if symmetry in ['p31m', 'p6']:
        vb_ = np.copy(vb)
        VA, VB = np.copy(vb), va*np.array((-3, -1))
        VA_, VB_ = VA, VB
        
    if symmetry in ['p6m']:
        vb_ = np.copy(vb)
        VA, VB = vb*2, va*np.array((-3, -1))
        VA_, VB_ = VA, VB

    return [ss, ts, va, vb, VA, VB, VA_, VB_, shape]




def vector_translate(rec_crop, output_size, start_point, VA, VB):
    
    if VA[0] < 0:
        VA = VA + VB
        VB = (2*VB[0], 0)
    
    crop_h, crop_w = rec_crop.shape[:2]
    image = np.zeros((output_size[0]+VB[0]*4, output_size[1]+(VA[1])*4, output_size[2])).astype(np.uint8)
    
    roll = np.copy(image)
    roll[start_point[0]:rec_crop.shape[0]+start_point[0], start_point[1]:rec_crop.shape[1]+start_point[1]] = rec_crop
    compensate_times = 0
    
    for i in range(int(np.floor(image.shape[0]/VB[0]))):
        left_side = np.abs(VB[1]*i) - VA[1]*compensate_times
        while left_side > VA[1]:
            roll = np.roll(roll, shift=VA[1], axis=1)
            left_side -= VA[1]
            compensate_times += 1
        
        for j in range(int(np.floor(image.shape[1]/VA[1]))):
            image += np.roll(np.roll(roll, shift=(VA[1])*j+VB[1]*i, axis=1), shift=VB[0]*i, axis=0) 

    start_x = VA[1] + VB[1]
    while start_x < start_point[1]:
        start_x += VA[1]
    return image[VB[0]:VB[0]+output_size[0], start_x:start_x+output_size[1]]


def center_ts(ts, VA, VB, out_size, shape, symmetry):
    
    # move the translation start point close to center
    debug_show(image=None, title=f"Before move:{ts}, VA:{VA}, VB:{VB}, shape:{shape}")

    if shape in ['rectangular', 'square'] or symmetry in ['p4m', 'p4g']:
        while out_size[1]//2 - ts[1] > VA[1]:
            ts += VA
        while out_size[0]//2 - ts[0] > VB[0]:
            ts += VB

    elif shape in ['oblique', 'hexagonal'] or symmetry in ['p3', 'p3m1', 'p31m', 'p6', 'p6m']:
        while out_size[0]//2 - ts[0] > VB[0]:
            ts += VB
        while out_size[1]//2 - ts[1] > VA[1]:
            ts += VA

    elif shape in ['triangle']:
        while out_size[0]//2 - ts[0] > VB[0]:
            ts -= VA
        while out_size[1]//2 - ts[1] > VA[1]:
            ts += VB 
        while ts[1] - out_size[1]//2 > VA[1]:
            ts -= VB 

    elif shape in ['rhombic']:
        while out_size[0]//2 - ts[0] > VB[0]:
            ts = ts - VA + VB
        while out_size[1]//2 - ts[1] > VA[1]:
            ts = ts + VA + VB      
            
    debug_show(image=None, title=f"After move:{ts}, VA:{VA}, VB:{VB}")
    return ts
                


def re_center_ts(ts, VA, VB, out_size):
    
#     move start point inside the image

    debug_show(image=None, title=f"outsize:{out_size}, initial - start point:{ts}, VA:{VA}, VB:{VB}")
    
    ts = np.array(ts)
    VA = np.array(VA)
    VB = np.array(VB)
    
    rep_times = 0
    new_dist, old_dist = 1, 0
    center = np.array(out_size)[:2]//3
    
    while rep_times < 4:
        rep_times = 0
        old_dist = np.sqrt( (ts[1] - center[1])**2 + (ts[0] - center[0])**2 )
        new_dist = np.sqrt( ((ts+VA)[1] - center[1])**2 + ((ts+VA)[0] - center[0])**2 )
        if new_dist < old_dist: 
            ts += VA
            debug_show(image=None, title=f"start point:{ts}, VA:{VA}")
        else: rep_times += 1
            
        old_dist = np.sqrt( (ts[1] - center[1])**2 + (ts[0] - center[0])**2 )
        new_dist = np.sqrt( ((ts-VA)[1] - center[1])**2 + ((ts-VA)[0] - center[0])**2 )
        if new_dist < old_dist: 
            ts -= VA
            debug_show(image=None, title=f"start point:{ts}, -VA:{-VA}")
        else: rep_times += 1    
            
        old_dist = np.sqrt( (ts[1] - center[1])**2 + (ts[0] - center[0])**2 )
        new_dist = np.sqrt( ((ts+VB)[1] - center[1])**2 + ((ts+VB)[0] - center[0])**2 )
        if new_dist < old_dist: 
            ts += VB
            debug_show(image=None, title=f"start point:{ts}, VB:{VB}")
        else: rep_times += 1       
            
        old_dist = np.sqrt( (ts[1] - center[1])**2 + (ts[0] - center[0])**2 )
        new_dist = np.sqrt( ((ts-VB)[1] - center[1])**2 + ((ts-VB)[0] - center[0])**2 )
        if new_dist < old_dist: 
            ts -= VB
            debug_show(image=None, title=f"start point:{ts}, -VB:{-VB}")
        else: rep_times += 1        

    debug_show(image=None, title=f"final start point:{ts}")
    return ts
    
    
    
    
    
    
                
def rotate_xy(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    oy, ox = origin
    py, px = point
    
    theta = np.radians(-angle)
    qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
    qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)
    return int(qy), int(qx)



def p1_unit(region, va, vb, shape='rectangular'):
    
    if shape in ['rectangular', 'square']:
        unit = np.copy(region)
    
    if shape in ['oblique', 'hexagonal']:
        p1 = np.copy(region)[:,:np.abs(vb[1])]
        p2 = np.copy(region)[:,np.abs(vb[1]):region.shape[1]-np.abs(vb[1])]
        p3 = np.copy(region)[:,-np.abs(vb[1]):]

        k = np.abs(vb[0]/vb[1])
        b = vb[0]
        for i in range(p1.shape[1]):
            for j in range(p1.shape[0]):
                if k*i + j < b:
                    p1[j,i] = p3[j,i]
        unit = np.concatenate((p2, p1), axis=1)
    
    if shape in ['rhombic']:
        unit = np.copy(region)
        # triangle primitive unit cell
        crop_region_roll = np.roll(np.roll(region, shift=region.shape[0]//2, axis=0), shift=region.shape[1]//2, axis=1)
        
        polygon = Polygon([(vb[0],0), (0,va[1]), (vb[0],2*va[1]), (2*vb[0],va[1])])
        for i in range(region.shape[1]):
            for j in range(region.shape[0]):
                if not polygon.contains(Point(j,i)):
                    unit[j,i] = crop_region_roll[j,i]
        unit = np.roll(unit, shift=vb[0], axis=0)
    
    return unit


# 180 degree rotation
def p2_unit(region, va, vb, shape='rectangular'):
    
    if shape in ['rectangular', 'square']:
        unit = np.vstack((region, np.rot90(region, 2)))       
    
    if shape in ['oblique', 'hexagonal']:
        p1 = np.copy(region)[:,:np.abs(vb[1])]
        p2 = np.copy(region)[:,np.abs(vb[1]):region.shape[1]-np.abs(vb[1])]
        p3 = np.copy(region)[:,-np.abs(vb[1]):]

        k = np.abs(vb[0]/vb[1])
        b = vb[0]
        for i in range(p1.shape[1]):
            for j in range(p1.shape[0]):
                if k*i + j < b:
                    p1[j,i] = p3[j,i]
        top_part = np.hstack((p1, p2))

        unit_ = np.vstack((top_part, np.rot90(top_part, 2)))
        unit = np.hstack((unit_, unit_))
        
        unit = unit[:, -vb[1]:-vb[1]+va[1]]
         
    if shape in ['triangle']:
        unit = np.vstack((region, np.rot90(region,2)))
        debug_show(unit, "Translational unit cell (step 1)")
        
        # triangle primitive unit cell
        unit_roll = np.roll(np.roll(unit, shift=unit.shape[0]//2, axis=0), shift=unit.shape[1]//2, axis=1)
        
        h, w = unit.shape[:2]
        polygon = Polygon([(h//2,0), (h,w//2), (h//2,w), [0,w//2]])
        for i in range(unit.shape[1]):
            for j in range(unit.shape[0]):
                if not polygon.contains(Point(j,i)):
                    unit[j,i] = unit_roll[j,i]
        unit = np.roll(unit, shift=h//2, axis=0)
    
    return unit


# mirror
def pm_unit(crop_region):
    # only rectangular and square are allowed
    unit = np.vstack((crop_region, cv2.flip(crop_region,0,dst=None))).astype(np.int32)
    return unit

# glide
def pg_unit(crop_region):
    # only rectangular and square are allowed
    unit = np.vstack((crop_region, cv2.flip(np.rot90(crop_region,2),0,dst=None))).astype(np.int32)
    return unit

# # 180 rotation
# def p2_unit(crop_region):
#     if vb[1] < 0: # oblique and hexagonal: pad extra space for 180 rotation
#         crop_region = np.hstack(( np.zeros([vb[0], np.abs(vb[1]), 3]), crop_region ))
#     # compromise for triangle
#     if va[0] < 0 and vb[0] == 0:
#         unit = np.vstack((np.rot90(crop_region, 2), np.roll(crop_region, -1, axis=1))).astype(np.int32)
#     else:
#         unit = np.vstack((np.roll(crop_region, -1, axis=1), np.rot90(crop_region, 2))).astype(np.int32)
#     return unit

# mirror + glide
def cm_unit(crop_region):
    # only rectangular and square are allowed
    unit = np.vstack((crop_region, cv2.flip(crop_region,0,dst=None))).astype(np.int32)
    unit = np.hstack((unit, np.roll(unit,shift=crop_region.shape[0],axis=0))).astype(np.int32)
    return unit

# 2 perpendicular directon mirror axises
def pmm_unit(crop_region):
    # only rectangular and square are allowed
    unit = np.vstack((crop_region, cv2.flip(crop_region,0,dst=None))).astype(np.int32)
    unit = np.hstack((unit, cv2.flip(unit,1,dst=None))).astype(np.int32)
    return unit

# mirror + 180 rotation
def pmg_unit(crop_region):
    # only rectangular and square are allowed
    unit = np.hstack((crop_region, np.rot90(crop_region,2))).astype(np.int32)
    unit = np.vstack((unit, cv2.flip(unit,0,dst=None))).astype(np.int32)
    return unit

# 180 rotation + glide
def pgg_unit(crop_region):
    # only rectangular and square are allowed
    unit = np.vstack((crop_region, np.rot90(crop_region,2))).astype(np.int32)
    unit = np.hstack((unit, np.roll(cv2.flip(unit,0,dst=None), shift=crop_region.shape[0], axis=0))).astype(np.int32)
    return unit

# 90 rotation
def p4_unit(crop_region):
    # only rectangular and square are allowed
    unit = np.hstack((crop_region, np.rot90(crop_region))).astype(np.int32)
    unit = np.vstack((unit, np.rot90(unit,2))).astype(np.int32)
    return unit

# 2 perpendicular directon mirror axises + 180 rotation (not on mirror axis)
def cmm_unit(crop_region):

    # triangle primitive unit cell
    crop_region_180 = np.rot90(crop_region, 2)
    
    k = crop_region.shape[0]/crop_region.shape[1]
    b = crop_region.shape[0]
    for i in range(crop_region.shape[1]):
        for j in range(crop_region.shape[0]):
            if k*i + j < b:
                crop_region[j,i] = crop_region_180[j,i]
    debug_show(crop_region, "Translational unit cell (step 1)")
    
    unit = np.vstack((crop_region, cv2.flip(crop_region,0,dst=None))).astype(np.int32)
    debug_show(unit, "Translational unit cell (step 2)")
    unit = np.hstack((cv2.flip(unit,1,dst=None), unit)).astype(np.int32)
    return unit


#  two rotation centres of order four (90°), 
# and reflections in four distinct directions (horizontal, vertical, and diagonals)
def p4m_unit(crop_region):

    # triangle primitive unit cell
    crop_region_flip = cv2.flip(np.rot90(crop_region, 1), 1, dst=None)
    
    b = crop_region.shape[0]
    for i in range(crop_region.shape[1]):
        for j in range(crop_region.shape[0]):
            if i + j < b:
                crop_region[j,i] = crop_region_flip[j,i]
    debug_show(crop_region, "Translational unit cell (step 1)")

    unit = np.vstack((cv2.flip(crop_region,0,dst=None), crop_region)).astype(np.int32)
    debug_show(unit, "Translational unit cell (step 2)")

    unit = np.hstack((unit, cv2.flip(unit,1,dst=None))).astype(np.int32)
    return unit

#  two rotation centres of order four (90°), 
# and reflections in four distinct directions (horizontal, vertical, and diagonals)

def p4g_unit(crop_region):

    # triangle primitive unit cell
    crop_region_45_flip = cv2.flip(np.rot90(crop_region, 1), 1, dst=None)
    
    k = crop_region.shape[0]/crop_region.shape[1]
    b = crop_region.shape[0]
    for i in range(crop_region.shape[1]):
        for j in range(crop_region.shape[0]):
            if k*i + j < b:
                crop_region[j,i] = crop_region_45_flip[j,i]
    debug_show(crop_region, "Translational unit cell (step 1)")

    unit = np.vstack((np.rot90(crop_region,1), crop_region)).astype(np.int32)
    debug_show(unit, "Translational unit cell (step 2)")
    
    unit = np.hstack((np.rot90(unit,2), unit)).astype(np.int32)
    return unit


def p3_unit(region):
    
    p2 = np.hstack((np.zeros(region.shape), region, np.zeros(region.shape))).astype(np.uint8)
    h, w = p2.shape[:2]

    p2_im = Image.fromarray(np.uint8(p2)) # have to transfer with uint8 type
    
    p1_im = p2_im.rotate(120, center=(w//3, h*2//3))
    p5_im = p2_im.rotate(-120, center=(w//3, h*2//3))
    
    p3_im = p2_im.rotate(120, center=(w*2//3, h//3))
    p4_im = p2_im.rotate(-120, center=(w*2//3, h//3))

    p1, p3 = np.array(p1_im), np.array(p3_im)
    p4, p5 = np.array(p4_im), np.array(p5_im)
    
    unit = np.copy(p2)
    polygon_p1 = Polygon([(-1,-1), (-1,w//3), (h*2//3,w//3), (h+1,-1)])
    for i in range(unit.shape[1]):
        for j in range(unit.shape[0]):
            if polygon_p1.contains(Point(j,i)):
                unit[j,i] = p1[j,i]  

    polygon_p3 = Polygon([(-1,w+1), (h//3,w*2//3-1), (h+1,w*2//3-1), (h+1,w+1)])
    for i in range(unit.shape[1]):
        for j in range(unit.shape[0]):
            if polygon_p3.contains(Point(j,i)):
                unit[j,i] = p3[j,i]
 
    polygon_p4 = Polygon([(-1,w//3), (h//3,w*2//3), (-1,w+1)])
    for i in range(unit.shape[1]):
        for j in range(unit.shape[0]):            
            if polygon_p4.contains(Point(j,i)):
                unit[j,i] = p4[j,i]
        
    polygon_p5 = Polygon([(h-1,-1), (h*2//3-1,w//3), (h+1,w*2//3)])
    for i in range(unit.shape[1]):
        for j in range(unit.shape[0]):            
            if polygon_p5.contains(Point(j,i)):
                unit[j,i] = p5[j,i]
        
    p1 = np.copy(unit)[:, :w//3]
    p2 = np.copy(unit)[:, w//3:2*w//3]
    p3 = np.copy(unit)[:, -w//3:]

    k = np.abs(h/(w//3))
    b = h
    for i in range(w):
        for j in range(h):
            if k*i + j < b:
                p1[j,i] = p3[j,i]
    unit = np.concatenate((p2, p1), axis=1)
    
    # approximation
    unit = remove_black_spot(unit)

    return unit


def p3m1_unit(region):    
    
    # construct the hexagonal region(pri-unit_cell in p3) first
        
    region_top = np.rot90(region, -1)
    region_bottom = cv2.flip(region, 1, dst=0)

        
    h, w = region_bottom.shape[:2]
    region_bottom_im = Image.fromarray(np.uint8(region_bottom)) # have to transfer with uint8 type
    region_bottom_im = region_bottom_im.rotate(-30, center=(w//2, h*2//3), expand=True)
    region_bottom = np.array(region_bottom_im)

    # crop the bottom w part in height
    region_bottom = region_bottom[-w:]
        
    while np.mean(region_bottom[:,0]) == np.sum(region_bottom[:,0]):
        region_bottom = region_bottom[:,1:]

    region_bottom = region_bottom[:,:h]

    h, w = region_top.shape[:2]
    region_top = np.concatenate((region_top, np.zeros((h//2, w, 3), dtype=np.int32)), 
                                axis=0) 
    
    region_bottom = np.concatenate((np.zeros((h//2, w, 3), dtype=np.int32), region_bottom), 
                                axis=0) 
        
    region = np.copy(region_top)
    polygon_top = Polygon([(h,0), (h//2,w), (int(h*3//2),w), (int(h*3//2),0)])
    for i in range(region_top.shape[1]):
        for j in range(region_top.shape[0]):
            if polygon_top.contains(Point(j,i)):
                region[j,i] = region_bottom[j,i]  
    
    unit = p3_unit(region)
    return unit


def p31m_unit(region):    
    # construct the hexagonal region(pri-unit_cell in p3) first
    region_flatten = np.concatenate((region, cv2.flip(region, 0, dst=0)))
    h, w = region_flatten.shape[:2]
    
    # rotate to right orientation
    region = rotate(region_flatten, angle = -60, reshape=True)
    h_f = int(round(1.5 * region_flatten.shape[0]))
    
    # crop 
    region = region[(region.shape[0]-h_f)//2: (region.shape[0]-h_f)//2 + h_f,
                    round(region.shape[1]/4):round(3*region.shape[1]/4)]

    # then follow the same step as p3
    unit = p3_unit(region)
    return unit

def p6_unit(region):    
    # construct the hexagonal region(pri-unit_cell in p3) first
    region_flatten = np.concatenate((region, np.rot90(region, 2)))
    h, w = region_flatten.shape[:2]
    
    region = rotate(region_flatten, angle = -60, reshape=True)
    h_f = int(round(1.5 * region_flatten.shape[0]))
    region = region[(region.shape[0]-h_f)//2: (region.shape[0]-h_f)//2 + h_f,
                    round(region.shape[1]/4):round(3*region.shape[1]/4)]

    # then follow the same step as p3
    unit = p3_unit(region)
    return unit

def p6m_unit(region):    
    # construct the hexagonal region(pri-unit_cell in p3) first
    region_top = np.concatenate((region, cv2.flip(region, 1, dst=0)), axis=1)
    region_flatten = np.concatenate((region_top, np.rot90(region_top, 2)))
    
    h, w = region_flatten.shape[:2]
    
    region = rotate(region_flatten, angle = -60, reshape=True)
    h_f = int(round(1.5 * region_flatten.shape[0]))
    region = region[(region.shape[0]-h_f)//2: (region.shape[0]-h_f)//2 + h_f,
                     :region_flatten.shape[1]]
    
    region = region[:, round(region.shape[1]/4):round(3*region.shape[1]/4)]
    debug_show(region, "Translational unit cell (step 1)")
    
    # then follow the same step as p3
    unit = p3_unit(region)
    return unit



def transformation(source_image, out_size, symmetry, shape='random', rotation=False, file=None):
    
    step_outputs = {}
    
    if source_image.dtype == np.float32 or source_image.dtype == np.float64:
        source_image = (source_image*255).astype(np.uint8)
    step_outputs['source_image'] = source_image

    if symmetry =='p1':
        shape_list = ['rectangular', 'square', 'rhombic', 'oblique', 'hexagonal']

    if symmetry =='p2' :
        shape_list = ['rectangular', 'square', 'triangle', 'oblique', 'hexagonal']

    if symmetry in ['pm', 'pg', 'cm', 'pmm', 'pmg', 'pgg']:
        shape_list = ['rectangular', 'square']

    if symmetry == 'p4': 
        shape_list = ['square']

    if symmetry == 'cmm' or symmetry == 'p4m' or symmetry == 'p4g':
        shape_list = ['triangle']
        
    if symmetry in ['p3', 'p3m1', 'p31m', 'p6', 'p6m']:
        shape_list = ['triangle']
    
    if shape == 'random':
        shape = random.choice(shape_list)

    if shape not in shape_list: 
        raise ValueError(shape, 'is not allowed for ', symmetry, 'symmetry.')            
        
    
    # crop primitive unit cell

    ### crop rectangular region first
    region, ss, ts, va, vb = crop_area(source_image, shape, symmetry)

    ##  detect the blank region:
    attempt = 0
    while blank_detection(region):
        print(f'Attempt {attempt}: detected blank image')
        region, ss, ts, va, vb = crop_area(source_image, shape, symmetry)
        attempt+=1
        if attempt>5:
            raise ValueError(f'Cannot find a valid region to crop, source image maybe blank or has a incompatible shape: \n image size={source_image.shape}, shape of crop region={region.shape}, shape is {shape}, symmetry is {symmetry}, file is {file}')
    step_outputs['rectangular'] = source_image
        
    
    ### define all the parameters then
    [ss, ts, va, vb, VA, VB, VA_, VB_, shape] = set_parameters(symmetry, shape, ss, ts, va, vb)
    debug_show(source_image, "Source image", vectors=[ss, va, vb])
    debug_show(region, "Primitive unit cell")
    step_outputs['primitive_unit_cell'] = region
    
    # construct translation unit cell
    if symmetry == 'p1':
        unit = p1_unit(region, va, vb, shape)
        
    if symmetry == 'p2':
        unit = p2_unit(region, va, vb, shape)
        
    if symmetry =='pm':
        unit = pm_unit(region)
    if symmetry =='pg':
        unit = pg_unit(region)
    if symmetry =='cm':
        unit = cm_unit(region)

    if symmetry =='pmm':
        unit = pmm_unit(region)
    if symmetry =='pmg':
        unit = pmg_unit(region)
    if symmetry =='pgg':
        unit = pgg_unit(region)
    if symmetry =='p4':
        unit = p4_unit(region)
        
    if symmetry == 'cmm':
        unit = cmm_unit(region)
    if symmetry == 'p4m':
        unit = p4m_unit(region)
    if symmetry == 'p4g':
        unit = p4g_unit(region)

    if symmetry == 'p3':
        unit = p3_unit(region)
    if symmetry == 'p3m1':
        unit = p3m1_unit(region)
    if symmetry == 'p31m':
        unit = p31m_unit(region)
    if symmetry == 'p6':
        unit = p6_unit(region)
    if symmetry == 'p6m':
        unit = p6m_unit(region)
    debug_show(region, f"translational unit cell\nVA:{VA}; VB:{VB}; VA_:{VA_}; VB_:{VB_}")
    step_outputs['translational_unit_cell'] = region
        
    # temporarily fix the issue that VB_ is one pixel large than right value
    if symmetry == 'p6m':
        VB_[1] = -unit.shape[1]//2


    # translate the translation unit cell
    if not rotation:
        result = vector_translate(unit, out_size, ts, VA_, VB_)
            
    else:
        angle = np.random.randint(0,360)
        result = vector_translate(unit, (int(out_size[0]*1.42), int(out_size[0]*1.42), out_size[2]),
                                    ts, VA_, VB_)
        
        ts = re_center_ts(ts, VA, VB, out_size)

        origin = int(out_size[0]*1.42//2), int(out_size[1]*1.42//2)

        debug_show(result, f'Before rotate: ts:{ts}, va:{va}, vb:{vb}', vectors=[ts, va, vb])
        debug_show(result, f'Before rotate: VA:{VA}, VB:{VB}, origin:{origin}, angle:{angle}', vectors=[ts, VA, VB])
                    
        result = rotate(result, angle, reshape=False)
        result = result[int((out_size[0]*1.42-out_size[0])//2):int((out_size[0]*1.42-out_size[0])//2)+out_size[0],
                        int((out_size[1]*1.42-out_size[1])//2):int((out_size[1]*1.42-out_size[1])//2)+out_size[1]]
        
        ts = rotate_xy(origin, ts, angle)
        ts = ts[0]-int((out_size[0]*1.42-out_size[0])//2), ts[1]-int((out_size[1]*1.42-out_size[1])//2)
        
        va, vb = rotate_xy((0,0), va, angle), rotate_xy((0,0), vb, angle)
        VA, VB = rotate_xy((0,0), VA, angle), rotate_xy((0,0), VB, angle)
        
        debug_show(result, f'After rotate: ts:{ts}, va:{va}, vb:{vb}', vectors=[ts, va, vb])
        debug_show(result, f'After rotate: VA:{VA}, VB:{VB}, origin:{origin}, angle:{angle}', vectors=[ts, VA, VB])
            

    # finalize the metadata and center the starting point
    ts = re_center_ts(ts, VA, VB, out_size)
            
    shape_dict = { 'rectangular': 0, 'square': 0, 'oblique': 1, 'hexagonal': 1, 'rhombic': 2, 'triangle': 3 }
    shape_int = shape_dict[shape]

    symmetry_dict = {'p1': 0, 'p2': 1, 'pm': 2, 'pg': 3, 'cm': 4, 'pmm': 5, 'pmg': 6, 'pgg': 7, 'cmm': 8, 
                     'p4': 9, 'p4m': 10, 'p4g': 11, 'p3': 12, 'p3m1': 13, 'p31m': 14, 'p6': 15, 'p6m': 16}
    symmetry_int = symmetry_dict[symmetry]

             
#     process the unit cell and metadata
    primitive_unit_cell_full = process_unit_cell(result, ts, va, vb, out_size)
    translational_unit_cell_full = process_unit_cell(result, ts, VA, VB, out_size)
    step_outputs['primitive_unit_cell_full'] = primitive_unit_cell_full
    step_outputs['translational_unit_cell_full'] = translational_unit_cell_full 
                                               
    metadata = np.array([ss, ts, va, vb, VA, VB, (symmetry_int, shape_int)] )  
        
    debug_show(result, "Output image")
    step_outputs['output_image'] = result

    return result, metadata, step_outputs


class DebugConfig:
    def __init__(self, show_steps=False):
        self.show_steps = show_steps

    def set_show_steps(self, show):
        self.show_steps = show

# Global Debug Configuration Instance
debug_config = DebugConfig()

def debug_show(image=None, title=None, vectors=None):
    if debug_config.show_steps:
        if image is not None:
            if vectors is not None:
                fig, ax = plt.subplots(1, 1)
                verify_image_vector(ax, image, vectors[0], vectors[1], vectors[2], title=title)
            else:
                plt.imshow(image)
                if title:
                    plt.title(title)
                plt.show()
            
        elif title:
            print(title)