
import numpy as np  # numerics
import math as math  # math
import os as os
from skimage.filters import threshold_otsu, threshold_li,threshold_yen

from skimage.morphology import binary_dilation, binary_erosion
from skimage.measure import label, regionprops
from skimage.segmentation import flood_fill






def make_contour_mask(img, threshold_type="li"):
    """ Expects a MIP fluo img"""

    ## Determining if cells are on the left of right
    if (np.mean(img[:][0,len(img[0])//2]) > np.mean(img[:][len(img[0])//2::])):
        left = True
        print("Left")
    else:
        left=False
        print("Right")

    if (threshold_type == "li"):
        thresholded = img>threshold_li(img)
    elif (threshold_type =="otsu"):
        thresholded = img>threshold_otsu(img)
    dilated = thresholded.copy()
    
    connected_contour_flag = False
    N_it = 0
    while not (connected_contour_flag):
        N_it +=1
        
        contour = np.bitwise_xor(binary_dilation(dilated), dilated)

        print("Labelling")
        lab = label(contour)

        max_size = 0
        candidates = []
        print("Finding the biggest contour")
        for i in range(1,np.amax(lab)):
            sub_contour = lab==i
            
            if (np.sum(sub_contour[0]) >0 and np.sum(sub_contour[len(sub_contour)-1] > 0)): #checking that it connects top and bottom$
                if (get_migration_index(sub_contour)<5):
                    candidates.append(i)
                    print("CANDIDATE")
                    break
                
        print("Flood fill")

        if (len(candidates) > 0):
            connected_contour_flag = True
        
        dilated = binary_dilation(dilated)
        

        if (N_it==50):
            print("/!\ Aborted")
            return np.zeros_like(img),np.zeros_like(img)

    print("Successful after", N_it, " iterations")
    tentative_contour = lab==candidates[0]      
    tentative_contour = np.array(tentative_contour,dtype=int)

    ## Start flood filling from the opposite direction
    if not (left):
        fill = flood_fill(tentative_contour,(int(len(img)/2),0),2,connectivity=1)
    else:
        fill = flood_fill(tentative_contour,(int(len(img)/2),len(img[0])-1),2,connectivity=1)

    
    #fill = flood_fill(tentative_contour,(int(len(img)/2),int(len(img[0])/2)),2,connectivity=1)
    fill = fill==2
        

    print("Cleaning")
    contour_final = np.bitwise_xor(binary_dilation(fill),fill)
    return contour_final,fill



def get_contour_end_points(contour):
    top = np.where(contour[0])[0][0]
    bottom = np.where(contour[len(contour)-1])[0][0]
    return top,bottom


def get_migration_index(contour):
    #print("Computing MI")
    top,bottom = get_contour_end_points(contour)
    #print(top, bottom)
    delta_y_squared = len(contour)**2
    base_distance = np.sqrt((top - bottom)**2 + delta_y_squared)
    
    return np.sum(contour)/base_distance

def get_contour_distances(contour):
    top,bottom = get_contour_end_points(contour)
    delta_y_squared = len(contour)**2
    base_distance = np.sqrt((top - bottom)**2 + delta_y_squared)
    
    return base_distance,np.sum(contour)
                      


