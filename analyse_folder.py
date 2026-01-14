#!/usr/bin/python3

"""
Copyright: Combriat, 2025

"""


import matplotlib.pyplot as plt 
import numpy as np  # numerics
import math as math  # math
import os as os
import random
import sys as sys
import nd2 as nd2
from skimage.filters import threshold_otsu, threshold_li,threshold_yen

from skimage.morphology import binary_dilation, binary_erosion
from skimage.measure import label, regionprops
from skimage.segmentation import flood_fill
from src.ContourUtils import *
from src.Segmenter import *
import pickle
from skimage.transform import rotate




rotatee = False

folder = sys.argv[1]
if (folder[-1] != os.sep):
    folder+=os.sep

ls = os.listdir(folder)
ls_filtered = []
for i in ls:
    if (".nd2" in i and not "BF" in i and not ".tiff" in i):
        ls_filtered.append(i)

ls = ls_filtered

try:
    os.mkdir(folder+"results")
except:
    pass

if (len(sys.argv)>2):
    for i in range(2,len(sys.argv)):
        if (sys.argv[i]=="--rotate"):
            print("* Images will be rotated")
            rotatee=True



model_file = open("models/202511_cB.pic","rb")
mySegmenter = pickle.load(model_file)
model_file.close()


res_output=open(folder+"results"+os.sep+"res.txt","w")
log_output=open(folder+"results"+os.sep+"log.txt","w")
fig = plt.figure(figsize=(30,30))

for i in range(len(ls)):
    print("* Analysing:",ls[i])

    img = nd2.imread(folder+ls[i])
    img = np.amax(img,axis=0) # MIP
    if (img.ndim == 2):
        img = np.array([img])
    print(img.shape)   
    if (rotatee):
        for j in range(len(img)):
            img[j] = rotate(img[j],90,preserve_range=True, resize=True)
    print(img.shape)
    for j in range(len(img)): ## loop on channels
        successful = True
        contour,fill = make_contour_mask(img[j])
        method = "li"
        if (np.sum(contour)==0):
            ## Second attempt
            print("Switching to OTSU")
            contour,fill = make_contour_mask(img[j], threshold_type="otsu")
            method = "otsu"
            
            if (np.sum(contour)==0):
                print("Switching to ML")
                contour,fill = make_contour_mask(img[j],threshold_type="manual",mask= mySegmenter.segment(img[j]))
                method="ml"
                
                if (np.sum(contour)==0):
                    res_output.write("#### File: "+ls[i]+" Channel: "+str(j)+" SKIPPED\n")
                    successful = False
            #break
        if (successful):
            migration_index = get_migration_index(contour)

            res_output.write("#### File: "+ls[i]+"\n")
            res_output.write("## Channel: "+str(j)+"\n")
            res_output.write("## Dimension: "+str(img.shape)+"\n")
            res_output.write("## Method: "+method+"\n")


            if (np.mean(img[j][:,0:int(len(img[j][0])/2)]) > (np.mean(img[j][:,int(len(img[j][0])/2)::]))):
                res_output.write("## Cells on the left\n")
            else:
                res_output.write("## Cells on the right\n")

            res_output.write("## Contour_length Shortest_line_length Ratio\n")
            base_distance, contour_length = get_contour_distances(contour)
            res_output.write("%f %f %f\n" % (contour_length,base_distance,contour_length/base_distance))
            res_output.write("\n\n")


            plt.imshow(img[j],cmap="gist_gray")
            #plt.imshow(contour,cmap="inferno",alpha=.7)
            plt.contour(contour,colors="white",levels=[.5],linewidths=.5)
            plt.savefig(folder+"results"+os.sep+ls[i]+"_ch"+str(j)+".png")
            plt.clf()
        else:
            log_output.write("File: "+ls[i]+"  CHAN: " + str(j)+" was not successfully analysed\n")

        
            
                                                    
res_output.close()
log_output.close()
        
