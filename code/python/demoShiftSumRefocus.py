import numpy as np
import os
import matplotlib.pyplot as plt
from genLfSequence import genLfSequence
from changeBaseView import changeBaseView
from shiftSumRefocus import shiftSumRefocus
## Description

# This script serves as a demo to illustrate how the functions in this
# package can be used to perform shift-sum-refocus on sub-aperture images
# from the technicolor lightField dataset. The script will perform
# shift-sum refocus at 6 pre-determined depth levels and diplay the
# outputs. To modify this, please modify line 73.
#
# < Required Input >
# 1. Sub Aperture Image Frames from the technicolor lightField Dataset
#
# Sample images (frame 85 from the lightField dataset) have been included 
# in the 'Sample-Data/Sample-ShiftSum' folder. Refocused output will be
# displayed on screen.
#
# NOTE: All views from only frame 85 are included in the package due to file 
# size restrictions.
#
# This script is capable of running as is and without any modification. All
# paths are relative and should function identically in Windows/Mac/Linux
# environments. The script has been tested in Windows 10 and works as
# expected.
#

## Set Paths and Parameters Specific to Data
scriptPath = os.path.dirname(os.path.abspath(__file__)) # Get Path to Current Script
inputPath = os.path.join(scriptPath, "Sample_Data", "Sample_ShiftSum") # Specifies path to input sub-aperture images
views = 16                                 # Specifies number of sub-aperture images to process
frameOfInterest = 85                       # Specifies frame of interest

## Load Images into LightField
lightField = genLfSequence(inputPath,"Painter_pr_00",views,frameOfInterest,"png")
# Refer to 'genLfSequence.m' Function File for more information about this
# function.

## Display All Sub-Aperture Views of Given Frame
plt.figure(figsize=(10,10))
plt.subplot(4,4,1), plt.imshow(lightField[:,:,:,0]), plt.title("0")
plt.subplot(4,4,2), plt.imshow(lightField[:,:,:,1]), plt.title("1")
plt.subplot(4,4,3), plt.imshow(lightField[:,:,:,2]), plt.title("2")
plt.subplot(4,4,4), plt.imshow(lightField[:,:,:,3]), plt.title("3")
plt.subplot(4,4,5), plt.imshow(lightField[:,:,:,4]), plt.title("4")
plt.subplot(4,4,6), plt.imshow(lightField[:,:,:,5]), plt.title("5")
plt.subplot(4,4,7), plt.imshow(lightField[:,:,:,6]), plt.title("6")
plt.subplot(4,4,8), plt.imshow(lightField[:,:,:,7]), plt.title("7")
plt.subplot(4,4,9), plt.imshow(lightField[:,:,:,8]), plt.title("8")
plt.subplot(4,4,10), plt.imshow(lightField[:,:,:,9]), plt.title("9")
plt.subplot(4,4,11), plt.imshow(lightField[:,:,:,10]), plt.title("10")
plt.subplot(4,4,12), plt.imshow(lightField[:,:,:,11]), plt.title("11")
plt.subplot(4,4,13), plt.imshow(lightField[:,:,:,12]), plt.title("12")
plt.subplot(4,4,14), plt.imshow(lightField[:,:,:,13]), plt.title("13")
plt.subplot(4,4,15), plt.imshow(lightField[:,:,:,14]), plt.title("14")
plt.subplot(4,4,16), plt.imshow(lightField[:,:,:,15]), plt.title("15")
plt.suptitle("All Views from Frame-"+str(frameOfInterest))

## Compute Refocused Image

# Define Shift Matrix for Painter Scene
shiftMat = changeBaseView(1,1)

# Run Loop To Generate Views at Different Depths (0.5 Increments)

depth = 2 # Starting Depth

plt.figure()
for i in range(6):
     temp = shiftSumRefocus(lightField,4,4,shiftMat,depth) # Compute Shift Sum Refocus
     plt.subplot(2,3,i+1), plt.imshow(temp.astype(np.uint8)), plt.title("Z = "+str(depth)+" meters")
     depth = depth+0.5

plt.suptitle("Refocused at Different Depths")
plt.show()