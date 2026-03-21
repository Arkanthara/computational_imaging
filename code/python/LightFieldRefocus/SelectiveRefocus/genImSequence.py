import numpy as np
from skimage.io import imread

def genImSequence(path,canonicalName,frames,view,format):
    """
    GENIMSEQUENCE Reads Images from Path and Returns Image Sequence for Video
       The function loads all frames of a specific view. The combined image
       sequence can be then played back as a video.

       ------------------
       Function Inputs
       ------------------

       path            --->  (String) Full File System Path to Folder Containing
                             Images.
       canonicalName   --->  (String) Common Part of File Name
       frames          --->  Total Number of Frames to Read
       view            --->  View of Interest from Light Field Dataset (0-15)
       format          --->  (String) Image File Format eg: "png" , "jpg"

       Note: The file naming is specfiic to the technicolor light-field
       dataset.

       File Naming Example
       --------
       Filename      = "Painter_pr_00013_05.png"
       canonicalName = "Painter_pr_00";
       frame         = 13
       view          = 5

       ------------------
       Function Output
       ------------------
    
       imSequence      --->  A 4D matrix of images (Width x Height x Color Channels x Number of Frames).
                             Can be played back using MATLAB's 'implay'
                             function.

     NOTE: This function is hard-coded for RGB images of size
     1088x2048. This is sufficient for images from the Technicolor dataset
     that was used for this project. To extend usage to images from another
     dataset please make sure to update the dimensions (line 59) appropriately.                       
    """

    # Format View Number in Filename
    sView = ""
    if view < 10:
        sView = "_0"+str(view)
    elif view >= 10:
        sView = "_"+str(view)
    else:
        sView = ""

    # Define Matrix to Store ImSequence
    imSequence = np.uint8(np.zeros((1088,2048,3,frames)))

    # Read Images
    for i in range(frames):
        sFrame = str(i)
        # Format Frame Number in Filename
        frame = i-1
        match len(str(frame)):
            case 1:
                sFrame = "00"+str(frame)
            case 2:
                sFrame = "0"+str(frame)
            case 3:
                sFrame = str(frame)

        # Set File Path
        fullURI = path+canonicalName+sFrame+sView+"."+format
        # NOTE: This line is intentionally left uncommented so that  it can serve 
        # as a progress indicator of sorts.

        # Read Images
        imSequence[:,:,:,i] = imread(fullURI)
    return imSequence

