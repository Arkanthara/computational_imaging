import numpy as np
import os
from skimage.io import imread

def genLfSequence(path: str, canonicalName: str, views: int, frame: int, format: str) -> np.ndarray:
    """
    Reads images from a path and returns the light field as a 4D array.

    Loads all views of a specific frame. The combined image sequence is the light field (of sub-aperture images) for that frame.

    Parameters
    ----------
    path : str
        Full file system path to folder containing images.
    canonicalName : str
        Common part of frame name.
    views : int
        Total number of sub-aperture views to read (1-16).
    frame : int
        Frame of interest from light field dataset.
    format : str
        Image file format, e.g., "png", "jpg".

    Returns
    -------
    np.ndarray
        4D matrix of images (Width x Height x Color Channels x Number of Sub-Aperture Views).

    Notes
    -----
    The file naming is specific to the Technicolor light-field dataset.
    Example:
        Filename      = "Painter_pr_00013_05.png"
        canonicalName = "Painter_pr_00"
        frame         = 13
        view          = 5

    This function is hard-coded for RGB images of size 1088x2048. This is sufficient for images from the Technicolor dataset used for this project. To extend usage to images from another dataset, update the dimensions appropriately.
    """

    
    # Format Frame Number in Filename
    sFrame = ""
    match len(str(frame)):
        case 1:
            sFrame = "00"+str(frame)
        case 2:
            sFrame = "0"+str(frame)
        case 3:
            sFrame = str(frame)

    # Get dimensions of first image to initialize the output matrix
    dim = imread(os.path.join(path, canonicalName + sFrame + "_00." + format)).shape
    assert len(dim) == 3, "Input images should be RGB. Please check the input image format and update the code if necessary."
    # Define Matrix to Store ImSequence
    lfSequence = np.zeros((dim[0], dim[1], dim[2], views), dtype=np.uint8)

    # Read Images
    for i in range(views):

        # Format View Number in Filename
        view = i
        sView = ""
        if view < 10:
            sView = "_0"+str(view)
        else:
            sView = "_"+str(view)

        # Set File Path
        fullURI = os.path.join(path, canonicalName + sFrame + sView + "." + format)

        # Read Images
        lfSequence[:,:,:,i] = imread(fullURI)
    return lfSequence

