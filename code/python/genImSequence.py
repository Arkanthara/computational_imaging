import numpy as np
import os
from skimage.io import imread

def genImSequence(path: str, canonicalName: str, frames: int, view: int, format: str) -> np.ndarray:
    """
    Reads images from a path and returns an image sequence for video as a 4D array.

    Loads all frames of a specific view. The combined image sequence can be played back as a video.

    Parameters
    ----------
    path : str
        Full file system path to folder containing images.
    canonicalName : str
        Common part of file name.
    frames : int
        Total number of frames to read.
    view : int
        View of interest from light field dataset (0-15).
    format : str
        Image file format, e.g., "png", "jpg".

    Returns
    -------
    np.ndarray
        4D matrix of images (Width x Height x Color Channels x Number of Frames). Can be played back using MATLAB's 'implay' function.

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

    # Format View Number in Filename
    sView = ""
    if view < 10:
        sView = "_0"+str(view)
    elif view >= 10:
        sView = "_"+str(view)
    else:
        sView = ""

    # Get dimensions of first image to initialize the output matrix
    dim = imread(os.path.join(path, canonicalName + "000" + sView + "." + format)).shape
    assert len(dim) == 3, "Input images should be RGB. Please check the input image format and update the code if necessary."
    # Define Matrix to Store ImSequence
    imSequence = np.zeros((dim[0], dim[1], dim[2], frames), dtype=np.uint8)

    # Read Images
    for i in range(frames):
        sFrame = str(i)
        # Format Frame Number in Filename
        frame = i
        match len(str(frame)):
            case 1:
                sFrame = "00"+str(frame)
            case 2:
                sFrame = "0"+str(frame)
            case 3:
                sFrame = str(frame)

        # Set File Path
        fullURI = os.path.join(path, canonicalName + sFrame + sView + "." + format)
        # NOTE: This line is intentionally left uncommented so that  it can serve 
        # as a progress indicator of sorts.

        # Read Images
        imSequence[:,:,:,i] = imread(fullURI)
    return imSequence

