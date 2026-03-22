import numpy as np
from skimage.transform import warp as imtranslate
from skimage.transform import AffineTransform
from skimage.exposure import rescale_intensity as mat2gray

def shiftSumRefocus(lightField: np.ndarray, arrayLength: int, arrayDepth: int, shiftMat: np.ndarray, depth: float) -> np.ndarray:
    """
    Refocuses a scene using the shift-sum refocus method.

    The function takes the sub-aperture images of a scene as input and
    uses the shift and sum technique to compute a refocused image of
    the scene at a specified depth.

    Parameters
    ----------
    lightField : np.ndarray
        4D array containing sub-aperture images of a scene. Use output from 'genLfSequence'.
    arrayLength : int
        Number of sub-aperture images along the first array dimension.
    arrayDepth : int
        Number of sub-aperture images along the second array dimension.
    shiftMat : np.ndarray
        NxNx2 array containing U-V shift relative to reference camera. This parameter is unique to a camera rig. Will need to be modified when processing light-fields from a different rig.
    depth : float
        Depth (in meters). Specifies the depth plane that will be in focus. Depth is measured along the axis that runs from the camera and into the scene.

    Returns
    -------
    np.ndarray
        Refocused image of a scene.

    Notes
    -----
    This function is hard-coded for RGB images of size 1088x2048. This is sufficient for images from the Technicolor dataset that was used for this project. To extend usage to images from another dataset, please update the dimensions appropriately.
    """

    # Define Expression for Depth Parameter, d(z)
    z = depth
    z0 = 100   # Pre-Defined
    z1 = 1.63  # Pre-Defined

    d = ((1/z)-(1/z0))/((1/z1)-(1/z0))

    # NOTE: The depth expression comes from the "Dataset and Pipeline for 
    # Multi-view Light-Field Video" cited below.

    # N. Sabater et al., "Dataset and Pipeline for Multi-view Light-Field Video", 
    # 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 
    # Honolulu, HI, 2017, pp. 1743-1753.

    # Get dimensions of first image to initialize the output matrix
    dim = lightField[:,:,:,0].shape
    # Run Loop through Sub-Aperture Images
    refocusedImage = np.zeros((dim[0], dim[1], dim[2]), dtype=np.double)
    index = 0
    print("shiftMat: ", shiftMat)
    print("shiftMat type: ", type(shiftMat))
    for i in range(arrayLength):
        for j in range(arrayDepth):
            index = index
            print("Transform: ", [-1*d*shiftMat[i,j,0], -1*d*shiftMat[i,j,1]])
            tform = AffineTransform(translation=[-1*d*shiftMat[i,j,0], -1*d*shiftMat[i,j,1]])
            # Perform Shift and Sum
            refocusedImage = refocusedImage + np.double(np.array(imtranslate(lightField[:,:,:,index], tform)))

    # Compute Average
    refocusedImage = np.uint8(255*mat2gray(refocusedImage/(arrayLength*arrayDepth)))
    return refocusedImage
