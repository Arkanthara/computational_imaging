import numpy as np
from scipy.stats import mode
import skimage as sk

def selectiveBlurring(img: np.ndarray, imgDepth: np.ndarray, numLevels: int, focusRange: list, upperFilt: int):
    """
    Selectively blurs an input image based on depth.

    Pixels at depths within the 'focusRange' are not blurred and remain in focus. Other pixels are blurred using Gaussian filters of different sizes.

    Parameters
    ----------
    img : np.ndarray
        RGB input image (uint8) to be selectively blurred.
    imgDepth : np.ndarray
        Grayscale depthmap image (uint8) corresponding to the input image. Depth levels closer to the viewer have a low value (closer to 0), while depth levels further away have a high value (closer to 255).
    numLevels : int
        Number of bins for grouping depth levels. Must be greater than 1.
    focusRange : list
        Length-2 list specifying the range of depth levels that should not be blurred. Example: [30, 40] instructs the function to blur all pixels whose depth lies in bins outside of the specified range.
    upperFilt : int
        Upper limit for the kernel size of the Gaussian filter to be used for blurring.

    Returns
    -------
    np.ndarray
        Selectively blurred RGB image (uint8).

    Examples
    --------
    >>> imshow(selectiveBlurring(imread('Sample-Depth/painter-02.png'), imread('Sample-Depth/painter-02-depth.png'), 10, [0, 15], 25))

    Notes
    -----
    This function is written for images from the Technicolor dataset used for this project. To extend usage to images from another dataset, minor modifications may be needed.
    """

    # Convert imgDepth to grayscale if it is not already
    if len(imgDepth.shape) == 3:
        imgDepth = imgDepth[:, :, 0]  # Extract first channel (depth maps are often redundantly 3D)

    ## Initialization
    blurredImage = np.zeros(img.shape) # Initialize Output Image
    sizeImg = img.shape # Get Size of Input Image

    ## Levels and Dynamic Code
    modeValue = int(mode(imgDepth, axis=None, keepdims=False).mode)
    uLimit = min(modeValue + 30, 255)
    # We noticed that in the output depthmap, despite spreading the 50 available depth
    # levels between 0 and 255, there were rarely a few pixels at depth levels
    # higher than 65 (deeper levels are closer to the background). Inspecting the 
    # depthmaps across various frames we noticed that the effective upper limit 
    # was close to the mode of the intensity value of the depthmap image. Good 
    # outputs were obtained by setting the upper limit to 30 greater than the mode.
    #
    # For different scenes it may be necessary to tweak line 62 if the output
    # is not satisfactory.

    bins = np.linspace(0, float(uLimit), numLevels)
    # Available depth levels are binned into groups. There will be as many bins
    # as the value of 'numLevels' specified when calling the function.


    # Lines 35 to 43 define variables used later for checking bounds and assigning
    # filter levels to each bin.
    numBinsBelow = len(bins[bins < focusRange[0]]) - 1
    if numBinsBelow == -1:
        numBinsBelow = 0

    numBinsAbove = len(bins[bins > focusRange[1]])
    if numBinsAbove == -1:
        numBinsAbove = 0


    # Initialize 'chooseFilt'
    chooseFilt = np.zeros((numLevels), dtype=np.int32) # Initialize with all 0

    # This section (line 107 to 124) constitute the logic for assigning filter-levels 
    # to each bin. The mapping is stored in 'chooseFilt'. The idea is that all bins
    # containing levels within the 'focusRange' are assigned a filter-level of 1. 
    # Bins outside the focusRange are assigned a higher filter-level depending 
    # on the bin's distance from the focus range.
    #
    # --------
    # Example
    # --------
    # Let numLevels = 10, focusRange = [11 30]
    # 
    # bins                 = [00.00   10.11   20.22   30.33   40.44   50.55   60.66   70.77   80.88   91.00]
    # 
    # chooseFilt           = [  2       1       1       2       3       4       5       6       7       8  ]

    # MATLAB code uses 1-based indices; convert those writes to 0-based Python indices.
    chooseFilt[numBinsBelow] = 1
    for j in range(numBinsBelow + 1, numLevels - numBinsAbove):
        chooseFilt[j] = 1

    if((numBinsAbove!=0) or (numBinsBelow!=0)):
        if(numBinsAbove!=0):
            for j in range(numLevels - numBinsAbove, numLevels):
                chooseFilt[j] = chooseFilt[j-1]+1


        if(numBinsBelow!=0):
            for j in range(numBinsBelow - 1, -1, -1):
                chooseFilt[j] = chooseFilt[j+1]+1

    ## Generate Images Filtered at Each Level

    # This section (lines 150 and 151) generates a list of gaussian filter-sizes ('filtSize')
    # from 0.5 (no filtering) to upperFilt. There are as many filter levels as
    # there are bins.
    #
    # Note that filter-size in this context is actually the standard deviation 
    # of the gaussian distribution. MATLAB internally determines the filter size 
    # based on the standard deviation when using 'imgaussFilt'
    # 
    #
    # NOTE: Spacing between consecutive filter levels is currently linear. This
    # can be tweaked/modified to vary the difference in blurring across levels.
    # For instance, changing to a logarithmic or exponential spacing can have
    # different results. To make modifications, replace line 151 with another
    # line indicating the spacing of choice.
    #
    # --------
    # Example
    # --------
    # Let numLevels = 10, and upperFilt = 25
    #
    # filtSize             = [ 0.50    3.22    5.94    8.66   11.38   14.11   16.83   19.55   22.27   25.00]

    filt =np.zeros((sizeImg[0],sizeImg[1],sizeImg[2],numLevels), dtype=np.uint8) # Initialize Matrix to Store Filtered Images
    filtSize = np.arange(0.5, upperFilt+(upperFilt-0.5)/(numLevels-1), (upperFilt-0.5)/(numLevels-1))

    # After computing the filter sizes ('filtSize'), pre-filtered versions of the
    # input image are created and stored ('filt') for reference. This is later used 
    # as a look-up-table when generating the selectively blurred image. 
    #
    # filt is a 4D matrix with dimensions ImageWidth x ImageHeight x ColorChannels x numLevels.

    for i in range(numLevels):
        filtIdx = int(np.clip(chooseFilt[i] - 1, 0, numLevels - 1))
        filt[:, :, :, i] = sk.filters.gaussian(
            img,
            filtSize[filtIdx],
            preserve_range=True,
            channel_axis=-1,
        )

    # By iterating through the 4th dimension of filt, one can view versions
    # of the input image filtered by a different Gaussian filter chosen based 
    # on the bin to filter-level mapping ('chooseFilt') defined above. Please refer 
    # to the consolidated example below for an illustration.

    # --------
    #  Consolidated Example
    # --------
    # Let numLevels = 10, focusRange = [11 30] and upperFilt = 25
    # 
    # bins                 = [00.00   10.11   20.22   30.33   40.44   50.55   60.66   70.77   80.88   91.00]
    # 
    # chooseFilt           = [  2       1       1       2       3       4       5       6       7       8  ]
    #
    # filtSize             = [ 0.50    3.22    5.94    8.66   11.38   14.11   16.83   19.55   22.27   25.00]
    #
    # filtSize(chooseFilt) = [ 3.22    0.50    0.50    3.22    5.94    8.66   11.38   14.11   16.83   19.55]
    #
    # filt                 = 4D Matrix with dimensions ImageWidth x ImageHeight x ColorChannels x numLevels
    #
    # Descrption of Filtered Images in filt
    #
    # filt(:,:,:,1) --> Input image filtered using a Gaussian filter with Standard Deviation = 3.22 [filtSize(chooseFilt) (1)]
    # filt(:,:,:,2) --> Input image filtered using a Gaussian filter with Standard Deviation = 0.50 [filtSize(chooseFilt) (2)]
    # filt(:,:,:,3) --> Input image filtered using a Gaussian filter with Standard Deviation = 0.50 [filtSize(chooseFilt) (3)]
    # filt(:,:,:,4) --> Input image filtered using a Gaussian filter with Standard Deviation = 3.22 [filtSize(chooseFilt) (4)]
    # filt(:,:,:,5) --> Input image filtered using a Gaussian filter with Standard Deviation = 5.94 [filtSize(chooseFilt) (5)]
    # 
    # and so on...
    # 

    ## Generate Selectively Blurred Output Image
    #
    # In this section, we loop through the depthmap, pixel by pixel. At each
    # pixel location we read the depth level from the depth map and identify the
    # appropriate 'bin'. Based on the bin we know the level of blurring. We look
    # up the pixel value from the appropriately filtered image in 'filt' and use 
    # it to fill in the pixel value in the output 'blurredImage'.
    #
    # --------
    # Example
    # --------
    #
    # bins                 = [00.00   10.11   20.22   30.33   40.44   50.55   60.66   70.77   80.88   91.00]
    # 
    # chooseFilt           = [  2       1       1       2       3       4       5       6       7       8  ]
    #
    # filtSize(chooseFilt) = [ 3.22    0.50    0.50    3.22    5.94    8.66   11.38   14.11   16.83   19.55]
    #
    # filt                 = 4D Matrix with dimensions ImageWidth x ImageHeight x ColorChannels x numLevels
    # 
    # Given the parameters above. Suppose we are looping through 'imgDepth'
    # 
    # At i=100 and j=50 (location 100,50), let the depth level = 35 which falls
    # in bin 4 which has a filter-level of 2 that corresponds to a gaussian
    # filter with standard deviation of 3.22.
    #
    # filt(:,:,:,4) is a pre-filtered version of the input image filtered with
    # the appropriate gaussian filter. We copy the pixel value at (100,50) in
    # this image to the output image.
    #
    # That is, blurredImage(100,50,:) = filt(100,50,:,4)
    #


    #[~,loc] = max(chooseFilt)
    loc = np.argmax(chooseFilt)

    for i in range(sizeImg[0]):
        for j in range(sizeImg[1]):

            if(imgDepth[i,j]>=uLimit):
                blurredImage[i,j,:] = filt[i,j,:,loc]
                continue

            choosePix = len(bins[bins<imgDepth[i,j]])-1
            if choosePix==-1:
                choosePix = 0

            location = choosePix
            blurredImage[i,j,:] = filt[i,j,:,location]

    # Return Final Output with uint8 Datatype
    blurredImage = np.uint8(blurredImage)
    return blurredImage

