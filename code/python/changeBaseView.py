import numpy as np

def changeBaseView(s: int, t: int) -> np.ndarray:
    """
    Returns the shift matrix relative to a specific camera in a 4x4 camera array.

    The base shift matrix is relative to view (1,1), the center camera. This function takes a camera's position in the array (s, t) as input and returns a shift matrix relative to the specified camera.

    Parameters
    ----------
    s : int
      Camera array row number (0-3).
    t : int
      Camera array column number (0-3).

    Returns
    -------
    np.ndarray
      New shift matrix for the specific camera (s, t) in the array.

    Notes
    -----
    The base shift matrix is as specified in:
      N. Sabater et al., "Dataset and Pipeline for Multi-view Light-Field Video", 
      2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 
      Honolulu, HI, 2017, pp. 1743-1753.
    """

    shiftMat = np.zeros((4,4,2));
    shiftMat[:,:,0] = np.array([[100,-0.36,-97.19,-195.55],
                               [98.67,0,-96.18,-197.85],
                               [99.17,0.21,-98.33,-197],
                               [99.08,-1.22,-99.26,-198.36]]);
    shiftMat[:,:,1] = np.array([[98.28,98.14,98.07,97.35],
                       [-1.73,0,0.74,0.11],
                       [-99.93,-99.11,-101.12,-99.07],
                       [-197.68,-198.14,-198.89,-199.37]]);

    # Initialize New Shift Matrix
    newShiftMat = np.zeros((4,4,2));

    # Compute New Shift Matrix for Camera at (s,t)
    newShiftMat[:,:,0] = shiftMat[:,:,0] - shiftMat[s,t,0]
    newShiftMat[:,:,1] = shiftMat[:,:,1] - shiftMat[s,t,1]
    return newShiftMat

