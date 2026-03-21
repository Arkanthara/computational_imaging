import numpy as np

def changeBaseView(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Parameters
    -----------
    s : np.ndarray
        Camera Array Row Number. Runs from 0 - 3 like in the paper.
    t : np.ndarray
        Camera Array Column Number. Runs from 0 - 3 like in the paper.
    Returns
    -----------
    newShiftMat : np.ndarray
        New shift matrix, for a specific camera (s,t) in the array.
    """

    shiftMat = np.zeros((4,4,2));
    shiftMat[:,:,1] = np.array([[100,-0.36,-97.19,-195.55],
                               [98.67,0,-96.18,-197.85],
                               [99.17,0.21,-98.33,-197],
                               [99.08,-1.22,-99.26,-198.36]]);
    shiftMat[:,:,2] = np.array([[98.28,98.14,98.07,97.35],
                       [-1.73,0,0.74,0.11],
                       [-99.93,-99.11,-101.12,-99.07],
                       [-197.68,-198.14,-198.89,-199.37]]);

    # Initialize New Shift Matrix
    newShiftMat = np.zeros((4,4,2));

    # Compute New Shift Matrix for Camera at (s,t)
    newShiftMat[:,:,1] = shiftMat[:,:,1] - shiftMat[s+1,t+1,1]
    newShiftMat[:,:,2] = shiftMat[:,:,2] - shiftMat[s+1,t+1,2]
    return newShiftMat

