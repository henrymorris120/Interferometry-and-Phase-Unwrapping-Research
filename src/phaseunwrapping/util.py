import numpy as np



def wrap_function(phi):
    """Computes the wrap function elementwise for the input array.
    """

    return ( (phi + np.pi) % (2*np.pi) ) - np.pi





