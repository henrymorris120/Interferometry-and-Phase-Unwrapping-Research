import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
from scipy.fft import dctn, idctn



def dct_diagonalized_operator_get_eigvals(A, grid_shape):
    """Given a LinearOperator A that is diagonalized by the 2-dimensional DCT, computes its eigenvalues.
    """
    M, N = grid_shape
    v = np.random.normal(size=(M,N))
    tmp = A @ ( idctn( v, norm="ortho" ).flatten()  )
    tmp = tmp.reshape((M,N))
    tmp = dctn( tmp, norm="ortho" ).flatten()
    res = tmp/v.flatten()
    return res



def dct_diagonalized_operator_sqrt(A, grid_shape, offset=True, offset_val=1e-2):
    """Given a LinearOperator A that is diagonalized by the DCT, performs the diagonalization (computes eigenvalues),
    computes the square root E in M = E E^T, and returns a LinearOperator representing E^{-1}.

    This can be used as a preconditioner. Instead of solving A x = b, it is equivalent to solve
        E^{-1} A E^{-T} w = E^{-1} b
    and recovering x = E^{-T} w.
    """
    # Get eigenvalues
    eigvals = dct_diagonalized_operator_get_eigvals(A, grid_shape)

    # Offset, if E is singular
    if offset:
        eigvals[0] = offset_val

    # Shape
    M, N = grid_shape

    def _matvec(x):
        x = x.reshape(grid_shape)
        tmp = dctn( x, norm="ortho" ).flatten()
        tmp = tmp/np.sqrt(eigvals)
        return tmp
    
    def _rmatvec(x):
        tmp = x/np.sqrt(eigvals)
        tmp = tmp.reshape(grid_shape)
        tmp = idctn( tmp, norm="ortho" ).flatten()
        return tmp

    Einv = LinearOperator(A.shape, matvec=_matvec, rmatvec=_rmatvec)

    return Einv