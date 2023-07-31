import numpy as np
import scipy.sparse as sps

import scipy.sparse as scipy_sparse
from scipy.sparse import linalg as scipy_splinalg
from scipy.sparse import csc_matrix


def wrap_function(phi):
    """Computes the wrap function elementwise for the input array.
    """

    return ( (phi + np.pi) % (2*np.pi) ) - np.pi



def upsampling_matrix(n, sparse=False):
    """Constructing the upsampling matrix.
    """

    if not sparse:
        P = np.zeros((n+1, n))
        P[1:, :] = np.eye(n)
        return P
    else:
        P = csc_matrix((n+1,n))
        P.setdiag(1,k=-1)
        return P

        


def banded_cholesky_factor(A, check=False):
    """
    Given a sparse banded matrix (SciPy or CuPy) :math:`A`, returns the Cholesky factor :math:`L` in the factorizations
    :math:`A = L L^T` with :math:`L` lower-triangular. Here :math:`A` must be a positive definite matrix.
    """

    xp = np
    sp = scipy_sparse
    splinalg = scipy_splinalg
    
    # Shape
    n = A.shape[0]
    
    # Sparse LU
    LU = splinalg.splu(A, diag_pivot_thresh=0, permc_spec="NATURAL") 

    # Check for positive-definiteness
    posdef_check = ( LU.perm_r == xp.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all()
    assert posdef_check, "Matrix not positive definite!"
    
    # Extract factor
    L = LU.L.dot( sp.diags(LU.U.diagonal()**0.5) )
    
    # Check?
    if check:
        guess = L @ L.T
        norm_error = xp.linalg.norm( guess.toarray() - A.toarray() )
        print(f"L2 norm between L L^T and A: {norm_error}")
    
    return L, LU



def build_1d_first_order_grad(N, boundary="none"):
    """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions.
    """
    
    assert boundary in ["none", "periodic", "zero"], "Invalid boundary parameter."
    
    d_mat = sps.eye(N)
    d_mat.setdiag(-1,k=-1)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        d_mat[0,-1] = -1
    elif boundary == "zero":
        pass
    elif boundary == "none":
        d_mat = d_mat[1:,:]
    else:
        pass
    
    return d_mat



def build_2d_first_order_grad(M, N, boundary="none"):
    """Constructs a SciPy sparse matrix that extracts the discrete gradient of an input image.
    Assumes periodic BCs. Input image should have original dimension (M,N), must be flattened
    to compute matrix-vector product. First set is horizontal gradient, second is vertical.
    """

    # Construct our differencing matrices
    d_mat_horiz = build_1d_first_order_grad(N, boundary=boundary)
    d_mat_vert = build_1d_first_order_grad(M, boundary=boundary)
    
    # Build the combined matrix
    eye_vert = sps.eye(M)
    d_mat_one = sps.kron(eye_vert, d_mat_horiz)
    
    eye_horiz = sps.eye(N)
    d_mat_two = sps.kron(d_mat_vert, eye_horiz)

    full_diff_mat = sps.vstack([d_mat_one, d_mat_two])
    
    return d_mat_one, d_mat_two



def build_1d_second_order_grad(N, boundary="none"):
    """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions.
    """
    
    assert boundary in ["none", "zero", "periodic"], "Invalid boundary parameter."
    
    d_mat = -2*sps.eye(N)
    d_mat.setdiag(1,k=-1)
    d_mat.setdiag(1,k=1)
    d_mat = d_mat.tolil()
    
    if boundary == "none":
        d_mat = d_mat[1:-1,:]
    elif boundary == "zero":
        pass
    elif boundary == "periodic":
        d_mat[0,-1] = 1
        d_mat[-1,0] = 1
    else:
        raise NotImplementedError

    return d_mat



# FROM HSIP

# def build_diff_mat_2nd_order(N, boundary="none"):
#     """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
#     Boundary parameter specifies how to handle the boundary conditions.
#     """
    
#     assert boundary in ["periodic"], "Invalid boundary parameter."
    
#     d_mat = -sp.eye(N)
#     d_mat.setdiag(2,k=1)
#     d_mat.setdiag(-1,k=2)
#     d_mat = sp.csc_matrix(d_mat) 
    
#     if boundary == "periodic":
#         d_mat[-2,0] = -1
#         d_mat[-1,0] = 2
#         d_mat[-1,1] = -1
#     elif boundary == "zero":
#         pass
#     elif boundary == "none":
#         pass
#     else:
#         pass
    
#     return d_mat



# def build_diff_mat_2nd_order_2d(M, N, boundary="none"):
#     """Constructs a SciPy sparse matrix that extracts the discrete gradient of an input image.
#     Assumes periodic BCs. Input image should have original dimension (M,N), must be flattened
#     to compute matrix-vector product. First set is horizontal gradient, second is vertical.
#     """

#     # Construct our differencing matrices
#     d_mat_horiz = build_diff_mat_2nd_order(N, boundary=boundary)
#     d_mat_vert = build_diff_mat_2nd_order(M, boundary=boundary)
    
#     # Build the combined matrix
#     eye_vert = sp.eye(M)
#     d_mat_one = sp.kron(eye_vert, d_mat_horiz)
    
#     eye_horiz = sp.eye(N)
#     d_mat_two = sp.kron(d_mat_vert, eye_horiz)
    
#     full_diff_mat = sp.vstack([d_mat_one, d_mat_two])
    
#     return full_diff_mat


