import numpy as np
from .util import wrap_function



def mle_1d(psi, weights=None):
    """Given a 1D wrapped phase vector psi, computes the MLE estimator corresponding to the D2 data fidelity term.
    
    psi: the wrapped phase.
    weights: weighint for the D2 penalty function.

    """

    # Figure out shape
    n = len(psi)

    # Build F matrix
    F = np.eye(n)
    np.fill_diagonal(F[1:], -1)
    F = F[1:,:]

    # 
    # Make $\phi_1$
    phi1 = np.zeros(n)
    phi1[0] = psi[0]

    # Build the undersampling matrix
    P = np.eye(n)[1:,:].T

    # Set initial weights
    if weights is None:
        weights = np.ones(F.shape[0])
    else:
        assert len(weights) == len(psi), "psi and weight vector must have same length!"

    # rhs vector
    rhs = P.T @ F.T @ np.diag(weights) @ ( wrap_function(F @ psi) - (F @ phi1) )

    # Q matrix
    Q = P.T @ F.T @ np.diag(weights) @ F @ P  

    # Solve system for answer
    phi2 = np.linalg.solve(Q, rhs)

    # Append first entry
    reconstructed_phi = np.zeros(n)
    reconstructed_phi[1:] = phi2
    reconstructed_phi[0] = psi[0]

    return reconstructed_phi
    



















