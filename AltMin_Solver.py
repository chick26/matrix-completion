from scipy.sparse import csc_matrix, csr_matrix, dok_matrix, lil_matrix, coo_matrix
from sklearn.metrics import mean_squared_error

from cvxpy import Minimize, Problem, Variable, SCS
from cvxpy import norm as cvxnorm
from cvxpy import vec as cvxvec

from scipy.optimize import least_squares
from scipy.linalg import norm as spnorm
import SetDataMatrix as Init
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rmse(A, B):
    """
    rmse(A, B) is a function to compute the relative mean-squared
    error between two matrices A and B, given by
    1/(m*n) * sum_ij (A_ij - B_ij)^2
    where A and B are m-by-n numpy arrays.
    """
    return np.sqrt(mean_squared_error(A,B))

def vec(A, stack="columns"):
    """
    vec(A) returns the vectorization of the matrix A
    by stacking the columns (or rows, respectively) of A.
    """
    if stack[0].lower() == 'c':
        return A.T.ravel()
    elif stack[0].lower() == 'r':
        return A.ravel()
    else:
        raise ValueError('Expected \'columns\' or \'rows\' for argument stack.')

def _get_sparse_type(st=None):
    """
    _get_sparse_type(st=None) is a bookeeping function to that determines 
    which type of sparse matrix to return, given its argument st.
    Note: st can be a function (e.g. scipy.sparse.csr_matrix), a string 
    (e.g., 'csr', 'csr_matrix'), or None (returns scipy.sparse.csr_matrix). 
    The output of this function is the corresponding sparse matrix constructor
    (e.g., scipy.sparse.csr_matrix). 
    """
    if (st is None) or (isinstance(st, 'str') and (st[:3] == 'csr')):
        from scipy.sparse import csr_matrix
        return csr_matrix
    elif isinstance(st, 'str') and (st[:3] == 'csc'):
        from scipy.sparse import csc_matrix
        return csc_matrix
    elif isinstance(st, 'str') and (st[:3] == 'coo'):
        from scipy.sparse import coo_matrix
        return coo_matrix
    elif isinstance(st, 'str') and (st[:3] == 'dok'):
        from scipy.sparse import dok_matrix
        return dok_matrix
    elif isinstance(st, 'str') and (st[:3] == 'lil'):
        from scipy.sparse import lil_matrix
        return lil_matrix
    else:
        raise ValueError('Could not detect type of sparse matrix constructor to return.')
        
def unvec(vecA, shape):
    """
    _unvec(A, shape) returns the "unvectorization" of the
    matrix A by unstacking the columns of vecA to return
    the matrix A of shape shape.
    """
    return vecA.reshape(shape, order='F')


def matIndicesFromMask(mask):
    """
    matIndicesFromMask(mask) returns the matrix-indices 
    corresponding to mask == 1. This operation returns a 
    tuple containing a list of row indices and a list of 
    column indices.
    """
    return np.where(mask.T==1)[::-1]


def masked(A, mask):
    """
    masked(A, mask) returns the "observed entries" of the
    matrix A, as a vector, determined according to the 
    condition mask == 1 (alternatively, the entries for 
    which mask is True).
    """
    return A[matIndicesFromMask(mask)]

def matricize_right(V, Omega, m=None, sparse=True, sparse_type=None):
    """
    matricize_right(V, Omega, m=None, sparse=True, sparse_type=None) 
    turns the problem 
        M_Omega = (U @ V.T)_Omega 
    into the matrix problem
        vec(M_Omega) = W @ vec(U)
    where U is an m-by-r matrix, V is an n-by-r matrix and
        vec([[1,2,3],[4,5,6]]) = [1,4,2,5,3,6].T

    Input
              V : the right n-by-r matrix
          Omega : the mask / list of indices of observed entries
         sparse : whether to return a sparse matrix (default: true)
    sparse_type : what kind of sparse matrix to return (default: csr)

    Output
    V_op : The operator for V in matrix form so that vec(U @ V.T) is 
           equivalent to V_op @ vec(U).
    """
    if isinstance(Omega, tuple):
        Omega_i = Omega[0]
        Omega_j = Omega[1]
        if m is None:
            raise ValueError('input number of columns for left' +
                             ' factor is required when Omega is a ' +
                             'list of indices')
    elif isinstance(Omega, np.ndarray):
        m = Omega.shape[0]
        Omega_i, Omega_j = matIndicesFromMask(Omega)
    else:
        raise ValueError('type of Omega not recognized; ' + 
                         'expected tuple of indices or mask array.')
    r = V.shape[1]
    sizeU = m*r
    if sparse:
        sp_mat = _get_sparse_type(sparse_type)
        row_idx = np.repeat(range(Omega_i.size), r)
        col_idx = [np.arange(Omega_i[n], sizeU, m, dtype=int) 
                   for n in range(Omega_i.size)]
        col_idx = np.concatenate(col_idx)
        vals = np.concatenate([V[j,:] for j in Omega_j])
        V_op = sp_mat((vals, (row_idx, col_idx)), shape=(Omega_i.size, sizeU))
    else:
        V_op = np.zeros((Omega_i.size, sizeU))
        for n in range(Omega_i.size):
            i = Omega_i[n]
            j = Omega_j[n]
            V_op[n, i::m] = V[j,:]
    return V_op


def matricize_left(U, Omega, n=None, sparse=True, sparse_type=None):
    """
    matricize_left(U, Omega, n=None, sparse=True, sparse_type=None) 
    turns the problem
        M_Omega = (U @ V.T)_Omega
    into the matrix problem
        vec(M_Omega) = W @ vec(V)
    where U is an m-by-r matrix, V is an n-by-r matrix and
        vec([[1,2,3],[4,5,6]]) = [1,4,2,5,3,6].T

    Input
              U : the left m-by-r matrix
          Omega : the mask / list of indices of observed entries
         sparse : whether to return a sparse matrix (default: true)
    sparse_type : what kind of sparse matrix to return (default: csr)

    Output
    U_op : The operator for U in matrix form so that vec(U @ V.T) is 
           equivalent to U_op @ vec(V).
    """
    if isinstance(Omega, tuple):
        Omega_i = Omega[0]
        Omega_j = Omega[1]
        if n is None:
            raise ValueError('input number of columns for right' +
                             ' factor is required when Omega is a ' +
                             'list of indices')
    elif isinstance(Omega, np.ndarray):
        n = Omega.shape[1]
        Omega_i, Omega_j = matIndicesFromMask(Omega)
    else:
        raise ValueError('type of Omega not recognized; ' + 
                         'expected tuple of indices or mask array.')

    r = U.shape[1]
    sizeV = n*r

    if sparse:
        sp_mat = _get_sparse_type(sparse_type)
        row_idx = np.repeat(range(Omega_j.size), r)
        col_idx = [np.arange(Omega_j[idx], sizeV, n, dtype=int) 
                   for idx in range(Omega_j.size)]
        col_idx = np.concatenate(col_idx)
        vals = np.concatenate([U[i,:] for i in Omega_i])
        U_op = sp_mat((vals, (row_idx, col_idx)), shape=(Omega_j.size, sizeV))
    else:
        U_op = np.zeros((Omega_j.size, sizeV))
        for idx in range(Omega_j.size):
            i = Omega_i[idx]
            j = Omega_j[idx]
            U_op[idx, j::n] = U[i,:]
    return U_op

def mcFrobSolveRightFactor_cvx(U, M_Omega, mask, **kwargs):
    """
    A solver for the right factor, V, in the problem 
        min FrobNorm( P_Omega(U * V.T - M) )
    where U is an m-by-r matrix, V an n-by-r matrix.
    M_Omega is the set of observed entries in matrix form, while
    mask is a Boolean array with 1/True-valued entries corresponding 
    to those indices that were observed.

    This function is computed using the CVXPY package (and 
    thus is likely to be slower than a straight iterative 
    least squares solver).
    """
    # Options
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    solver = kwargs.get('solver', SCS)
    verbose = kwargs.get('verbose', False)

    if isinstance(verbose, int):
        if verbose > 1:
            verbose = True
        else:
            verbose = False

    # Parameters
    n = mask.shape[1]
    r = U.shape[1]

    Omega_i, Omega_j = matIndicesFromMask(mask)
    
    # Problem
    V_T = Variable((r, n))
    obj = Minimize(cvxnorm(cvxvec((U @ V_T)[Omega_i, Omega_j]) - M_Omega))
    prob = Problem(obj)
    prob.solve(solver=solver, verbose=verbose)
    V = V_T.value.T
    if returnObjectiveValue:
        return (V, prob.value)
    else:
        return V


def mcFrobSolveLeftFactor_cvx(V, M_Omega, mask, **kwargs):
    """
    mcFrobSolveLeftFactor_cvx(V, M_Omega, mask, **kwargs)
    A solver for the left factor, U, in the problem
        min FrobNorm( P_Omega(U * V.T - M) )
    where U is an m-by-r matrix, V an n-by-r matrix.
    M_Omega is the set of observed entries in matrix form, while
    mask is a Boolean array with 1/True-valued entries corresponding 
    to those indices that were observed.

    This function is computed using the CVXPY package (and 
    thus is likely to be slower than a straight iterative 
    least squares solver).
    """
    # Options
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    solver = kwargs.get('solver', SCS)
    verbose = kwargs.get('verbose', False)

    if isinstance(verbose, int):
        if verbose > 1:
            verbose = True
        else:
            verbose = False

    # Parameters
    m = mask.shape[0]
    if V.shape[0] < V.shape[1]:
        # make sure V_T is "short and fat"
        V = V.T
    r = V.shape[1]

    Omega_i, Omega_j = matIndicesFromMask(mask)

    # Problem
    U = Variable((m, r))
    obj = Minimize(cvxnorm(cvxvec((U @ V.T)[Omega_i, Omega_j]) - M_Omega))
    prob = Problem(obj)
    prob.solve(solver=solver, verbose=verbose)
    if returnObjectiveValue:
        return (U.value, prob.value)
    else:
        return U.value


def mcFrobSolveLeftFactor_ls(V, M_Omega, mask, **kwargs):
    r = V.shape[1]
    Vop = matricize_right(V, mask)
    
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    verbose = kwargs.get('verbose', 1)
    x0 = kwargs.get('x0', np.random.rand(Vop.shape[1]))
    
    def frob_error(x, Vop, M_Omega):
        return spnorm((Vop @ x) - M_Omega)

    ls = least_squares(frob_error, x0=x0, kwargs={'Vop' : Vop, 'M_Omega': M_Omega}, verbose=1)
    if returnObjectiveValue:
        return (unvec(ls.x, (-1, r)), ls.cost)
    else:
        return unvec(ls.x, (-1, r))


def mcFrobSolveRightFactor_ls(U, M_Omega, mask, **kwargs):
    """
    mcFrobSolveRightFactor_ls(U, M_Omega, mask, **kwargs)
    solves for the right factor, V, using least squares.
    """
    r = U.shape[1]
    Uop = matricize_left(U, mask)

    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    verbose = kwargs.get('verbose', 1)
    x0 = kwargs.get('x0', np.random.rand(Uop.shape[1]))
    
    def frob_error(x, Uop, M_Omega):
        return spnorm((Uop @ x) - M_Omega)

    ls = least_squares(frob_error, x0=x0,
                       kwargs={'Uop' : Uop, 'M_Omega': M_Omega})
    if returnObjectiveValue:
        return (unvec(ls.x, (-1, r)), ls.cost)
    else:
        return unvec(ls.x, (-1, r))


def altMinSense(Morigin,M_Omega, Omega_mask, r, **kwargs):
    """
    altMinSense(M_Omega, Omega_mask, r, **kwargs)
    The alternating minimization algorithm for a matrix completion
    version of the matrix sensing problem
    
    Input
    max_iters : the maximum allowable number of iterations of the algorithm
    optCond : the optimality conditions that is measured 
              (default: absolute difference)
    optTol : the optimality tolerance used to determine stopping conditions
    solveLeft : a function to solve for the left matrix, Uj, on iteration j
                (default: mcFrobSolveLeftFactor_cvxpy)
    solveRight : a function to solve for the right matrix, Vj, on iteration j
                (default: mcFrobSolveRightFactor_cvxpy)
    solver : which solver to use (for cvxpy only) (default: SCS)
    verbose : 0 (none), 1 (light, default) or 2 (full) level of verbosity

    Ouptut
    U : the left m-by-r factor
    V : the right n-by-r factor
    """
    max_iters = kwargs.get('max_iters', 50)
    method = kwargs.get('method', 'cvx')
    optCond = kwargs.get('optCond', lambda x, y: np.abs(x - y))
    optTol = kwargs.get('optTol', 1e-4)
    solveLeft = kwargs.get('leftSolve', None)
    solveRight = kwargs.get('rightSolve', None)
    opts = kwargs.get('methodOptions', None)
    verbose = kwargs.get('verbose', 1)

    if method == 'lsq':
        solveLeft = mcFrobSolveLeftFactor_ls
        solveRight = mcFrobSolveRightFactor_ls
        if opts is None:
            opts = {'verbose' : verbose}
    elif method == 'cvx':
        solveLeft = mcFrobSolveLeftFactor_cvx
        solveRight = mcFrobSolveRightFactor_cvx
        if opts is None:
            opts = {'solver': SCS, 'verbose': verbose}
        elif opts.get('solver') is None:
            opts['solver'] = SCS

    if not verbose:
        verbose = False
        verbose_solve = False
    elif (verbose is True) or (verbose == 1):
        verbose = True
        verbose_solve = False
    elif (verbose == 2):
        verbose = True
        verbose_solve = True

    m, n = Omega_mask.shape
    # # Create initial guess from unbiased estimator # #
    # Set initial entries of estimator
    unbiased = np.zeros(Omega_mask.shape)
    unbiased[matIndicesFromMask(Omega_mask)] = M_Omega
    # scale entries of estimator by an estimate on the sampling
    # probability p so that this estimator is unbiased
    unbiased /= (M_Omega.size / Omega_mask.size)
    # compute svd of the unbiased estimator
    unbiased_left, unbiased_sing, unbiased_right = np.linalg.svd(unbiased)
    U = unbiased_left[:, :r]
    objPrevious = np.inf
    RMSE=[]
    RMSE.append(np.linalg.norm((unbiased-Morigin), ord='fro') / np.linalg.norm(Morigin, ord='fro'))
    for T in range(max_iters):
        V = solveRight(U, M_Omega, Omega_mask, **opts)
        U, objValue = solveLeft(V, M_Omega, Omega_mask, **opts,
                                returnObjectiveValue=True)
        X=U @ V.T
        RMSE.append(np.linalg.norm((X-Morigin), ord='fro') / np.linalg.norm(Morigin, ord='fro'))
        if optCond(objValue, objPrevious) < optTol:break
        else:
            #if verbose:
            #    print('Iteration {}: Objective = {}'.format(T, objValue), end='\r')
            objPrevious = objValue
    x_coordinate = range(len(RMSE))
    plt.title('GMM-NOISE')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    #plt.yscale('log')
    plt.plot(x_coordinate,RMSE,'-')
    plt.show()
    return U, V

if __name__ == "__main__":
    (Morigin,P_Omega,GMM_noise,Omega)= Init.SetMat(300,150,10,3)
    n1,n2=Morigin.shape[0],Morigin.shape[1]
    Omega_mask=np.zeros((n1,n2))
    #omega=1->1  omega=0->nan
    Omega_mask[Omega]=1
    Omega_mask[np.where(Omega_mask==0)]=np.nan
    M_Omega = masked(Morigin, Omega_mask)
    U_ls, V_ls = altMinSense(Morigin=Morigin,M_Omega=M_Omega,
                         Omega_mask=Omega_mask,
                         r=10, method='lsq')