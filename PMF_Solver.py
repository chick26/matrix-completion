import numpy as np
import random
import math
import scipy.sparse as ss
import matplotlib.pyplot as plt
import SetDataMatrix as Init

def Mask(n1,n2,Omega):
    mask=np.zeros((n1,n2))
    mask[Omega]=1
    return mask

def pmf_solve(A, mask, k, mu, epsilon=1e-3, max_iterations=30):
    """
    Solve probabilistic matrix factorization using alternating least squares.
    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.
    Parameters:
    -----------
    k : integer
        how many factors to use
    mu : float
        hyper-parameter penalizing norm of factored U, V
    epsilon : float
        convergence condition on the difference between iterative results
    max_iterations: int
        hard limit on maximum number of iterations
    Returns:
    --------
    X: m x n array
        completed matrix
    """
    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)
    RMSE=[]
    RMSE.append(np.linalg.norm(np.subtract(A,Morigin), ord='fro') / np.linalg.norm(Morigin, ord='fro'))
    for _ in range(max_iterations):

        for i in range(m):
            U[i] = np.linalg.solve(np.linalg.multi_dot([V.T, C_u[i], V]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([V.T, C_u[i], A[i,:]]))

        for j in range(n):
            V[j] = np.linalg.solve(np.linalg.multi_dot([U.T, C_v[j], U]) +
                                   mu * np.eye(k),
                                   np.linalg.multi_dot([U.T, C_v[j], A[:,j]]))

        X = np.dot(U, V.T)
        #mean_diff = np.linalg.norm(X - prev_X) / m / n
        prev_X = X
        RMSE.append(np.linalg.norm(np.subtract(X,Morigin), ord='fro') / np.linalg.norm(Morigin, ord='fro'))
    x_coordinate = range(len(RMSE))
    plt.title('GMM-NOISE')
    plt.xlabel('Number of iterations')
    plt.ylabel('RMSE')
    #lg scale
    #plt.yscale('log')
    plt.plot(x_coordinate,RMSE,'-')
    plt.show()    
    return X

if __name__ == "__main__":
    (Morigin,P_Omega,GMM_noise,Omega)= Init.SetMat(300,150,10,12)
    mask=Mask(300,150,Omega)
    R_hat = pmf_solve(P_Omega.toarray(), mask, 10, 1e-2)
    #print(np.linalg.norm(np.subtract(R_hat,Morigin), ord='fro') / np.linalg.norm(Morigin, ord='fro'))