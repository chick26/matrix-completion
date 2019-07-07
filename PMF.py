import numpy as np
import random
import math
import scipy.sparse as ss
import matplotlib.pyplot as plt

def init(n1,n2,r,p,snr):
    #M(n1*r)M(r*n2)-->M(n1*n2) with rank r
    #self.M=np.dot(np.random.normal(0,1,(n1,r)),np.random.normal(0,1,(r,n2)))
    M = np.dot(np.random.randn(n1,r),np.random.randn(r,n2))
    Morigin=M
    #Sample index for range(n1*n2)
    Omega_index=random.sample(range(n1*n2),round(n1*n2*p))
    #Set up n1*n2 Find index of Omega
    Omega=np.unravel_index(Omega_index,(n1,n2))
    #(array([3, 0, 4, 1, 2, 1, 4, 2, 4, 1, 0], dtype=int64), array([2, 3, 1, 4, 2, 3, 0, 1, 2, 1, 4], dtype=int64))
    #GMM
    GMM_noise=GMM(n1,n2,0,6,M,Omega)
    v=GMM_noise[Omega]
    #GMM_noise Remix
    M=M+GMM_noise
    M_Omega=M[Omega]
    mask=np.zeros((n1,n2))
    mask[Omega]=1
    #Save as sparse
    P_Omega=ss.csr_matrix((M_Omega,Omega),shape=(n1,n2))
    return mask, P_Omega, Morigin

def GMM(n1,n2,mu,snr,M,Omega):
    # Get M_omega from Origin Matrix sampled by Omega
    M_omega=ss.csr_matrix((M[Omega],Omega),shape=(n1,n2)).toarray()
    #Calculate ||M||_F
    M_f=np.linalg.norm(M_omega, ord='fro')
    #Calculate sigma_1
    sigma_v=math.sqrt(M_f**2/(len(Omega[0])*math.pow(10.0,snr/10.0)))
    sigma=math.sqrt(1.0/10.9)*sigma_v
    #Obtain GMM Sample index by sampling from n1*n2 by 10%
    BIG_indexnum=random.sample(range(n1*n2),round(n1*n2*0.1))
    BIG_index=np.unravel_index(BIG_indexnum,(n1,n2))
    OmegaT=list(map(list,zip(*BIG_index)))
    #Obtain two-term Gaussian Matrix
    #GMM_SMALL=np.random.normal(mu, sigma, (self.n1,self.n2))
    GMM_SMALL=np.random.randn(n1,n2)*sigma
    #GMM_BIG=np.random.normal(mu, 10.0*sigma, (self.n1,self.n2))
    GMM_BIG=np.random.randn(n1,n2)*10*sigma
    for var in OmegaT:
        GMM_SMALL[var[0],var[1]]=GMM_BIG[var[0],var[1]]
    return GMM_SMALL

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
    '''
    U = np.random.randn(5, 5)
    V = np.random.randn(5, 5)
    R = np.random.randn(5, 5) + np.dot(U, V.T)
    '''
    mask,P_Omega,Morigin=init(300,150,10,0.45,12)
    R_hat = pmf_solve(P_Omega.toarray(), mask, 10, 1e-2)
    #print(np.linalg.norm(np.subtract(R_hat,Morigin), ord='fro') / np.linalg.norm(Morigin, ord='fro'))