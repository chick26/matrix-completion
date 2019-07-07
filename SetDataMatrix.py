from sparsesvd import sparsesvd
import matplotlib.pyplot as plt
import scipy.sparse as ss
import numpy as np
import random
import math

def SetMat(n1,n2,r,snr):
    #M(n1*r)M(r*n2)-->M(n1*n2) with rank r
    #M=np.dot(np.random.normal(0,1,(n1,r)),np.random.normal(0,1,(r,n2)))
    M = np.dot(np.random.randn(n1,r),np.random.randn(r,n2))
    Morigin=M
    #Sample index for range(n1*n2)
    Omega_index=random.sample(range(n1*n2),round(n1*n2*0.45))
    #Set up n1*n2 Find index of Omega
    Omega=np.unravel_index(Omega_index,(n1,n2))
    #(array([3, 0, 4, 1, 2, 1, 4, 2, 4, 1, 0], dtype=int64), array([2, 3, 1, 4, 2, 3, 0, 1, 2, 1, 4], dtype=int64))
    #GMM & T_Cond
    GMM_noise=GMM(M,Omega,n1,n2,0,snr)
    #GMM_noise Remix
    M=M+GMM_noise
    M_Omega=M[Omega]
    #Save as sparse
    P_Omega=ss.csr_matrix((M_Omega,Omega),shape=(n1,n2))
    return Morigin,P_Omega,GMM_noise,Omega

def GMM(M,Omega,n1,n2,mu,snr):
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
    #GMM_SMALL=np.random.normal(mu, sigma, (n1,n2))
    GMM_SMALL=np.random.randn(n1,n2)*sigma
    #GMM_BIG=np.random.normal(mu, 10.0*sigma, (n1,n2))
    GMM_BIG=np.random.randn(n1,n2)*10*sigma
    for var in OmegaT:
        GMM_SMALL[var[0],var[1]]=GMM_BIG[var[0],var[1]]
    return GMM_SMALL