import matplotlib.pyplot as plt
import scipy.sparse.linalg as ssl
from scipy.sparse.linalg import norm
import scipy.sparse as ss
from sparsesvd import sparsesvd
import SetDataMatrix as Init
import numpy as np
import random
import math

class SVT(object):

    def __init__(self,Morigin,P_Omega,Omega,GMM_noise,r):
        self.r=r
        self.Morigin=Morigin
        #Set up n1*n2 Find index of Omega
        self.Omega=Omega
        #(array([3, 0, 4, 1, 2, 1, 4, 2, 4, 1, 0], dtype=int64), array([2, 3, 1, 4, 2, 3, 0, 1, 2, 1, 4], dtype=int64))
        #GMM & T_Cond
        self.GMM_noise=GMM_noise
        #Save as sparse
        self.P_Omega=P_Omega
    
    def TruncatedSvd(self,sigma,u,v,r,tau):
        m = len(u)
        n = len(v[0])
        a = np.zeros((m, n))
        for k in range(r):
            uk = u[:, k].reshape(m, 1)
            vk = v[k].reshape(1, n)
            a += (sigma[k]-tau) * np.dot(uk, vk)
        return a

    def _solve(self):
        n1,n2=self.Morigin.shape[0],self.Morigin.shape[1]
        RMSE=[]
        incre=4
        tol=0.2
        delta=1.8
        tau = 20*math.sqrt(n1*n2)
        s,r,i=0,0,0
        X=self.P_Omega.toarray()
        normProjM=np.linalg.norm(X,ord=2)
        k0=np.ceil(tau / (delta*normProjM))
        Y=k0*delta*X

        while i<100:
            s=r+1
            #s = min(s+incre,self.n1,self.n2)
            while True:
                u1,s1,v1 = sparsesvd(ss.csc_matrix(Y),s)
                if s1[-1] <= tau : break
                s = min(s+incre,n1,n2)
                if s == min(n1,n2): break

            r = np.sum(s1>tau)
            U = u1.T[:,:r]
            V = v1[:r,:]
            S = s1[:r]-tau
            x = (U*S).dot(V)
            x_omega = ss.csr_matrix((x[self.Omega],self.Omega),shape=(n1,n2))
            diff = self.P_Omega-x_omega
            Y += delta*diff
            #if norm(x_omega-self.P_Omega)/norm(self.P_Omega) < tol:break
            RMSE.append(np.linalg.norm((x-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            i+=1
        x_coordinate = range(len(RMSE))
        plt.title('Noise-free')
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #log
        #plt.yscale('log')
        plt.plot(x_coordinate,RMSE,'-')
        plt.show()
        return RMSE

if __name__ == "__main__":
    (Morigin,P_Omega,GMM_noise,Omega)= Init.SetMat(300,150,10,12)
    obj=SVT(Morigin,P_Omega,Omega,GMM_noise,10)
    obj._solve()

