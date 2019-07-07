import matplotlib.pyplot as plt
import scipy.sparse.linalg as ssl
from scipy.sparse.linalg import norm
import scipy.sparse as ss
from sparsesvd import sparsesvd
import numpy as np
import random
import math

class SVT(object):

    def __init__(self,n1,n2,r,snr):
        self.n1=n1
        self.n2=n2
        self.r=r
        #M(n1*r)M(r*n2)-->M(n1*n2) with rank r
        #self.M=np.dot(np.random.normal(0,1,(n1,r)),np.random.normal(0,1,(r,n2)))
        self.M = np.dot(np.random.randn(n1,r),np.random.randn(r,n2))
        self.Morigin=self.M
        #Sample index for range(n1*n2)
        Omega_index=random.sample(range(n1*n2),round(n1*n2*0.45))
        #Set up n1*n2 Find index of Omega
        self.Omega=np.unravel_index(Omega_index,(n1,n2))
        #(array([3, 0, 4, 1, 2, 1, 4, 2, 4, 1, 0], dtype=int64), array([2, 3, 1, 4, 2, 3, 0, 1, 2, 1, 4], dtype=int64))
        #GMM & T_Cond
        self.GMM_noise=self.GMM(0,snr)
        #GMM_noise Remix
        self.M=self.M+self.GMM_noise
        self.M_Omega=self.M[self.Omega]
        #Save as sparse
        self.P_Omega=ss.csr_matrix((self.M_Omega,self.Omega),shape=(n1,n2))

    def GMM(self,mu,snr):
        # Get M_omega from Origin Matrix sampled by Omega
        M_omega=ss.csr_matrix((self.M[self.Omega],self.Omega),shape=(self.n1,self.n2)).toarray()
        #Calculate ||M||_F
        M_f=np.linalg.norm(M_omega, ord='fro')
        #Calculate sigma_1
        sigma_v=math.sqrt(M_f**2/(len(self.Omega[0])*math.pow(10.0,snr/10.0)))
        sigma=math.sqrt(1.0/10.9)*sigma_v
        #Obtain GMM Sample index by sampling from n1*n2 by 10%
        BIG_indexnum=random.sample(range(self.n1*self.n2),round(self.n1*self.n2*0.1))
        BIG_index=np.unravel_index(BIG_indexnum,(self.n1,self.n2))
        OmegaT=list(map(list,zip(*BIG_index)))
        #Obtain two-term Gaussian Matrix
        #GMM_SMALL=np.random.normal(mu, sigma, (self.n1,self.n2))
        GMM_SMALL=np.random.randn(self.n1,self.n2)*sigma
        #GMM_BIG=np.random.normal(mu, 10.0*sigma, (self.n1,self.n2))
        GMM_BIG=np.random.randn(self.n1,self.n2)*10*sigma
        for var in OmegaT:
            GMM_SMALL[var[0],var[1]]=GMM_BIG[var[0],var[1]]
        return GMM_SMALL
    
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
        RMSE=[]
        incre=4
        tol=0.2
        delta=1.8
        tau = 20*math.sqrt(self.n1*self.n2)
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
                s = min(s+incre,self.n1,self.n2)
                if s == min(self.n1,self.n2): break

            r = np.sum(s1>tau)
            U = u1.T[:,:r]
            V = v1[:r,:]
            S = s1[:r]-tau
            x = (U*S).dot(V)
            x_omega = ss.csr_matrix((x[self.Omega],self.Omega),shape=(self.n1,self.n2))
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
    obj=SVT(150,300,10,6)
    obj._solve()

