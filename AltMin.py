import math
import random
import numpy as np
import cvxpy as cvx 
import scipy.sparse as ss
from scipy.linalg import orth

class AltMin(object):

    def __init__(self,n1,n2,r,T,p,snr):
        self.n1=n1
        self.n2=n2
        self.r=r
        self.T=T
        self.p=p
        #M(n1*r)M(r*n2)-->M(n1*n2) with rank r
        #self.M=np.dot(np.random.normal(0,1,(n1,r)),np.random.normal(0,1,(r,n2)))
        self.M = np.dot(np.random.randn(n1,r),np.random.randn(r,n2))
        self.Morigin=self.M
        #Sample index for range(n1*n2)
        Omega_index=random.sample(range(n1*n2),round(n1*n2*self.p))
        #Set up n1*n2 Find index of Omega
        self.Omega=np.unravel_index(Omega_index,(n1,n2))
        #(array([3, 0, 4, 1, 2, 1, 4, 2, 4, 1, 0], dtype=int64), array([2, 3, 1, 4, 2, 3, 0, 1, 2, 1, 4], dtype=int64))
        #GMM & T_Cond
        self.GMM_noise=self.GMM(0,snr)
        v=self.GMM_noise[self.Omega]
        #GMM_noise Remix
        #self.M=self.M+self.GMM_noise
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

    def _split_omega(self):
        omegas = [np.zeros(self.M.shape) for t in range(2 * self.T + 1)]
        omega=np.zeros((self.n1,self.n2))
        omega[self.Omega]=1
        for i in range(self.n1):
            for j in range(self.n2):
                idx = random.randint(0, 2 * self.T)
                omegas[idx][i, j] = omega[i, j]
        return omegas
    
    def _init_U(self,omegas,mu):
        M=self.P_Omega.toarray()
        M = M / self.p
        U, S, V = np.linalg.svd(np.multiply(M,omegas[0]))#,full_matrices=False)
        #U_hat=np.multiply(U,Temp)
        #U_hat=U[:,0:self.r]
        U_hat = U.copy()
        clip_threshold = 2 * mu * math.sqrt(self.r / max(M.shape))
        U_hat[U_hat > clip_threshold] = 0
        U_hat = orth(U_hat)
        '''
        print("|U_hat-U|_F/|U|_F:",
            np.linalg.norm(np.subtract(U_hat, U), ord='fro') / np.linalg.norm(
                U, ord='fro'))
        '''
        return U_hat

    def _get_V_from_U(self,U,M,omega):
        column = M.shape[1]
        rank = U.shape[1]
        V = np.empty((rank, column), dtype=M.dtype)
        #V=cvx.Variable((rank,column))
        U_ = U.copy()
        
        #V = np.linalg.lstsq(U_, M, rcond=None)[0]
        V = np.linalg.lstsq(U_, np.multiply(M,omega), rcond=None)[0]
        '''
        print(np.multiply((np.dot(U,V.T)-M),omega))
        objection=cvx.Minimize(cvx.norm(np.multiply((np.dot(U,V.T)-M),omega),'fro'))
        prob=cvx.Problem(objection)
        '''
        return V.T#prob.solve()

    def _get_U_from_V(self,V, M, omega):
        row = M.shape[0]
        rank = V.shape[1]
        U_ = np.empty((rank,row), dtype=M.dtype)

        V_ = V.copy()
        U = np.linalg.lstsq(V_, (np.multiply(M,omega)).T, rcond=None)[0]
        '''
        U=cvx.Variable((row,rank))
        objection=cvx.Minimize(cvx.norm(np.multiply((np.dot(U,V.T))-M,omega),'fro'))
        prob=cvx.Problem(objection)
        '''
        return U#prob.solve()

    def _solve(self):
        RMSE=[]
        omega=np.zeros((self.n1,self.n2))
        omega[self.Omega]=1
        omegas = self._split_omega()
        U = self._init_U(omegas, 100)
        M=self.P_Omega.toarray()
        print('')
        for t in range(self.T):
            V = self._get_V_from_U(U, M, omegas[t + 1])
            U = self._get_U_from_V(V, M, omegas[self.T + t + 1])
            X=np.dot(U,V.T)
            RMSE.append(np.linalg.norm(np.subtract(X,self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        assert V is not None
        return X,RMSE
    
if __name__ == "__main__":
    AltMin=AltMin(300,150,10,12,0.1,1) #n1,n2,r,T,P_sampling,snr
    X,RMSE=AltMin._solve()
    print(RMSE)
    