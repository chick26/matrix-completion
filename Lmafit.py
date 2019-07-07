import numpy as np
import math
import random
import scipy.sparse as ss
import matplotlib.pyplot as plt

class Lmafit(object):

    def __init__(self,n1,n2,k,snr):
        self.n1=n1
        self.n2=n2
        self.k=k
        self.tol=1.25e-4
        self.DoQR=True
        self.est_rank=10
        self.rank_max=max(math.floor(0.1*min(n1,n2)),2*k)
        self.rank_min=1
        self.rk_inc=1
        self.rk_jump=10
        self.init=0
        self.save_res=0
        self.M = np.dot(np.random.randn(n1,k),np.random.randn(k,n2))
        self.Morigin=self.M
        #Sample index for range(n1*n2)
        Omega_index=random.sample(range(n1*n2),round(n1*n2*0.45))
        #Set up n1*n2 Find index of Omega
        self.Omega=np.unravel_index(Omega_index,(n1,n2))
        #(array([3, 0, 4, 1, 2, 1, 4, 2, 4, 1, 0], dtype=int64), array([2, 3, 1, 4, 2, 3, 0, 1, 2, 1, 4], dtype=int64))
        #GMM & T_Cond
        self.GMM_noise=self.GMM(0,snr)
        #GMM_noise Remix
        #self.M=self.M+self.GMM_noise
        self.M_Omega=self.M[self.Omega]
        self.datanrm=max(1,np.linalg.norm(self.M_Omega,2))
        self.Zfull=(len(self.M_Omega)/(n1*n2)>0.2) or k>0.02*min(n1,n2) or n1*n2<5e5
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

    def _solve_(self):
        Z=self.P_Omega.toarray()
        Known=self.Omega
        if self.init==0:
            X=np.zeros((self.n1,self.k))
            Y=np.eye(self.k,self.n2)
            Res=self.M_Omega
            res=self.datanrm
        if self.est_rank==1: rank_max=min(self.rank_max,self.k)
        #if self.n1>self.n2: Z=Z.T

        #parameter for alf
        alf,increment,itr_rank=0,1,0
        RMSE=[]
        while True:
            X0,Y0,Res0,res0=X,Y,Res,res
            itr_rank+=1
            if self.Zfull:
                Z0=Z
                X=np.dot(Z,Y.T)
                if self.est_rank==1:
                    X,R=np.linalg.qr(X)
                    Y=np.dot(X.T,Z)
                elif self.DoQR:
                    X,R=np.linalg.qr(X)
                    Y=np.dot(X.T,Z)
                else:
                    Xt=X.T
                    Y=np.solve(np.dot(Xt,X),np,dot(Xt,Z))
                Z=np.dot(X,Y)
                Res=self.M_Omega-Z[Known]
            res=np.linalg.norm(Res,2)
            relres=res/self.datanrm
            ratio=res/res0
            reschg=abs(1-res/res0)
            RMSE.append(res/np.linalg.norm(self.Morigin, ord='fro'))
            if itr_rank==100: break
        x_coordinate = range(len(RMSE))
        plt.title('GMM-NOISE')
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #lg scale
        #plt.yscale('log')
        plt.plot(x_coordinate,RMSE,'-')
        plt.show()
        return 0

if __name__ == "__main__":
    obj=Lmafit(300,150,10,12)
    obj._solve_()
