from sparsesvd import sparsesvd
import matplotlib.pyplot as plt
import scipy.sparse as ss
import numpy as np
import random
import math

'''
    min_X  ||X.*A = Y.*A||_F^2 + r*||X||_Sp^p
    matrix completion problem
function [X, obj, st]=OtraceEIC_my(Y, A, p, r, X0, st)
'''
class EIC(object):

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
    
    def _solve_(self):
        r,p=1,0.5
        iter=0
        A=np.zeros((self.n1,self.n2))
        A[self.Omega]=1
        Y=self.P_Omega.toarray()
        X=Y
        temp=np.dot(Y.T,Y)
        st=0.002*max(abs(np.diag(temp)))
        [U,S,V]=np.linalg.svd(X)
        #s=np.zeros((self.n2,1))
        s=np.zeros(self.n1)
        for i in range(len(S)):
            s[i]=S[i]
        D=p/2*np.dot(np.dot(U,np.diag((s*s+st)**(p/2-1))),U.T)
        #D = p/2*(np.dot(X.T,X)+st*np.eye(self.n2))**(p/2-1)
        RMSE=[]
        RMSE.append(np.linalg.norm((X-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        while True:
            '''
            for i in range(self.n1):
                Ai=A[i,:]
                Yi=Y[i,:]
                AY=Ai*Yi
                Aid=np.diag(Ai)
                X[i,:]=np.dot(AY,np.linalg.inv(Aid+r*D))
            '''
            for i in range(self.n2):
                Ai=A[:,i]
                Yi=Y[:,i]
                AY=Ai*Yi
                Aid=np.diag(Ai)
                X[:,i]=np.dot(np.linalg.inv(Aid+r*D),AY)
            iter+=1
            [U,S,V]=np.linalg.svd(X)
            s=np.zeros(self.n1)
            for i in range(len(S)):
                s[i]=S[i]
            #D=p/2*np.sign(np.dot(Y.T,Y))*(abs(np.dot(Y.T,Y))**(p/2-1))
            D=p/2*np.dot(np.dot(U,np.diag((s*s+st)**(p/2-1))),U.T)
            #D = p/2*(np.dot(X.T,X)+st*np.eye(self.n2))**(p/2-1)
            RMSE.append(np.linalg.norm((X-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            if iter==40:break
        x_coordinate = range(len(RMSE))
        plt.title('GMM-NOISE')
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #plt.yscale('log')
        plt.plot(x_coordinate,RMSE,'-')
        plt.show()

if __name__ == "__main__":
    obj=EIC(300,150,10,12)
    obj._solve_()
            