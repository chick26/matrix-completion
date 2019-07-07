from sparsesvd import sparsesvd
import matplotlib.pyplot as plt
import SetDataMatrix as Init
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

    def __init__(self,Morigin,P_Omega,GMM_noise,Omega):
        self.Morigin=Morigin
        self.Omega=Omega
        self.GMM_noise=GMM_noise
        self.P_Omega=P_Omega
    
    def _solve_(self):
        n1,n2=self.Morigin.shape[0],self.Morigin.shape[1]
        r,p=1,0.5
        iter=0
        A=np.zeros((n1,n2))
        A[self.Omega]=1
        Y=self.P_Omega.toarray()
        X=Y
        temp=np.dot(Y.T,Y)
        st=0.002*max(abs(np.diag(temp)))
        [U,S,V]=np.linalg.svd(X)
        #s=np.zeros((self.n2,1))
        s=np.zeros(n1)
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
            for i in range(n2):
                Ai=A[:,i]
                Yi=Y[:,i]
                AY=Ai*Yi
                Aid=np.diag(Ai)
                X[:,i]=np.dot(np.linalg.inv(Aid+r*D),AY)
            iter+=1
            [U,S,V]=np.linalg.svd(X)
            s=np.zeros(n1)
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
    (Morigin,P_Omega,GMM_noise,Omega)= Init.SetMat(300,150,10,12)
    obj=EIC(Morigin,P_Omega,GMM_noise,Omega)
    obj._solve_()
            