import matplotlib.pyplot as plt
import SetDataMatrix as Init
import scipy.sparse as ss
import numpy as np
import random
import math


class APA(object):

    def __init__(self,Morigin,P_Omega,Omega,GMM_noise,r,mode):
        self.r=r
        self.mode=mode
        self.Morigin=Morigin
        #Sample index for range(n1*n2)
        self.Omega=Omega
        self.GMM_noise=GMM_noise
        v=GMM_noise[Omega]
        self.T_Cond1=np.linalg.norm(v,1)
        self.T_Cond2=np.linalg.norm(v,2)
        #GMM_noise Remix
        #self.M=self.M+self.GMM_noise
        #Save as sparse
        self.P_Omega=P_Omega

    def Equality(self):
        RMSE=[]
        RMSE.append(np.linalg.norm((self.P_Omega.toarray()-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        X=self.P_Omega.toarray()
        m=self.P_Omega.toarray()
        i=0
        while i<500:
            u,s,v=np.linalg.svd(X)
            Y=self.TruncatedSvd(s,u,v)
            Y_Omega=Y[self.Omega]
            Z=ss.csr_matrix((Y_Omega,self.Omega),shape=(X.shape[0],X.shape[1]))
            X=m+Y-Z.toarray()
            RMSE.append(np.linalg.norm((X-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            i+=1
        x_coordinate = range(len(RMSE))
        plt.title('Noise-free')
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #log
        plt.yscale('log')
        plt.plot(x_coordinate,RMSE,'-')
        plt.show()
        return RMSE
    
    def NORM_1(self,epsilon1):
        RMSE=[]
        RMSE.append(np.linalg.norm(self.P_Omega.toarray()-self.Morigin, ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        X=self.P_Omega.toarray()
        m=self.P_Omega.toarray()
        OmegaT=list(map(list,zip(*self.Omega)))
        i=0
        while i<100:
            u,s,v=np.linalg.svd(X)
            Y=self.TruncatedSvd(s,u,v)
            Y_Omega=Y[self.Omega]
            Z=ss.csr_matrix((Y_Omega,self.Omega),shape=(X.shape[0],X.shape[1]))
            lamb=self.bisection(np.sort(abs((Z.toarray()-m)[self.Omega])),epsilon1)
            X=np.sign(Z.toarray()-m)*np.maximum(abs(Z.toarray()-m)-lamb,0)+Y-Z.toarray()+m
            RMSE.append(np.linalg.norm((X-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            i+=1
        x_coordinate = range(len(RMSE))
        plt.title('GMM-NOISE')
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #lg scale
        #plt.yscale('log')
        plt.plot(x_coordinate,RMSE,'-')
        plt.show()
        return RMSE[-1]

    def NORM_2(self,epsilon2):
        RMSE=[]
        RMSE.append(np.linalg.norm((self.P_Omega.toarray()-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
        X=self.P_Omega.toarray()
        m=self.P_Omega.toarray()
        i=0
        while i<100:
            u,s,v=np.linalg.svd(X)
            #Y=self.TruncatedSvd(s,u,v,r)
            Y=self.TruncatedSvd(s,u,v)
            Y_Omega=Y[self.Omega]
            Z=ss.csr_matrix((Y_Omega,self.Omega),shape=(X.shape[0],X.shape[1]))
            X=epsilon2*(Z.toarray()-m) / np.linalg.norm((Z.toarray()-m),ord='fro')+m+Y-Z.toarray()
            #X=m+(Z.toarray()-m)/np.maximum(np.linalg.norm((Z.toarray()-m),ord=2)**2,self.T_Cond2)+Y-Z.toarray()
            RMSE.append(np.linalg.norm((X-self.Morigin), ord='fro') / np.linalg.norm(self.Morigin, ord='fro'))
            i+=1
        x_coordinate = range(len(RMSE))
        plt.title('GMM-NOISE')
        plt.xlabel('Number of iterations')
        plt.ylabel('RMSE')
        #plt.yscale('log')
        plt.plot(x_coordinate,RMSE,'-')
        plt.show()
        return RMSE[-1]

    def TruncatedSvd(self,sigma, u, v):
        m = len(u)
        n = len(v[0])
        a = np.zeros((m, n))
        for k in range(self.r):
        #for k in range(r):
            uk = u[:, k].reshape(m, 1)
            vk = v[k].reshape(1, n)
            a += sigma[k] * np.dot(uk, vk)
        return a

    def Print2TXT(self):
    # The output part 2
        list=self.RMSE
        output = open('MatrixCompletion\data.txt','w',encoding='gbk')
        for row in list:
	        output.write(str(row))
	        output.write('\n')
        output.close()
        return 0
    
    def KKT_NORM1(self,x,array,epsilon1):
        sum=0.0
        for var in array:
            sum+=np.maximum(var-x,0.0)
        return sum-epsilon1

    def bisection(self,array,epsilon1):
        a=array[0]
        b=array[-1]
        while(1):
            x = (a + b)/2.0
            if self.KKT_NORM1(x,array,epsilon1)==0.0:
                break
            elif ( self.KKT_NORM1(x,array,epsilon1)*self.KKT_NORM1(a,array,epsilon1)<0.0 ):
                b = x
            elif ( self.KKT_NORM1(x,array,epsilon1)*self.KKT_NORM1(a,array,epsilon1)>0.0 ):
                a = x
            elif (a > b):
                return a
            if abs(a - b)< 1e-5:
                break
        return x

    def effi(self):
        RMSE=[[]for i in range(20)]
        for var in range(1,20):
            epsilon=self.T_Cond1*var/10
            RMSE[var-1]=self.NORM_1(epsilon)
        plt.title('Result Analysis')
        x_axix=range(len(RMSE[0]))
        for i in range(19):
            plt.plot(x_axix, RMSE[i], marker='x',label=str((i+1)/10))
        plt.legend()
        plt.xlabel('iteration times')
        plt.ylabel('RMSE')
        plt.show()
    
    def rank(self):
        RMSE=[]
        for var in range(4,16):
            RMSE.append(self.NORM_1(self.T_Cond1*0.7,var))
        plt.title('Estimated rank')
        x_axix=range(4,16)
        plt.plot(x_axix, RMSE, marker='x')
        plt.legend()
        plt.xlabel('Rank')
        plt.ylabel('RMSE')
        plt.show()

if __name__ == "__main__":
    #SNR
    '''
    RMSE=[]
    for var in range(8):
        obj=APA(150,300,10,1,var*3)
        RMSE.append(obj.NORM_1(obj.T_Cond1*0.7))
    #plt.title('Estimated rank')
    x_axix=range(0,24,3)
    plt.yscale('log')
    plt.plot(x_axix, RMSE, marker='x')
    plt.legend() # 显示图例
    plt.xlabel('SNR')
    plt.ylabel('RMSE')
    plt.show()
    '''
    #alpha
    #obj.effi()
    #estimated rank
    #obj.rank()
    (Morigin,P_Omega,GMM_noise,Omega)= Init.SetMat(300,150,10,12)
    obj=APA(Morigin,P_Omega,Omega,GMM_noise,10,1)
    if obj.mode==0:
        obj.Equality()
    elif obj.mode==1:
        obj.NORM_1(obj.T_Cond1*0.7)
    elif obj.mode==2:
        obj.NORM_2(obj.T_Cond2*0.8)
