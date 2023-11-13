"""
Created on   Oct. 2023 
@ max capacity by RIS , Ga ,Gd, Test under different setting, for loops for  user number K 
@authors:Xin He, Lei Huang, Jiangzhou Wang, and Yi Gong, "Learn to Optimize RIS Aided Hybird Beamforming With Strong Out-of-Distribution Generalization"
You can  change the parameter "dmode" to get different test results
"""
 
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np 
import matplotlib.pyplot as plt
import torch as t 
import math
from math import e
import time  
import pickle  
# ---- --the default parameters--------------
K=10 #UE no
R = 128      # RIS no
M = 64     # Tx antennas
L= 16 #RF number
SNR=   120 ; #dB
Pt=10**(SNR/10)
stepSize=50*1e-2;      
Xu=100;Xr=100#UE RIS locations
test_size = 100 #test numbers
 
# ---- load unrolled GP model----  -------------- 
# file = open('./Figs/muParaSNR30.pkl', 'rb')
# file = open('./Figs/muLargeSNR130.pkl', 'rb')
file = open('muLargeSNR130Eno30Tno100.pkl', 'rb')#the trained model 
muA2,muD2,mu2= pickle.load(file); file.close();  
  
# ----------------the  different settings by changing the "dmode" ----------------
dmode= 6 #1=change UE no 2=change antenna no 3=change RIS no  4=change SNR  5= change users location 6=change RIS locations
 
if dmode==1:#1=UE no  
    lopPara=np.linspace( 1, L, L)
elif dmode==2:#  2=antenna no 
    lopPara=[32,64,128,256]
elif dmode==3:# 3= RIS no
    lopPara=[64,128,256,512] 
elif dmode==4:# 4= Pt change SNR dB
    lopPara=np.linspace( 90, 130, 5)
elif dmode==5:# 5= change users location
    lopPara=np.linspace(50.0, 1e3, 5)
elif dmode==6:# 6=change RIS locations
    lopPara=np.linspace( 0, 1e2, 5)

#----------differnt algorithms-------------------- 
def GP(mu , hr,h0,hd,L,Pt):
    # ------- the GP method   --------- 
    Tno=hr.size()[0]; n=hr.size()[1]; R=hr.size()[2];  M=h0.size()[2] 
    # ---- initializing variables; initializing theta as random phase
    theta = t.ones(Tno, R,dtype=t.cfloat) #fixed as 1
    Ga = t.randn(Tno, M, L,dtype=t.cfloat)  
    Gd = t.randn(Tno, L, n,dtype=t.cfloat)  
    gradAll=t.ones(Tno, R,dtype=t.cfloat) 
    G=Ga@Gd
    G2=G@t.transpose(G, 1, 2).conj()
    H=t.ones(Tno, n,M,dtype=t.cfloat)#initialization
    for j in range(Tno):#matrix 2 tensor
        H[j]=hr[j]@t.diag(theta[j])@h0[j] +hd[j] #compont channel  
    obj0=0    
    while 1:
        grada=t.transpose(H,1,2).conj()@t.linalg.solve(t.eye(n).reshape(( 1, n, n))+H@G2@t.transpose(H,1,2).conj(), H@G@t.transpose(Gd,1,2).conj())
        Ga=Ga+mu * grada 
        Ga = Ga  /t.abs(Ga)
        #----Gd
        G=Ga@Gd
        G2=G@t.transpose(G, 1, 2).conj()
        gradd=t.transpose(Ga,1,2).conj()@t.transpose(H,1,2).conj()@t.linalg.solve(t.eye(n).reshape(( 1, n, n))+H@G2@t.transpose(H,1,2).conj(), H@G)
        Gd=Gd+mu * gradd 
        Gd=t.sqrt(Pt / ( t.linalg.matrix_norm(Ga @ Gd , ord='fro') ** 2 )).reshape(Tno,1,1) *Gd
        # ---------- RIS --------------- 
        G=Ga@Gd
        G2=G@t.transpose(G, 1, 2).conj()  
        grad=t.transpose(hr,1,2).conj()@t.linalg.solve(t.eye(n).reshape(( 1, n, n))+H@G2@t.transpose(H,1,2).conj(), H@G2@t.transpose(h0,1,2).conj())
        # grad=t.randn(Tno, R,R,dtype=t.cfloat)
        for k2 in range(Tno):#tensor 2 matrix 2 
            gradAll[k2]  = t.diag( grad[k2])
        # gradient ascent 
        thetaT  = theta   + mu * gradAll    
        # projection
        theta = thetaT  /t.abs(thetaT  ) 
        for k3 in range(Tno):#matrix 2 tensor
            H[k3]=hr[k3]@t.diag(theta[k3])@h0[k3] +hd[k3] #compont channel 
        obj1=sum_capcity( G2, H,n,  Tno) 
        if abs(obj1-obj0)/obj1<=1e-4:
            break
        obj0=obj1
    return obj1
def FullDigit(  hr,h0,hd,mu,Pt ):
    # ------- the full digital BF(water filling)+RIS   ---------  
    # h - channel realization 
    # num_of_iter - num of iters of the PGA algorithm
    Tno=hr.size()[0]; n=hr.size()[1]; R=hr.size()[2];  M=h0.size()[2] 
    # ---- initializing variables; initializing theta as random phase
    thetat = t.randn(Tno, R,dtype=t.cfloat) 
    theta = thetat/t.abs(thetat) ;gradAll=t.randn(Tno, R,dtype=t.cfloat)
    H=t.randn(Tno, n,M,dtype=t.cfloat)#initialization
    for j in range(Tno):#matrix 2 tensor
        H[j]=hr[j]@t.diag(theta[j])@h0[j] +hd[j] #compont channel 
    obj0=0 ;lop=0
    while 1:
        # ---------- Water-filling---------------  
        U, S, Vh = t.linalg.svd(H, full_matrices=True )   #, full_matrices=True
        V = Vh.mH 
        S1=S.detach().numpy()
        sigama=t.ones([Tno,n,n],dtype=t.cfloat)
        for i in range(Tno):#power allocation for each channel
            power= Pt*WaterF(S1[i]**2*Pt,Pt/Pt)#SNR,total power;normalize power to avoid the underflow issue
            sigama[i,:,:]=t.diag(t.tensor(np.sqrt(power)))  
        G=V[:,:,:n]@sigama#not the Water filling solution
        # ---------- RIS --------------- 
        G2=G@t.transpose(G, 1, 2).conj() 
        for _ in range(4):
            grad=t.transpose(hr,1,2).conj()@t.linalg.solve(t.eye(n).reshape(( 1, n, n))+H@G2@H.mH, H@G2@ h0.mH)
            for j in range(Tno):#tensor 2 matrix 2 
                gradAll[j]  = t.diag( grad[j])
            # gradient ascent
            thetaT  = theta   + mu* gradAll   
            # projection
            theta = thetaT /t.abs(thetaT ) 
            for j in range(Tno):#matrix 2 tensor
                H[j]=hr[j]@t.diag(theta[j])@h0[j] +hd[j] #compont channel  
        obj1= sum_capcity( G2, H,n,  Tno) 
        lop+=1
        if abs(obj1-obj0)/obj1<=1e-3:
            break
        obj0=obj1
    return obj1 

def WaterF( S,Pt):
    alfa=1/S; pSum=1e19
    matr=np.matrix([np.zeros(len(S)), alfa])
    low=0; high=1/min(alfa)  
    while abs(pSum-Pt)>=1e-4:#underflow is possible
        mu=(low+high)/2;
        matr[1]= 1/mu-alfa 
        power=matr.max(0) ; pSum=power.sum()
        if pSum<=Pt:
            high=mu;
        else:
            low=mu; 
    return power.A[0]

def Unroll(muA,muD,mu , hr,h0,hd,L,Pt):
    # ------- the unrolled GP method --------- 
    Tno=hr.size()[0]; n=hr.size()[1]; R=hr.size()[2];  M=h0.size()[2] 
    lop1=muA.size()[0]; lop2=muA.size()[1]; lop3=mu.size()[1]; lop4=muA.size()[2];
    # ---- initializing variables; initializing theta as random phase
    theta = t.ones(Tno, R,dtype=t.cfloat) #fixed as 1
    Ga = t.randn(Tno, M, L,dtype=t.cfloat)  
    Gd = t.randn(Tno, L, n,dtype=t.cfloat)  
    gradAll=t.ones(Tno, R,dtype=t.cfloat) 
    G=Ga@Gd
    G2=G@t.transpose(G, 1, 2).conj()
    H=t.ones(Tno, n,M,dtype=t.cfloat)#initialization
    for j in range(Tno):#matrix 2 tensor
        H[j]=hr[j]@t.diag(theta[j])@h0[j] +hd[j] #compont channel   
    for i in range(lop1):
        
        for j1 in range(lop2):#_____analog and digital BF___
            for k in range(lop4): 
                grada=t.transpose(H,1,2).conj()@t.linalg.solve(t.eye(n).reshape(( 1, n, n))+H@G2@t.transpose(H,1,2).conj(), H@G@t.transpose(Gd,1,2).conj())
                Ga=Ga+muA[i,j1,k]* grada 
                Ga = Ga  /t.abs(Ga)
                G=Ga@Gd
                G2=G@t.transpose(G, 1, 2).conj()
            gradd=t.transpose(Ga,1,2).conj()@t.transpose(H,1,2).conj()@t.linalg.solve(t.eye(n).reshape(( 1, n, n))+H@G2@t.transpose(H,1,2).conj(), H@G)
            Gd=Gd+muD[i,j1]* gradd 
        Gd=t.sqrt(Pt / ( t.linalg.matrix_norm(Ga @ Gd , ord='fro') ** 2 )).reshape(Tno,1,1) *Gd
        # ---------- RIS --------------- 
        G=Ga@Gd
        G2=G@t.transpose(G, 1, 2).conj() 
        for j2 in range(lop3): 
            grad=t.transpose(hr,1,2).conj()@t.linalg.solve(t.eye(n).reshape(( 1, n, n))+H@G2@t.transpose(H,1,2).conj(), H@G2@t.transpose(h0,1,2).conj())
            # grad=t.randn(Tno, R,R,dtype=t.cfloat)
            for k2 in range(Tno):#tensor 2 matrix 2 
                gradAll[k2]  = t.diag( grad[k2])
            # gradient ascent 
            thetaT  = theta   + mu[i,j2]* gradAll    
            # projection
            theta = thetaT  /t.abs(thetaT  ) 
            for k3 in range(Tno):#matrix 2 tensor
                H[k3]=hr[k3]@t.diag(theta[k3])@h0[k3] +hd[k3] #compont channel  
                 
        
    return G2,H 
def sum_capcity( G2, H,n,  batch_size):#
    sign,aa=t.linalg.slogdet( t.eye(n).reshape(( 1, n, n)) + H @ G2  @ t.transpose(H, 1, 2).conj()) 
    return  sum(aa/np.log(2)) / batch_size  
    # return  sum(t.log2( (t.eye(n).reshape(( 1, n, n)) + H @ G2  @ t.transpose(H, 1, 2).conj()).det().real)) / batch_size
 


# ---- generating data set 

def Ploss2( Xr,Xu,test_size): #large scale loss
    loss=t.zeros([3,test_size])
    r=(2*t.rand( test_size)-1)/2#unit square deviation
    loss[0]=t.sqrt(10**-3*t.sqrt(r**2+(Xu+r)**2)**(-4))#BS 2 UE
    loss[1]=t.sqrt(10**-3*t.sqrt(10**2+(Xr*t.ones(test_size))**2)**(-2))#BS 2 RIS
    loss[2]=t.sqrt(10**-3*t.sqrt((10+r)**2+(Xu-Xr)**2)**(-2))#RIS 2 UE
    return loss
def Ploss1(  test_size,K,M,mode): #small scale loss
    if mode==1:
        H=t.randn(test_size,K,M,dtype=t.cfloat) 
    elif mode==2: 
        H1=t.randn(test_size,K,M,dtype=t.cfloat) 
        H2=t.zeros(test_size,K,M,dtype=t.cfloat)
        pathNO=15#15 LOS path
        Klin=np.linspace( 0, K-1, K); Mlin=np.linspace( 0, M-1, M);
        for i in range(test_size):
            Ht=t.zeros(K,M)
            for _ in range(pathNO):
                Ht=Ht+np.outer(e**(1j*np.pi*Klin*np.sin(2*np.pi*np.random.uniform())),e**(-1j*np.pi*Mlin*np.sin(2*np.pi*np.random.uniform()))) \
                *(np.random.normal()+1j*np.random.normal())/math.sqrt(2)
            H2[i]=Ht/math.sqrt(pathNO)
        H=(H1+H2)/math.sqrt(2)
    return H
 

dloss2=Ploss2( Xr,Xu,test_size)
RateAll=t.zeros([3,len(lopPara)]); TimeAll=np.zeros((3,len(lopPara)))
with t.no_grad():
    for lop in range(len(lopPara)):
        if dmode==1:#1=UE no  
            K = int(lopPara[lop] )     # Num of users
        elif dmode==2:#  2=antenna no 
            M =  lopPara[lop]     # Num of antenna
        elif dmode==3:# 3= RIS no
            R =  lopPara[lop]     # Num of RIS  
        elif dmode==4:
            SNR=   lopPara[lop] ; #dB
            Pt=10**(SNR/10)
        elif dmode==5:
            Xu=lopPara[lop] 
            dloss2=Ploss2( Xr ,Xu,test_size)
        elif dmode==6:
            Xr=lopPara[lop] 
            dloss2=Ploss2( Xr ,Xu,test_size)
        H_testR = t.reshape(dloss2[1],(test_size,1,1))*Ploss1(  test_size,K,R,mode=2)  ; 
        H_test0=  t.reshape(dloss2[2],(test_size,1,1))*Ploss1(  test_size,R,M,mode=2)  ; 
        H_testD = t.reshape(dloss2[0],(test_size,1,1))*Ploss1(  test_size,K,M,mode=1); 
    
    
        # executing classical GP on the test set 
        time_start = time.time() #开始计时
        RateAll[0,lop]=GP(stepSize ,H_testR,H_test0,H_testD,L,Pt) 
        time_end = time.time() ;TimeAll[0,lop]= time_end - time_start  ;print('GP time cost', TimeAll[0,lop], 's') #运行所花时间 
        # ---- unrolled GP ----
        # executing unrolled GP on the test set
        time_start = time.time() #开始计时
        G2,H=Unroll(muA2,muD2,mu2 ,H_testR,H_test0,H_testD,L,Pt ) 
        time_end = time.time() ;TimeAll[1,lop]= time_end - time_start  ;print('Unrolled GP time cost', TimeAll[1,lop], 's') #运行所花时间 
        RateAll[1,lop]=sum_capcity( G2, H,K,  test_size)
        # Full digital BF
        time_start = time.time() #开始计时
        RateAll[2,lop]=FullDigit(H_testR,H_test0,H_testD,stepSize,Pt)
        time_end = time.time() ;TimeAll[2,lop]= time_end - time_start  ;print('WF time cost', TimeAll[2,lop], 's') #运行所花时间 

# ploting the results
plt.figure()  
plt.plot(lopPara, RateAll[2].detach().numpy(), '--',label='Full digital BF+RIS') 
plt.plot(lopPara, RateAll[1].detach().numpy(), 'b-o',label='Unrolled GP') 
plt.plot(lopPara, RateAll[0].detach().numpy(), 'r-*', label=' GP') 
if dmode==1:#1=UE no  
    plt.xlabel('Number of users') 
elif dmode==2:#  2=antenna no 
    plt.xlabel('Number of transmit antennas') 
elif dmode==3:# 3= RIS no
    plt.xlabel('Number of RIS elements')  
elif dmode==4:# 4= Pt change SNR dB
    plt.xlabel('Transmit power') 
elif dmode==5:# 5= change users location
    plt.xlabel('Users x-axis coordinate (m)') 
elif dmode==6:# 6=change RIS locations
    plt.xlabel('RIS x-axis coordinate (m)') 
plt.ylabel('Achievable Rate (bps/Hz)')
plt.legend(loc='best')
plt.grid()
# plt.savefig('Figs/capacity1.eps', bbox_inches='tight')
plt.show() 

plt.figure()   
plt.plot(lopPara, TimeAll[0] , 'r-*', label=' GP') 
plt.plot(lopPara, TimeAll[1] , 'b-o',label='Unrolled GP') 
if dmode==1:#1=UE no  
    plt.xlabel('Number of users') 
elif dmode==2:#  2=antenna no 
    plt.xlabel('Number of transmit antennas') 
elif dmode==3:# 3= RIS no
    plt.xlabel('Number of RIS elements')  
elif dmode==4:# 4= Pt change SNR dB
    plt.xlabel('Transmit power') 
elif dmode==5:# 5= change users location
    plt.xlabel('Users x-axis coordinate (m)') 
elif dmode==6:# 6=change RIS locations
    plt.xlabel('RIS x-axis coordinate (m)') 
plt.ylabel('Compution Time (s)')
plt.legend(loc='best')
plt.grid()
# plt.savefig('Figs/capacity1.eps', bbox_inches='tight')
plt.show() 
 
 
# Saving the objects:
# file = open(f'./Figs/TimeGMode{dmode}.pkl', 'wb') 
# pickle.dump(  [RateAll,TimeAll,lopPara,dmode]  , file);file.close() 