# -*- coding: utf-8 -*-


#%clear
import numpy as np
import pandas as pd
import time
import networkx as nx
from gurobipy import *

#%%
start=time.time()
df=pd.read_csv(r'digsourceuser-10000-100000.csv')
df1=pd.read_csv(r'digtargetuser-10000-100000.csv')
c=[]
nodes=1203
for i in range(1203):
    c.append(str(i))
df2=pd.DataFrame(df,columns=c)
df3=pd.DataFrame(df1,columns=c)
k=300
start=time.time()
    
#%%
#define the state of users at the beginning time
A=pd.read_csv(r'116-1203nodes-adjacency.csv')
#get the adjacency matrix
A=A.values
A=A[:,1:]
B=[[0 for i in range(nodes)] for j in range(nodes)]
B=np.dot(A,A)

#%%
def randomwalk(j,trainkn,testkn, X_train,Y_train,X_test,Y_test):
    #Create optimization model
    m=Model('network')
    #define the varible 
    pxT=[]    
    for i in range(trainkn):
        pxT.append(m.addVar(vtype=GRB.BINARY,name="pxT[%d]"%(i)))
    
    w=[]
    for i in range(nodes):
        w.append([])
        for mm in range(2):
            w[i].append(m.addVar(lb=0,ub=1,vtype=GRB.CONTINUOUS,name="w[%d,%d]"%(i,mm)))

    z=[]
    for k in range(trainkn): #k represents the number of samples
        z.append(m.addVar(vtype=GRB.BINARY,name="z[%d]"%(k)))         
    # objective function
    m.update()
    m.modelSense=GRB.MINIMIZE
    m.setObjective(quicksum(z[k] for k in range(trainkn))/trainkn)
    
    for k in range(trainkn):
        m.addConstr(z[k]>=Y_train[k]-pxT[k])
        m.addConstr(z[k]>=pxT[k]-Y_train[k])
        m.addConstr(quicksum((A[i][j]*w[i][0]+B[i][j]*w[i][1])*X_train[k][i] for i in range(nodes) if (i!=j))-0.5001 >= -0.5001*(1-pxT[k]))
        m.addConstr(quicksum((A[i][j]*w[i][0]+B[i][j]*w[i][1])*X_train[k][i] for i in range(nodes) if (i!=j))-0.5 <= (quicksum((A[i][j]+B[i][j]) for i in range(nodes) if (i!=j))-0.5)*pxT[k])
    
    
    for i in range(nodes):
       m.addConstr(w[i][0]>=0)
       m.addConstr(w[i][0]<=1)
       m.addConstr(w[i][1]>=0)
       m.addConstr(w[i][1]<=1)
  
    
   
    m.optimize()
    for k in range(trainkn):
        pxT[k]=pxT[k].x
    #train accuracy
    n1=0
    for k in range(trainkn):
        if (pxT[k]==Y_train[k]):
            n1=n1+1
    accuracy1=n1/trainkn

    n3=0
    n4=0
    testpxT=[0 for j in range(testkn)]
    z2=[0 for j in range(testkn)]
    for k in range(testkn):
        if round(sum((A[i][j]*w[i][0].x+B[i][j]*w[i][1].x)*X_test[k][i] for i in range(nodes) if (i!=j)),4) >= 0.5001:
            testpxT[k]=1
        else:
            testpxT[k]=0
        
        z2[k]=abs(testpxT[k]-Y_test[k])
        if z2[k]==0:
            n3=n3+1
        else:
            n4=n4+1
            
    accuracy2=n3/testkn
    weight=[]
    prexT=[]
    for i in range(nodes):
        for j in range(2):
            weight.append(w[i][j].x)
    for j in range(trainkn):    
        prexT.append(pxT[j])
    return(m.objVal,n1,n3,accuracy1,accuracy2)
#%%

traintrue=0
testtrue=0
traintrue1=0
testtrue1=0
Acc1=[]
Acc2=[]
trainkno=[]
testkno=[]
Traint=[]
Testt=[]
ratio=[]
o=0
for i in range(nodes):
    df2train=df2.loc[:k-1]
    df2test=df2.loc[k:]
    df3train=df3.loc[:k-1]
    df3test=df3.loc[k:]
    df4train=df2train[df2train[str(i)]==0]  #choose the rows whose initial state is 0
    indtrain=pd.DataFrame(df4train.index).values
    indtrain=indtrain.reshape(-1)
    #print (len(df4train))
    df5train=df3train.loc[indtrain,:]
    df6train=df5train[[str(i)]]
    df6train=df6train.rename(columns={str(i): 'Y'})
    df7train=pd.concat([df4train,df6train], axis=1)
    df7train=df7train.reset_index(drop=True)
    df7train.head(3)
    df8train=df7train[df7train['Y']==1]
    if len(df7train)!=0:
        ratio.append(len(df8train)/len(df7train))
        trainkn=len(df7train)
    else:
        ratio.append(0)
    
    df4test=df2test[df2test[str(i)]==0]  #choose the rows whose initial state is 0
    indtest=pd.DataFrame(df4test.index).values
    indtest=indtest.reshape(-1)
    #print (len(df4test))
    df5test=df3test.loc[indtest,:]
    df6test=df5test[str(i)]
    df7test=pd.concat([df4test,df6test], axis=1)
    df7test=df7test.reset_index(drop=True)
    df7test.head(3)
    testkn=len(df7test)
    
    Xtrain=df7train.values
    Xtest=df7test.values
    X_train=Xtrain[:,:1203]
    Y_train=Xtrain[:,1203]
    X_test=Xtest[:,:1203]
    Y_test=Xtest[:,1203]
    Y_train=Y_train.reshape(-1)
    Y_test=Y_test.reshape(-1)
    
    trainkno.append(trainkn)
    testkno.append(testkn)
    if trainkn!=0 and testkn!=0:
        obj,trainn,testn,acc1,acc2=randomwalk(i,trainkn,testkn,X_train,Y_train,X_test,Y_test)
        print ('******************************************************')
        print ('Time: %f',time.time()-start)
        Acc1.append(acc1)
        Acc2.append(acc2)
        Traint.append(trainn)
        Testt.append(testn)
        print ('*********************************************************')
        print ('Number %d node:'% i)
        print ('Train samples are %d, test samples are %d.'% (trainkn,testkn))
        print ('Train accuracy is %f, test accuracy is %f.'% (acc1,acc2))
    else:
        print ('No training: %d'% i)
        trainn=0
        testn=0
        obj=0
    Traint.append(trainn)
    Testt.append(testn)
    traintrue=traintrue+trainn+len(df2train)-len(df4train)
    testtrue=testtrue+testn+len(df2test)-len(df4test)
    traintrue1=traintrue1+trainn
    testtrue1=testtrue1+testn
    

#%%
print ('********************************************')    
print ('Training Accuracy: %f'%(traintrue/(nodes*int(k))))
print ('Testing Accuracy: %f'%(testtrue/(nodes*(len(df2)-int(k)))))    
