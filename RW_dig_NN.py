# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import time
import networkx as nx
import tensorflow as tf
from tqdm import tqdm
from numpy.linalg import matrix_power
#%%
start=time.time()
df=pd.read_csv(r'digsourceuser-10000-100000.csv')
df1=pd.read_csv(r'digtargetuser-10000-100000.csv')

c=[]
for i in range(1203):
    c.append(str(i))
df2=pd.DataFrame(df,columns=c)
df3=pd.DataFrame(df1,columns=c)
X1=df2.values
X1=np.concatenate((X1,X1),axis=1)
Y1=df3.values
Shape=X1.shape[1]
tkn=len(df)
#%%
kn=300
X_train=X1[:kn,:]
X_test=X1[kn:tkn,:]
Y_train=Y1[:kn,:]
Y_test=Y1[kn:tkn,:]

tf.reset_default_graph()

#%%
A=pd.read_csv(r'116-1203nodes-adjacency.csv')

#get the adjacency matrix
A=A.values
A=A[:,1:]


#%%
#Parameters
learning_rate=0.001
num_steps=500
#epochs=50
batch_size=10
#iterations=len(Y_train)*epochs
#trainiterations=len(Y_train)
#titerations=len(Y_test)
#Network Parameters
num_input=1203*2
num_classes=1203
#%%
#tf input
X=tf.placeholder("float32",[None, num_input], name="X")
Y=tf.placeholder("float32",[None, num_classes], name="Y")


#%%
weights=tf.Variable(tf.truncated_normal([num_input, num_classes], stddev=0.1, seed=0))
biases=tf.constant(-0.5, shape=[num_classes])
#%%
#biases=tf.constant(-0.5, shape=[num_classes])
mask1 = np.ones((num_input,num_classes))    
mask2 = np.eye(num_input,num_classes,0)
mat1=mask1-mask2
place1=tf.placeholder(tf.float32, shape=(num_input,num_classes))
place2=tf.placeholder(tf.float32, shape=(num_input,num_classes))
#mask1 = np.ones((num_input,num_input))    
#mask2 = np.identity(num_input, dtype=np.float32)
#mask1=tf.constant(mask1,dtype=tf.float32)
#tensor_mask1 = tf.constant(mask1-mask2,dtype=tf.float32)
#tensor_mask2 = tf.constant(mask2,dtype=tf.float32)
#%%
place3=tf.placeholder(tf.float32, shape=(num_input,num_classes))

#Construct Model 
I=np.identity(num_classes)
A1=A+I
B=np.dot(A,A)
#C=np.dot(B,A)
adjacency=np.concatenate((A1,B),axis=0)

#adjacency=tf.constant(A)
#adjacency=tf.cast(adjacency,tf.float32)  
 
#%%
weights = tf.clip_by_value(tf.multiply(weights,place1) + place2,0,1)
capped_weights = weights
act=tf.matmul(X,tf.multiply(place3,weights))+biases
prediction=tf.nn.sigmoid(act)
#%%
#masks=np.ones((10,1203))-X
#sq=tf.multiply(masks,tf.square(Y-prediction))
#loss=tf.reduce_mean(sq)
loss_op=tf.losses.mean_squared_error(Y,prediction)
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

#%%
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.round(prediction),Y),tf.float32))

#%%
def next_batch(batch_size, data, labels,s):
    '''
    Return a total of `num` random samples and labels. 
    '''
    data_step=data[batch_size*s:batch_size*s+batch_size]
    labels_step=labels[batch_size*s:batch_size*s+batch_size]
    #                 print ("data_step:%d",data_step)
    #print ("labels_step:%d",labels_step)
    return np.asarray(data_step), np.asarray(labels_step)

#%%
#Start training
weight=[]
mid=[]
display_step=100
#clip_op= tf.assign(weights['h1'],tf.clip_by_value(weights['h1'],0,1))
with tf.Session() as sess:
    s=0
    #writer=tf.summary.FileWriter(r'C:\Users\zh846675\Project2\Demo1\Step Function\graphs',sess.graph)
    init=tf.global_variables_initializer() 
    sess.run(init)
    #sess.run(tf.assign(weights['h1'],tf.clip_by_value))
    for step in range(1, num_steps+1):
        batch_x, batch_y=next_batch(batch_size,X_train,Y_train,s)
        if s<29:
            s=s+1
        else:
            s=0
        #Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, place1:mat1, place2:mask2, place3:adjacency})
        #sess.run(clip_op)
        if step % display_step==0 or step==1:
            loss, acc, w, b=sess.run([loss_op, accuracy, capped_weights, biases], feed_dict={X: batch_x, Y: batch_y, place1:mat1, place2:mask2, place3:adjacency})
            weight.append(w)
            print("Step "+str(step)+", Minibatch Loss=" + "{:.4f}".format(loss)+", Training Accuracy="+"{:.3f}".format(acc))
    print ("Optimization Finished!")
        #Calculate accuracy for MNIST test images

    print ("Training Accuracy:", sess.run([loss_op,accuracy], feed_dict={X: X_train, Y: Y_train, place1:mat1, place2:mask2, place3:adjacency}))
    print ("Testing Accuracy:", sess.run([loss_op,accuracy], feed_dict={X: X_test, Y: Y_test, place1:mat1, place2:mask2, place3:adjacency}))
    print ("Sigmoid")
#writer.close()
print ("rwTime:%f",time.time()-start)
