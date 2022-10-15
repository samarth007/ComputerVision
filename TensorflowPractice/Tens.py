import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras import Model
import tensorflow.python.keras.layers as tk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv('./Conv2d/boston.csv')
X=data.drop('price',axis=1)
Y=data['price']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=23)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#EXAMPLE WITH TF2.XX
# class Regress(Model):
#     def __init__(self):
#         super(Regress,self).__init__()
#         self.l1=tk.Dense(1)
#     def call(self,x):
#         return self.l1(x)

# model=Regress()
# model.compile(loss='mse',optimizer='adam')
# model.fit(x_train,y_train,epochs=200,batch_size=20)
# model.evaluate(x_test,y_test)

#'-----------------------------------------------------------------------------------------'

#EXAMPLE WITH TF1.XX
epochs=100

batch_size=50
def invoke_train(batch_size,batch):
    off_1= batch * batch_size
    off_2= batch * batch_size + batch_size
    batch_x=x_train[off_1:off_2]
    batch_y=y_train[off_1:off_2]

    return batch_x,batch_y

grph=tf.Graph()
with grph.as_default():
    x=tf.compat.v1.placeholder(dtype=tf.float64,shape=(None,13))
    y=tf.compat.v1.placeholder(dtype=tf.float64)
    w=tf.Variable(tf.compat.v1.truncated_normal([13,1],mean=0,stddev=1.0,dtype=tf.float64))
    b=tf.Variable(tf.zeros(1,dtype=tf.float64))

    y_pred=tf.add(tf.matmul(x,w),b)
    loss=tf.reduce_mean(tf.square(y-y_pred))
    optim=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.compat.v1.Session(graph=grph) as sess:
    init=tf.compat.v1.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs): 
        num_batch=len(x_train)//batch_size
        train_loss=0
        for i in range(num_batch):
           x_batch,y_batch=invoke_train(batch_size,i)  
           optimizer,losses=sess.run([optim,loss],feed_dict={x:x_batch,y:y_batch})
           train_loss+=losses 
        print('Training loss for {} epoch is {}'.format(epoch,train_loss/num_batch))


