import tensorflow as tf
import numpy as np

kernel=tf.constant([[1,0,1],[1,0,1],[1,0,1]])
# print(kernel.shape)
kernel=tf.reshape(kernel,[*kernel.shape,1,1])
# print(kernel.shape)
kernel=tf.cast(kernel,dtype=tf.float32)  #cast convert dtype
# print(kernel.shape)
# print(np.squeeze(kernel).shape)

val=np.array([[[2,3,4],[5,3,7]]])
print(val.shape)                 # (1,2,3)
val1=val.reshape(-1)             #  (-1) means convert to 1d array returns (6,)
val2=np.reshape(val,(3,-1))      #  (3,-1) means convert to 2d array  returns (3,2)
val3=np.reshape(val,(3,2,-1))    #(3,2,1) means convert to 3d array returns (3,2,1)
valsq=np.squeeze(val3)           #np.squeeze() converts to 2d array
print(val2)                    
