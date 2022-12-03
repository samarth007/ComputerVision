import tensorflow as tf
from tensorflow.python.keras import Model
from keras.datasets.mnist import load_data
from tensorflow.python.keras import layers


(x_train,y_train),(x_test,y_test)=load_data('D:\ComputerVision\TensorflowPractice\dataset\mnist.npz')
x_train,x_test=x_train/255.0, x_test/255.0
# x_train=x_train.reshape(-1,28,28,1)
# x_train = tf.expand_dims(x_train, axis=-1)
x_test=x_test.reshape(-1,28,28,1)

epochs=1
batch_size=1000
class ConvTens(Model):
    def __init__(self):
        super(ConvTens,self).__init__()
        self.conv1=layers.Conv2D(filters=32,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu')
        self.pool=layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='valid')
        self.conv2=layers.Conv2D(filters=64,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu')
        self.flat=layers.Flatten()
        self.fc1=layers.Dense(1000,activation='relu')
        self.fc2=layers.Dense(128,activation='relu')
        self.fc3=layers.Dense(10,activation='softmax')

    def call(self,x):
        x=self.conv1(x)
        x=self.pool(x)
        x=self.conv2(x)
        x=self.pool(x)
        x=self.flat(x)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.fc3(x)
        return x    

# model=ConvTens()
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
# print(model.summary()) 
# model.save_weights('./TensorflowPractice/model.h5')  
 
 
       
# model.build(x_train.shape)
# model.load_weights('./TensorflowPractice/model.h5')
# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
 