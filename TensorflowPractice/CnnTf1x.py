from keras.datasets.mnist import load_data
import tensorflow as tf 


(x_train,y_train),(x_test,y_test)=load_data('D:\ComputerVision\TensorflowPractice\dataset\mnist.npz')
x_train,x_test=x_train/255.0, x_test/255.0
x_train=x_train.reshape(-1,28,28,1)
# x_train = tf.expand_dims(x_train, axis=-1)
x_test=x_test.reshape(-1,28,28,1)



cngrph=tf.Graph()
epochs=1
batch_size=1000

def invoke_train(batch,batch_size):
    off1=batch * batch_size
    off2=batch * batch_size + batch_size
    batch_x=x_train[off1:off2]
    batch_y=y_train[off1:off2]
    return batch_x,batch_y

with cngrph.as_default():
    def get_dense_layer(x,in_shape,out_shape):
        w=tf.Variable(tf.compat.v1.truncated_normal([in_shape,out_shape],mean=0.0,stddev=1.0,dtype=tf.float32))
        b=tf.Variable(tf.zeros(out_shape,dtype=tf.float32))
        return tf.nn.relu(tf.add(tf.matmul(x,w),b))

    def get_conv_layer(x,filtrs,strd,pads):
        conv=tf.nn.conv2d(input=x,filters=tf.Variable(tf.compat.v1.truncated_normal(filtrs,mean=0.0,stddev=0.1))
                          ,strides=strd,padding=pads)
        conv=tf.nn.bias_add(conv,tf.Variable(tf.constant(0.0,shape=[filtrs[-1]])))
        return conv

    with tf.name_scope('placeholder'):
        x=tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,28,28,1])
        y=tf.compat.v1.placeholder(dtype=tf.int32)

    with tf.name_scope('Hidden_Conv_layer'):
        conv1=get_conv_layer(x,filtrs=[5,5,1,32],strd=[1,1,1,1],pads='SAME')
        conv1=tf.nn.relu(conv1)
        conv1=tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

        conv2=get_conv_layer(conv1,filtrs=[5,5,32,64],strd=[1,1,1,1],pads='SAME')
        conv2=tf.nn.relu(conv2)
        conv2=tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

    with tf.name_scope('Hidden_Dense_layer'):
        flt=tf.reshape(conv2,[-1,6*6*64])
        fc1=get_dense_layer(flt,in_shape=6*6*64,out_shape=1000)
        fc2=get_dense_layer(fc1,in_shape=1000,out_shape=128)

    with tf.name_scope('Output_layer'):
        out_w=tf.Variable(tf.compat.v1.truncated_normal([128,10],mean=0.0,stddev=0.1,dtype=tf.float32)) 
        out_b=tf.Variable(tf.zeros(10,dtype=tf.float32))
        logits=tf.add(tf.matmul(fc2,out_w),out_b)


   # loss_for_sample=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
    loss_for_sample=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
    loss=tf.reduce_mean(loss_for_sample)
    optimizer=tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss)



with tf.compat.v1.Session(graph=cngrph) as sess:
    init=tf.compat.v1.global_variables_initializer()
    sess.run(init)
    for epoch in range(epochs):
        num_batch=len(x_train)//batch_size
        total_loss=0
        for j in range(num_batch):
           x_batch,y_batch=invoke_train(j,batch_size)
           optim,lss=sess.run([optimizer,loss],feed_dict={x:x_batch,y:y_batch})
           total_loss+=lss
        print(total_loss/num_batch)    



