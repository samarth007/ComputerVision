import tensorflow as tf


gph=tf.Graph()
with gph.as_default():
    a=tf.compat.v1.placeholder(dtype=tf.int8,shape=(3,))
    b=tf.compat.v1.placeholder(dtype=tf.int8,shape=(3,))
    
    x=tf.add(a,b)

with tf.compat.v1.Session(graph=gph) as s:
    sess=s.run([x],feed_dict={a:[2,3,4],b:[2,3,4]})
    print(sess)    