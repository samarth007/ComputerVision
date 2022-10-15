import tensorflow as tf

g=tf.Graph()
v=tf.Graph()
with g.as_default():
    a=tf.constant(value=[2,3,4],dtype=tf.int8)
    b=tf.constant(value=[4,5,6],dtype=tf.int8)
    c=tf.constant(value=[[6,4,1],[8,5,9]],dtype=tf.int8)

    x=tf.add(a,b)
    y=tf.multiply(x,c)


with tf.compat.v1.Session(graph=g) as sess:
    print(sess.run([x,y]))