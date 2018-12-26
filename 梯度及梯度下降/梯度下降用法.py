import tensorflow as tf
import numpy as np

x=tf.placeholder(tf.float32,shape=(2,1))
w=tf.constant([[3,4]],tf.float32)

y=tf.matmul(w,x)
F=tf.pow(y,3)

grads=tf.gradients(F,x)

session=tf.Session()

result=session.run(grads,feed_dict={x:np.array([[2],[3]])})

print(result)


'''
结果
[array([[108.],
       [144.]], dtype=float32)]

'''