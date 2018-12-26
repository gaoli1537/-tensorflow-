import tensorflow as tf


x1=tf.Variable([4,5],dtype=tf.float32)


y=5*tf.pow(x1[0],2.0)+3*tf.pow(x1[1],2.0)

opti=tf.train.GradientDescentOptimizer(0.01).minimize(y)

session=tf.Session()

session.run(tf.global_variables_initializer())

for i in range(1000):
    session.run(opti)
    print(session.run(y))
    print(session.run(x1))