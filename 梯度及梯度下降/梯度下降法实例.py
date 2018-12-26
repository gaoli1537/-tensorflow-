import tensorflow as tf


x=tf.Variable(4.0,dtype=tf.float32)
y=tf.pow(x-1,2.0)

opti=tf.train.GradientDescentOptimizer(0.25).minimize(y)

session=tf.Session()

session.run(tf.global_variables_initializer())

for i in range(10):
    session.run(opti)
    print(y)
    print(session.run(x))
