
import tensorflow as tf
import numpy as np
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir,now)

a = tf.placeholder(dtype=tf.float32,shape=[100,2],name="a")
b = tf.placeholder(dtype=tf.float32,shape=[100,1],name="b")
learning_rate = 0.01

# sample data
x_data = np.random.randn(100,2)
y_data = np.random.randn(100,1)

# parameters
weight = tf.Variable(tf.random_normal([2,1]),name="weight")

y_pred = tf.matmul(a,weight)
error = y_pred - b
mse = tf.reduce_mean(tf.square(error),name="mse")
gradients = (2/100)*tf.matmul(tf.transpose(a),error)
training_op = tf.assign(weight,weight - learning_rate*gradients)

mse_summary = tf.summary.scalar("MSE",mse)
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1000):
        summary_str = mse_summary.eval(feed_dict={a:x_data,b:y_data})
        file_writer.add_summary(summary_str,epoch)
        sess.run(training_op,feed_dict={a:x_data,b:y_data})
    updated_weights = weight.eval()
file_writer.close()
