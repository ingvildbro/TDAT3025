from __future__ import print_function

import csv
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
import math

rng = numpy.random

# Parameters
learning_rate = 0.000001
training_epochs = 5000
display_step = 50

# Array for data parsed from csv file
X_values = []
Y_values = []

# Import data from csv file
lenWeiReader = csv.reader(open('./data/day_head_circumference.csv', newline='\n'), delimiter=',')
for row in lenWeiReader:
    X_values.append(numpy.float32(row[0]))
    Y_values.append(numpy.float32(row[1]))

# Preparing training data
train_X = numpy.asarray(X_values)
train_Y = numpy.asarray(Y_values)
n_samples = train_X.shape[0]


# Linear regression model class
class NonLinearRegressionModel2d:
    def __init__(self):
        # Model input
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable(rng.randn(), name="circumference")
        self.b = tf.Variable(rng.random(), name="bias")

        # Predictor - 20 * sigmoid (x * W + b) + 31
        pred = 20 * tf.sigmoid(tf.add(tf.multiply(self.X, self.W), self.b)) + 31

        # Uses Mean Squared Error, although instead of mean, sum is used.
        self.loss = tf.reduce_sum(tf.pow(pred - self.Y, 2)) / (2 * n_samples)


# Instantiate model object
model = NonLinearRegressionModel2d()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(model.loss)

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Initialize control variables
    last_W = "float"
    last_b = "float"
    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={model.X: x, model.Y: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(model.loss, feed_dict={model.X: train_X, model.Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), \
                  "W=", sess.run(model.W), "b=", sess.run(model.b))
            if last_W == sess.run(model.W) and last_b == sess.run(model.b):
                break
            else:
                last_W = sess.run(model.W)
                last_b = sess.run(model.b)

    print("\n--------------------------------------------------------------\n")
    print("Optimization Finished!")
    training_loss = sess.run(model.loss, feed_dict={model.X: train_X, model.Y: train_Y})
    print("Training loss=", training_loss, "W=", sess.run(model.W), "b=", sess.run(model.b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(model.W) * train_X + sess.run(model.b), label='Fitted line')
    plt.legend()
    plt.show()