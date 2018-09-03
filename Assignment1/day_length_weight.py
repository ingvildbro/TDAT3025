from __future__ import print_function
import csv
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
rng = numpy.random

# Parameters
learning_rate = 0.0001
training_epochs = 5000
display_step = 50

# Array for data parsed from csv file
X_values = []
Y_values = []

# Import data from csv file
lenWeiReader = csv.reader(open('./data/day_length_weight.csv', newline='\n'), delimiter=',')
for row in lenWeiReader:
    X_values.append([numpy.float64(row[1]), numpy.float64(row[2])])
    Y_values.append([numpy.float64(row[0])])


# Preparing training data
train_X = numpy.mat(X_values)
train_Y = numpy.mat(Y_values)
n_samples = train_X.shape[0]

#print(train_X)
#print(train_Y)

# Linear regression model class
class LinearRegressionModel2d:
    def __init__(self):
        # Model input
        self.X_in = tf.placeholder(tf.float32, [None, 2], "X_in")
        self.Y_in = tf.placeholder(tf.float32, [None, 1], "y_in")

        # Model variables
        self.W = tf.Variable(tf.random_normal([2, 1]), name="w")
        self.b = tf.Variable(tf.constant(0.1, shape=[]), name="b")

        # Predictor
        pred = tf.add(tf.matmul(self.X_in, self.W), self.b)

        # Uses Mean Squared Error, although instead of mean, sum is used.
        self.loss = tf.reduce_mean(tf.square(tf.subtract(self.Y_in, pred)),
                                   name="loss")

    def f(self, x, x2):
        return x * self.W[0] + x2 * self.W[1] + self.b


# Instantiate model object
model = LinearRegressionModel2d()


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
            sess.run(optimizer, feed_dict={model.X_in: x, model.Y_in: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(model.loss, feed_dict={model.X_in: train_X, model.Y_in: train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(model.W), "b=", sess.run(model.b))
            #if last_W == sess.run(model.W) and last_b == sess.run(model.b):
                #break
            #else:
                #last_W = sess.run(model.W)
                #last_b =sess.run(model.b)

    print("\n--------------------------------------------------------------\n")
    print("Optimization Finished!")
    training_loss = sess.run(model.loss, feed_dict={model.X_in: train_X, model.Y_in: train_Y})
    print("Training loss=", training_loss, "W=", sess.run(model.W), "b=", sess.run(model.b), '\n')

    # Graphic display
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for x, y in zip(X_values, Y_values):
        xs = x[0]
        x2s = x[1]
        ys = y[0]
        ax.scatter(xs, x2s, ys, c='r')

    ax.set_xlabel('Length')
    ax.set_ylabel('Weight')
    ax.set_zlabel('Day')

    ax.hold(True)
    x = [numpy.min(train_X[0]), numpy.max(train_X[0])]
    x2 = [numpy.min(train_X[1]), numpy.max(train_X[1])]

    X, X2 = numpy.meshgrid(x, x2)
    Y = model.f(X, X2)
    print(type(Y))
    ax.plot_surface(X, X2, Y)

    plt.show()

"""
    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(model.W) * train_X + sess.run(model.b), label='Fitted line')
    plt.legend()
    plt.show()
"""