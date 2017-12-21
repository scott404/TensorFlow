from __future__ import absolute_import#Basic Imports
from __future__ import division#Basic Imports
from __future__ import print_function #Basic Imports

import argparse #More imports

from tensorflow.examples.tutorials.mnist import input_data# Looks like we are calling a specific what I know as "package" and importing specifically the input_date, likely the test and such data discussed in tutorial if my understanding is correct.

import tensorflow as tf#Imports tensor flow

FLAGS = None

def main(_):
    # Import data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #Setups MNIST Need more clarity on the Git hub version versus tutorial version. 
    #However based on further investigation from https://www.tensorflow.org/get_started/mnist/mechanics it looks like the first is the directory we are downloading train data to.
    #Where the second argument is labeled as being safe to be ignored, however still unsure about this. 
    
    #Create model
    x = tf.placeholder(tf.float32, [None, 784]) #Holds the multi dimensonal shape
    y_ = tf.placeholder(tf.float32, [None, 10]) #Holds the actual value of the object (Label)
    
    W = tf.Variable(tf.zeros([784, 10]))#Trainable settings
    b = tf.Variable(tf.zeros([10]))#Trainable settings
    y = tf.matmul(x, W) + b#Multiplies our two matrices of our place older and our trainable variable

    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) #Here we are setting up our optimization algorithm to make as optimized without completely giving up accuracy.
    
    
    sess = tf.InteractiveSession() # Creates new interactive session.
    tf.global_variables_initializer().run() #Need a little clarification. However, while not understanding "run()" I am understanding all we are doing is setting up some variables via a Method it would appear
    
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  #Need clarification on specifics but see that it just grabs a batch of 100 per loop, loop being 1000 iterations
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) # Runs with the optimizer we set up and a new variable "Feed_dict" set up here.
    
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) ## Pretty each to understand, y contains our guess, where as y_ contains the actual answer Returns booleans
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #Really just takes the booleans from above and makes a percentage
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) #This does a print of the objects results
    
    
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    
    #This appears to be the structure that actually runs the main from above. It is very reminiscent of java apps where the code above the "main" method is what's really happening and the true main is just a simple initializer for lack of better terminology.