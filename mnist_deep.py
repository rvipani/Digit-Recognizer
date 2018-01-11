from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import tensorflow as tf

FLAGS = None



class NN:
  def __init__(self):
    self.label_queue = np.random.choice(10,size=10,replace=False)
    # Import data
    self.mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)


  def deepnn(self,x):
    """deepnn builds the graph for a deep net for classifying digits.

    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.

    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = self.weight_variable([5, 5, 1, 32])
    b_conv1 = self.bias_variable([32])
    h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = self.max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = self.weight_variable([5, 5, 32, 64])
    b_conv2 = self.bias_variable([64])
    h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = self.max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
    b_fc1 = self.bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = self.weight_variable([1024, 10])
    b_fc2 = self.bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


  def conv2d(self,x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


  def max_pool_2x2(self,x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


  def weight_variable(self,shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


  def bias_variable(self,shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def mini_batch(self,epoch,batchsize=50,random_label=False):
    # starting batch position
    start = self.train_pos_counter
    # end batch position
    end = self.train_pos_counter + (batchsize-1)
    # move batch beggining indicator
    self.train_pos_counter += batchsize
    # if there are not enough elements left, redo the batch
    if end > len(self.mnist.train.images):
      self.idx_train = list(range(0,len(self.mnist.train.images)))
      np.random.shuffle(self.idx_train)
      self.train_pos_counter = batchsize
      start = 0
      end = (batchsize-1)
    perm0 = self.idx_train[start:end]
    perm1 = perm0.copy()
#    if random_label:
## HERE IS THE RANDOMIZATION WE SAW IN CLASS
    #np.random.shuffle(perm1)
    return [self.mnist.train.images[perm0],self.mnist.train.labels[perm1]]


  def execute(self,runid=0):
    batchsize = FLAGS.batch_size
    self.idx_train = list(range(0,len(self.mnist.train.images)))
    self.train_pos_counter = 0
    print("# Parameters: batchsize=",batchsize,"; learning rate =",FLAGS.lr,"; algorithm=",FLAGS.algo,"; training_samples=",FLAGS.training_samples)

    # Input image placeholder
    x = tf.placeholder(tf.float32, [None, 784])
    # Input image placeholder
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv, keep_prob = self.deepnn(x)

    #define the loss function
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # define the optimizer (no gradient momentum, only variance momentum)
    train_step = tf.train.AdamOptimizer(learning_rate=FLAGS.lr,beta1=0.0,beta2=0.99,use_locking=True).minimize(cross_entropy)
    # define what a correct prediction is
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    # define what accuracy means
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
      # initialize all model variables
      sess.run(tf.global_variables_initializer())
      # one-hot encoding of the digits 0-9
      label_vals = tf.one_hot(list(range(10)),depth=10).eval()
      # find the image indices of each label
      for i in range(FLAGS.training_samples//batchsize):
        # Mutilayer SGD strategy:
        batch = self.mini_batch(epoch=i,batchsize=batchsize)

        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('runid %d step %d, training accuracy %g' % (runid, i, train_accuracy))
        
        # train NN with current batch. For now we are keeping dropout at 0.5 (it should not matter).
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        if i % 1000 == 0:
          print('#runid %d step= %d temporary test acc: %g' % (runid, i,accuracy.eval(feed_dict={
              x: self.mnist.test.images, y_: self.mnist.test.labels, keep_prob: 1.0})))


      print('runid %d after %d steps test accuracy is %g' % (runid, i+1,accuracy.eval(feed_dict={
          x: self.mnist.test.images, y_: self.mnist.test.labels, keep_prob: 1.0})))

def main(_):
  NNmodel = NN()
  NNmodel.execute()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--batch_size', type=int,
                      default='50',
                      help='Batch Size')
  parser.add_argument('--algo', type=str,
                      default='Rand',
                      help='Mutilayer Algorithm (Rand, SingleDig, TwoDig, SingleDig10)')
  parser.add_argument('--lr', type=float,
                      default='1e-04',
                      help='Learning rate')
  parser.add_argument('--training_samples', type=int,
                      default='500000',
                      help='Learning rate')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

