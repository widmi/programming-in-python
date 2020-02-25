# -*- coding: utf-8 -*-
"""12_tensorflow_pytorch.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.10.2019

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

In this file we will learn how to create computational graphs, speed up your
Python code, utilize the GPU, and get ready for the basics of ML code with the
PyTorch and TensorFlow modules.
"""


###############################################################################
# General information on GPU programming
###############################################################################
# There are 2 major standards for communicating with GPUs:
# OpenGL: Supported by most GPUs (AMD and NVIDIA)
# CUDA: Optimized for scientific calculations and neural networks, restricted
# to NVIDIA GPUs.
# Both interfaces are relatively hardware dependent, especially newer NVIDIA
# GPUs benefit from the latest CUDA versions. However, not every NVIDIA GPU
# supports every CUDA version.
# There are 2 major factors that can slow down your GPU calculations:
# 1.: Bottleneck data-transfer: If your GPU has small memory or the transfer
# bandwidth is slow, loading your data from CPU RAM to GPU memory will become
# a major issue.
# 2.: Computation speed: The actual speed of your GPU computations might be
# not fast enough or the computations you issued might not be parallelized
# optimally.
# Rule of thumb: GPU utilization should be as high as possible,
# GPU memory and bandwidth utilization should be as low as possible. (You
# can check this in the NVIDIA server settings.)
# Important: Most GPUs only support float32 calculations!


###############################################################################
# Computational graphs
###############################################################################
# Using CUDA or OpenGL directly is possible but usually tedious. However, there
# are modules that allow for easier GPU utilization and optimization of your
# code. These modules typically require you to write abstract code that will
# get translated to a computational graph, which is optimized for one or
# multiple GPU(s) and/or CPU(s). Since we often need to calculate gradients in
# Machine Learning, some modules use the computational graph to automatically
# compute the gradients.
# Commonly used modules that also provide automatic gradient computation are:
# Theano (www.deeplearning.net/software/theano/):
#   Creates a static graph; Optimization for CPU(s) or a single GPU;
#   Predecessor of TensorFlow/PyTorch;
# TensorFlow1 (www.tensorflow.org):
#   Creates a rather static computational graph; Very popular in production
#   (pushed by Google/Deepmind); Optimization for CPU(s), GPU(s), TPU(s);
#   Very similar to Theano; Provides Tensorboard (visualization tools in
#   web-browser); Not very Python-like code;
# TensorFlow2 (www.tensorflow.org):
#   Creates a more dynamic computational graph; Very popular in production
#   (pushed by Google/Deepmind); Optimization for CPU(s), GPU(s), TPU(s);
#   More similar to PyTorch; Provides Tensorboard (visualization tools in
#   web-browser); More Python-like code; Partly uses Keras as interface;
# PyTorch (www.pytorch.org):
#   Creates a more dynamic computational graph; Very popular in research
#   (pushed by Facebook); Optimization for CPU(s), GPU(s)(, TPU(s));
#   Good for development and research; More Python-like code.
#
# All of these modules mainly deal with arrays and integrate nicely with
# numpy.
# There also exist modules that are wrappers for e.g. Neural Network design,
# such as Keras https://keras.io/ .


###############################################################################
# NVIDIA drivers, CUDA, CUDNN
###############################################################################
# If you want to utilize your GPU for high-performance computation, you will
# have to install appropriate drivers. Having the latest driver versions can
# give you speed-ups. For NVIDIA GPUs there are 3 components that you have to
# consider in the setup: The GPU driver, CUDA, CuDNN (make sure that PyTorch
# or whatever module you want to use can use the CUDA version!)
# Current PyTorch: nvidia-driver-440, CUDA 10.2, CuDNN 7.6
#
# Instructions for NVIDIA GPU on Ubuntu 18.04:
# In terminal:
# 1.) Add repository for graphics drivers:
# sudo add-apt-repository ppa:graphics-drivers
# sudo apt-get update
# 2.) Install the latest nvidia driver that supports the CUDA version you want.
# Check if version works with your GPU! E.g. here: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_19-11.html
# sudo apt-get install nvidia-driver-440
# Reboot machine.
# Activate the driver. (Go to Software -> Software&Updates -> additional drivers)
# Reboot machine.
# 3.) Install the latest CUDA supported by the module you want to use.
# Follow instructions on https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal
# Reboot machine.
# 4.) Install the latest CuDNN supported by the module you want to use. You
# will need to create a NVIDIA dev account.
# follow instructions on https://developer.nvidia.com/cudnn
# Reboot machine.
# 5.) Check your installation by typing
# nvidia-smi
# in the terminal.
# 6.) Install PyTorch
# pip3 install torch torchvision
# (See https://pytorch.org/get-started/locally/ for other versions, e.g.
# CPU-only or conda.)
# 7.)
# Open Python interpreter and type "import torch" to test your installation.


###############################################################################
# PyTorch
###############################################################################
# This section will give a very short introduction to PyTorch. For more
# detailed introductions please refer to the lecture in the next semester
# (Programming in Python II) or https://pytorch.org/tutorials/.

# We will start with a simple example for a computational graph. Let's assume
# we want to build a computational graph for the formula "c=a*b", where "a" and
# "b" are inputs and "c" should be our result.

# In Python we could write such a formula as:
a = 5.
b = 4.
c = a * b
print(f"Python c: {c}")
# ... which will give us 3 variables that point to float values. Variable "c"
# was created from variables "a" and "b". However, "c" only points to the
# result of the computation. If we had information about how "c" was computed
# (in our case that it is the result of a multiplication of "a" and "b"), we
# could apply optimization methods, automatic differentiation (=compute
# gradients automatically), and other magic. PyTorch (and others) store this
# information in a "computational graph". In case of PyTorch, this
# computational graph is built "on-the-fly" and quite pythonic, as we will
# see in a few lines. This is in contrast to Theano/Tensorflow 1, where one
# first creates the computational graph (symbolically without values) and then
# runs the graph while supplying input values.

# In PyTorch we could write our formula like this:
# Import numpy since we will be using arrays later
import numpy as np
# Import the PyTorch module, which is called "torch"
import torch
# Point "a" to PyTorch tensor (=node in graph) and keep track of gradients
a = torch.tensor([5.], requires_grad=True)
# Point "b" to PyTorch tensor (=node in graph) and keep track of gradients
b = torch.tensor([4.], requires_grad=True)
c = a * b  # Point "c" to multiplication of "a" and "b" (=node in graph)
# We now have defined 3 nodes in our graph (also called "tensors"). In PyTorch
# we can simply evaluate it by accessing the variable:
print(f"PyTorch c: {c}")  # Prints the tensor

# Since "c" is pointing to a PyTorch tensor, we have access to the benefits
# of the computational graph. Furthermore, the computation of "c" is optimized
# (by default for CPU). Some examples on what we can do with the tensor:

# Access the value as Python object:
print(f"c.item(): {c.item()}")

# Get computational graph for gradient computation
print(f"c.grad_fn: {c.grad_fn}")

# Compute gradients of "c" w.r.t. its input nodes...
c.backward()
# ... the gradients that were computed are now accumulated in the nodes:
print(f"a.grad: {a.grad}")  # this is the derivative of "c" w.r.t. "a"
# ("c=a*b" ... derivative of this "c" w.r.t. "a" is "1*b", which has value 4.)
print(f"b.grad: {b.grad}")  # this is the derivative of "c" w.r.t. "b"
# Important: The gradients are accumulated in the nodes. If you want to reset
# them you have to call
a.grad.data.zero_()
# to reset it. This comes in handy in ML applications (resetting is easier too,
# as we will see later).


#
# Let's take this a step further and make our graph more complex
#
a = torch.tensor([5.], requires_grad=True)
b = torch.tensor([4.], requires_grad=True)
c = a * b
c.retain_grad()  # this will keep the computed gradient for "c"
d = a * c
# ... now our graph is a little longer. We can still use
print(f"d.grad_fn: {d.grad_fn}")
d.backward()  # computes derivative of c*a = a*b*a
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"c.grad: {c.grad}")


#
# We can remove/detach a node from the graph using .detach()
#
a = torch.tensor([5.], requires_grad=True)
b = torch.tensor([4.], requires_grad=True)
c = a * b
c.retain_grad()  # this will keep the computed gradient for "c"
c = c.detach()  # this will detach "c" from the graph (=its "history")
d = a * c
print(f"d.grad_fn: {d.grad_fn}")
d.backward()  # computes derivative of c*a = 20*a
print(f"a.grad: {a.grad}")
print(f"b.grad: {b.grad}")
print(f"c.grad: {c.grad}")

# We can convert tensors that have no gradient information to numpy arrays:
print(f"c.detach().numpy(): {c.detach().numpy()}")

# If you want a code block were gradients are in general not stored, you can
# use
with torch.no_grad():
    # There are no gradients computed/stored in this code block
    e = a * b
    print(e)


#
# Optimizing parameters
#
# We can easily create parameters that we want to optimize/train in PyTorch.
a = torch.tensor([5.], requires_grad=True)
b = torch.tensor([4.], requires_grad=True)
# But now we create a trainable parameter "w" from tensor "a"
w = torch.nn.Parameter(a, requires_grad=True)
# We compute some output tensor "output" given "w" and "b":
output = w * b

# ... but we would like "output" to be close to the value "target":
target = torch.tensor(10.)

# We can use gradient descent to change "w" such that "output" is closer
# to "prediction":
optimizer = torch.optim.SGD([w], lr=0.01)  # use SGD optimizer
for update in range(25):  # for 25 updates
    output = w * b
    loss = (output - target) ** 2  # MSE loss
    loss.backward()  # Calculate gradients
    optimizer.step()  # Do 1 SGD step (changes the value of "w")
    optimizer.zero_grad()  # Reset gradients (or they would be accumulated)
    print(f"update:{update}; loss={loss.item()}; w={w.item()}")


#
# PyTorch and arrays
#
# PyTorch and numpy work together nicely. PyTorch tensors can be arrays and,
# for a large part, used the same way as numpy arrays (indexing, computations,
# arithmetic functions, etc.).
a = torch.arange(5*4).reshape(5, 4)
print(f"a.shape: {a.shape}")
print(f"a.sum(): {a.sum()}")

# We can also create tensors from numpy arrays:
a = torch.tensor(np.arange(5*4).reshape(5, 4))
print(f"a.shape: {a.shape}")
print(f"a.sum(): {a.sum()}")


#
# Utilizing GPU or CPU
#
# This section only works if you have a GPU and installed Pytorch with CUDA.
# PyTorch uses the CPUs as default device. To perform computations on a
# different device, you can either create tensors on this device or "send"
# tensors from one device to another device.
# Syntax for devices: "cpu" for CPU and "cuda:x" for GPU with ID x.
a = torch.tensor([5.], requires_grad=True, device='cpu')  # create on CPU...
a = a.to(device='cuda:0')  # ... and send to GPU0
b = torch.tensor([4.], requires_grad=True, device='cuda:0')  # create on GPU0
c = a * b  # this is computed on GPU0 since the nodes are on GPU0!
print(f"GPU c: {c}")  # Prints the tensor, which is on GPU0

# Since "c" is on the GPU, CPU operations will not work (e.g. numpy):
# print(c.detach().numpy())  # not possible for GPU tensors

# However, we can copy "c" to the CPU easily using .cpu():
print(f"Numpy c: {c.detach().cpu().numpy()}")

# We can compare the speed of matrix computations on CPU vs. GPU:
import time

a = torch.arange(1000*1000, dtype=torch.float32,
                 device='cpu').reshape((1000, 1000))
b = torch.arange(1000*1000, dtype=torch.float32,
                 device='cpu').reshape((1000, 1000))
c = torch.ones_like(a)
with torch.no_grad():
    start_time = time.time()
    for _ in range(1000):
        c = a * b / c.mean()
    c = c.mean()
    end_time = time.time()
print(f"CPU result: {c} ({end_time-start_time} sec)")

a = torch.arange(1000*1000, dtype=torch.float32,
                 device='cuda:0').reshape((1000, 1000))
b = torch.arange(1000*1000, dtype=torch.float32,
                 device='cuda:0').reshape((1000, 1000))
c = torch.ones_like(a)
with torch.no_grad():
    start_time = time.time()
    for _ in range(1000):
        c = a * b / c.mean()
    c = c.mean()
    end_time = time.time()
print(f"GPU result: {c} ({end_time-start_time} sec)")


#
# PyTorch for ML/NN
#

# PyTorch offers many functions for ML and especially NNs. This includes
# tools for creating networks (torch.nn),
# reading data (torch.utils.data.Dataset),
# optimizing parameters (torch.optim), etc..
# We will learn more about this next semester.


###############################################################################
# Tensorflow 1
###############################################################################
# This section is optional since it is for Tensorflow 1. Uncomment the next
# line if you want to execute this code.
exit()
# This is a short introduction to Tensorflow 1 from a past semester.
# Tutorials: https://www.tensorflow.org/get_started/
# Main source for this code:
# https://www.tensorflow.org/get_started/mnist/beginners

# The common abbreviation for tensorflow is tf:
import tensorflow as tf

#
# General usage schema
#

# Create a tf session (here we will define our graph in).
sess = tf.Session()

# Create tf variables
a = tf.constant(5, dtype=tf.float32)
b = tf.constant(3, dtype=tf.float32)
print("a: {}\nb: {}".format(a, b))

# Define your computations (these return symbolic tensors which will be
# added to our graph and computed only when we execute the graph)
c = a + b
print("c: {}".format(c))

# Evaulate (=run) the parts of the graph you want to evaluate:
c_eval = sess.run(c)  # specify tensors to compute as list
print("evaluated c: {}".format(c_eval))
# Session.run() will evaluate the tensors you provide as arguments to it;
# Tensors not used for the evaluations (=not connected to the evaluated nodes)
# will be ignored;

# By default, tf allocates all GPU memory on all GPUs available. We will
# see how to change this later. On some machines this will free the memory:
sess.close()  # if this doesn't work, close your python interpreter


#
# Dynamic memory allocation and device selection
#

# You can restrict the GPUs tf sees by setting this environment variable:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # this will only make GPU 0 visible

import numpy as np  # we will use numpy for its datatypes
import tensorflow as tf

# You can set flags at python start or via the tf config objects passed to the
# session:
config = tf.ConfigProto()  # Create a config object
config.inter_op_parallelism_threads = 2  # max processes for parallel tasks
config.intra_op_parallelism_threads = 2  # max processes for task
config.gpu_options.allow_growth = True  # allocate memory dynamically when needed

# Start tf session
sess = tf.Session(config=config)


#
# Passing values to the tf graph
#

# You can use placeholders to pass values to CPU; you have to specify the shape
# your values will have (empty tuples for scalars); 1 dimension may be set
# variable via None;
a = tf.placeholder(shape=tuple(), dtype=tf.float32)
print("a: {}".format(a))
# Note: This is considered slow in some cases; for best performance use
# background tensorflow queues.

# Create other tf variables
b = tf.constant(3, dtype=tf.float32)
print("b: {}".format(b))

# Define your computations (you can use placeholders like other tensors)
c = a + b
print("c: {}".format(c))

# Execute (=run) the parts of the graph you want to evaluate but you need to
# specify the values of your placeholders via a feed-dictionary:
python_variable = np.float32(5)
c_eval = sess.run(c, feed_dict={a: python_variable})
print("evaluated c: {}".format(c_eval))


#
# Building a 1-layer neural network
#

# Tensorflow already provides some example datasets, in this case we will use
# the MNIST digits dataset (handwritten digits that need to be classified by
# our neural network):
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST data; images will be 1D 784 element-vectors; targets digits
# between 0-9 will be one-hot encoded:
mnist = input_data.read_data_sets(train_dir='MNIST_data', one_hot=True)

# Create the placeholder variables
x = tf.placeholder(tf.float32, [None, 784])  # our placeholder for input data
y_ = tf.placeholder(tf.float32, [None, 10])  # our placeholder for our targets

# Create trainable network variables
w = tf.Variable(tf.zeros([784, 10]))  # our network weights
b = tf.Variable(tf.zeros([10]))  # our biases

# Calculate the network output (with identity activation function)
y = tf.matmul(x, w) + b

# Calculate the loss per sample
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                             logits=y)

# Take mean over losses in minibatch
cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

# Define our optimizer for the weights
optimizer = tf.train.GradientDescentOptimizer(0.5)

# Apply our optimizer on our loss - this would be 1 weight update
update_step = optimizer.minimize(cross_entropy_loss)

# Initialize our trainable variables
initializers = tf.global_variables_initializer()
sess.run(initializers)

# Create our update loop:
for episode in range(10000):
    # Get samples for next minibatch
    batch_x, batch_y_ = mnist.train.next_batch(100)
    
    # Feed minibatch to graph, perform/calculate weight update and loss; we
    # can specify multiple tensors to evaluate by putting them in a list:
    interesting_tensors = [update_step, cross_entropy_loss]
    feed_dictionary = {x: batch_x, y_: batch_y_}
    
    _, train_loss = sess.run(interesting_tensors, feed_dict=feed_dictionary)
    
    print("ep {}\n\ttraining_loss: {}".format(episode, train_loss))


###############################################################################
# Tensorboard
###############################################################################
# Tensorboard is an easy/quick but not reliable tool for visualization
# available through your web browser. You can add tensors as 'summaries' to
# tensorboard and tensorboard will print their values as graphs, histograms,
# images, etc.. You can also view the created computational graph.
# You can start tensorboard from you command line via
# tensorboard --logdir=path/to/log-directory --port 6060
# https://www.tensorflow.org/get_started/summaries_and_tensorboard

# We can add our variables from above to our tensorboard:
tf.summary.scalar('loss', cross_entropy_loss)
tf.summary.histogram('weights', w)
tf.summary.histogram('biases', b)

# After we specified our tensorboard variables, we have to 'merge' them; this
# will give us a tensor that we can pass to tf.Session.run()
merged = tf.summary.merge_all()

# We also have to specify a file for tensorboard to store the values in; we can
# do this via a FileWriter object:
tensorboard_writer = tf.summary.FileWriter('tensorboard/mynetwork', sess.graph)
sess.run(initializers)

# Create our update loop:
for episode in range(100000):
    # Get samples for next minibatch
    batch_x, batch_y_ = mnist.train.next_batch(100)
    
    # Feed minibatch to graph, perform/calculate weight update and loss; we
    # can specify multiple tensors to evaluate by putting them in a list:
    interesting_tensors = [update_step, cross_entropy_loss, merged]
    feed_dictionary = {x: batch_x, y_: batch_y_}
    
    _, train_loss, summary = sess.run(interesting_tensors,
                                      feed_dict=feed_dictionary)
    
    # the evaulated tensorboard summaries have to be added to our tensorboard file
    tensorboard_writer.add_summary(summary, global_step=episode)
    
    print("ep {}\n\ttraining_loss: {}".format(episode, train_loss))

# We can now run
# tensorboard --logdir=path/tensorboard/ --port 6060
# and open
# localhost:6060
# in your browser.


###############################################################################
# More Tensorflow 1
###############################################################################

#
# Assigning tensor calculations to devices
#
a = tf.constant(np.arange(100), dtype=tf.float32)
b = tf.constant(np.arange(100)*2, dtype=tf.float32)

# You can use 'with' blocks to specify which device to use; '/cpu:0' are CPUs,
# '/gpu:0' is GPU 0, '/gpu:1' is GPU2 etc.
with tf.device('/cpu:0'):
    c = a + b
    
    
#
# Namescopes and Variablescopes
#
# You can group your tensors into scopes by using name- or variablescopes:
with tf.name_scope('my_scope'):
    # Tensors in this scope will be grouped in tensorboard graph
    d = c * 2
    # Variables in this scope can be accessed by scope
    my_var = tf.Variable(tf.zeros([10]))

# ...and you can collect your variables by scope
my_scope_vars = tf.trainable_variables()
trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='my_scope')


#
# Example: Getting the gradients
#

from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST data; images will be 1D 784 element-vectors; targets digits
# between 0-9 will be one-hot encoded:
mnist = input_data.read_data_sets(train_dir='MNIST_data', one_hot=True)

# Create the placeholder variables
x = tf.placeholder(tf.float32, [None, 784])  # our placeholder for input data
y_ = tf.placeholder(tf.float32, [None, 10])  # our placeholder for our targets

# Create trainable network variables in a scope
with tf.variable_scope('layer'):
    w = tf.Variable(tf.zeros([784, 10]))  # our network weights
    b = tf.Variable(tf.zeros([10]))  # our biases

# Calculate the network output (with identity activation function)
y = tf.matmul(x, w) + b

# Calculate the loss per sample
cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                             logits=y)

# Take mean over losses in minibatch
cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)

# Define our optimizer for the weights
optimizer = tf.train.GradientDescentOptimizer(0.5)

# Get all trainables in our scope and calculate the gradients
trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='layer')
gradients = tf.gradients(cross_entropy_loss, trainables)

# Define the update step for our gradients and trainables
update_step = optimizer.apply_gradients(zip(gradients, trainables))

# Initialize our trainable variables
initializers = tf.global_variables_initializer()
sess.run(initializers)

# We can add our variables from above to our tensorboard
tf.summary.scalar('loss', cross_entropy_loss)
tf.summary.histogram('values/weights', w)
tf.summary.histogram('values/biases', b)

# Here we add our gradients to tensorboard
[tf.summary.histogram("gradients/{}".format(t.name), values=g)
 for g, t in zip(gradients, trainables)]

# After we specified our tensorboard variables, we have to 'merge' them; this
# will give us a tensor that we can pass to tf.Session.run()
merged = tf.summary.merge_all()

# We also have to specify a file for tensorboard to store the values in; we can
# do this via a FileWriter object:
tensorboard_writer = tf.summary.FileWriter('tensorboard/mynetwork', sess.graph)
sess.run(initializers)

# Create our update loop:
for episode in range(100000):
    # Get samples for next minibatch
    batch_x, batch_y_ = mnist.train.next_batch(100)
    
    # Feed minibatch to graph, perform/calculate weight update and loss; we
    # can specify multiple tensors to evaluate by putting them in a list:
    interesting_tensors = [update_step, cross_entropy_loss, merged]
    feed_dictionary = {x: batch_x, y_: batch_y_}
    
    _, train_loss, summary = sess.run(interesting_tensors,
                                      feed_dict=feed_dictionary)
    
    # the evaulated tensorboard summaries have to be added to our tensorboard file
    tensorboard_writer.add_summary(summary, global_step=episode)
    
    print("ep {}\n\ttraining_loss: {}".format(episode, train_loss))

# We can now run
# tensorboard --logdir=path/tensorboard/ --port 6060
# and open
# localhost:6060
# in your browser.
