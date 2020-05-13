# -*- coding: utf-8 -*-
"""06_tasks.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.02.2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Tasks for self-study. Try to solve these tasks on your own and
compare your solution to the solution given in the file 06_solutions.py.
See 06_neural_network_inference.py for more information on the tasks.

"""

#
# Task 01
#

# Implement a CNN as PyTorch module, which applies convolutional layers with
# an activation function of your choice. The CNN should furthermore employ
# skip-connections between the convolutional layers. The skip-connection
# arrays should be concatenated (e.g. `DenseNet`) instead of using an
# element-wise sum (e.g. `ResNet`). This can be done by concatenating the
# output channels of the current layer with the output of the layer below and
# feeding this concatenated arrays into the next layer.
# Your __init__ method should take the following arguments:
# n_conv_layers : int ... the number of conv. layers in the network
# n_kernels : int ... the number of kernels in each conv. layer
# kernel_size : int ... the size of the kernels in the conv. layers; you can
# expect this to be an odd integer.
# n_input_channels : int ... number of input channels
#
# Notes:
# You will need to apply padding at the borders of the CNN, otherwise you will
# not be able to concatenate the layer outputs.
# nn.Sequential() will not work here, you will need to store a list of layers
# and iterate over it in the .forward() method, to perform the skip-connections
# at each iteration. Don't forget to register each layer using
# `self.add_module()`.
# Exact design of the CNN is up to your choosing.


import torch
import torch.nn as nn
device = torch.device('cuda:0')  # set to 'cpu' if no GPU available
# Input minibatch with 4 samples, 100 by 100 images, and 3 color channels
input_tensor = torch.arange(4*3*100*100, dtype=torch.float32,
                            device=device).reshape((4, 3, 100, 100))

# Your code here #
