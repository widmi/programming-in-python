# -*- coding: utf-8 -*-
"""06_solutions.py

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

Example solutions for tasks in file 06_tasks.py.

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


class CNN(nn.Module):
    def __init__(self, n_input_channels: int, n_conv_layers: int,
                 n_kernels: int,  kernel_size: int):
        """CNN, consisting of `n_hidden_layers` linear layers, using relu
        activation function in the hidden CNN layers.
        
        Parameters
        ----------
        n_input_channels: int
            Number of features channels in input tensor
        n_conv_layers: int
            Number of conv. layers
        n_kernels: int
            Number of kernels in each layer
        kernel_size: int
            Number of features in output tensor
        """
        super(CNN, self).__init__()
        
        layers = []
        n_concat_channels = n_input_channels
        for i in range(n_conv_layers):
            # Add a CNN layer
            layer = nn.Conv2d(in_channels=n_concat_channels,
                              out_channels=n_kernels,
                              kernel_size=kernel_size,
                              padding=int(kernel_size/2))
            layers.append(layer)
            self.add_module(f"conv_{i:03d}", layer)
            # Prepare for concatenated input
            n_concat_channels = n_kernels + n_input_channels
            n_input_channels = n_kernels
        
        self.layers = layers
    
    def forward(self, x):
        """Apply CNN to `x`
        
        Parameters
        ----------
        x: torch.tensor
            Input tensor of shape (n_samples, n_input_channels, x, y)
        
        Returns
        ----------
        torch.tensor
            Output tensor of shape (n_samples, n_output_channels, u, v)
        """
        # Apply layers module
        skip_connection = None
        output = None
        for layer in self.layers:
            # If previous output and skip_connection exist, concatenate
            # them and store previous output as new skip_connection. Otherwise,
            # use x as input and store it as skip_connection.
            if skip_connection is not None:
                inp = torch.cat([output, skip_connection], dim=1)
                skip_connection = output
            else:
                inp = x
                skip_connection = x
            # Apply CNN layer
            output = torch.relu_(layer(inp))
        
        return output


# Create an instance of our CNN
cnn = CNN(n_input_channels=3, n_conv_layers=16, n_kernels=32, kernel_size=3)

# GPU will be much faster here
cnn.to(device=device)
print("\nApplying CNN")
print(f"input tensor shape: {input_tensor.shape}")
output_tensor = cnn(input_tensor)
print(f"output tensor shape: {output_tensor.shape}")
