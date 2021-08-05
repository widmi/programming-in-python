# -*- coding: utf-8 -*-
"""10_torchscript.py

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

In this file we will learn how to use TorchScript to optimize our Python and
PyTorch code further.
"""
# Warning: Some TorchScript features are still experimental and might change
# depending on the PyTorch version.
# This code was tested using PyTorch version 1.8.0 and 1.8.1.

###############################################################################
# Scripting a function
###############################################################################
# Assume we have a simple Python function `foo` that takes two tensors and
# returns the tensor with the largest maximal value

import torch


def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r


tensor_1 = torch.arange(0, 10)
tensor_2 = torch.arange(2, 12)
result = foo(x=tensor_1, y=tensor_2)
print(f"Result of original foo function: {result}")

# The operations on tensors, e.g. 'x.max() > y.max()', are added to the
# computational graph of PyTorch and optimized/evaluated in the background.
# However, the function `foo` is still executed line by line, since Python is
# an interpreted language.
# We can use TorchScript to compile the whole function and serialize and
# optimize it via PyTorch.

#
# Scripting of a function without decorator
#

# We can script the function `foo` by using torch.jit.script:
foo_scripted = torch.jit.script(foo)
result = foo_scripted(x=tensor_1, y=tensor_2)
# It will behave the same as `foo`
print(f"Result of scripted foo function: {result}")
# ...but it is actually a compiled torch.jit.ScriptFuncion
print(f"Type of original foo function: {type(foo)}")
print(f"Type of scripted foo function: {type(foo_scripted)}")

# We can inspect the compiled code:
print(foo_scripted.code)


#
# Scripting of a function using a decorator
#

# Alternatively, we can use torch.jit.script as a decorator to script our
# function

@torch.jit.script
def foo2(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r


result = foo2(x=tensor_1, y=tensor_2)
print(f"Result of scripted foo2 function: {result}")
print(f"Type of scripted foo2 function: {type(foo2)}")


#
# Static typing
# TorchScript expects static typing, dynamic changes of variable data types are
# not supported.
#

# This would raise an exception since `r` changes type:
#@torch.jit.script
def an_error(x):
    if x:
        r = torch.rand(1)
    else:
        r = 4
    return r


#
# Using non-Tensor objects
# Many Python objects besides PyTorch Tensors are supported in TorchScript.
# However, their types have to be explicitly stated. This can e.g. be done by
# using the typing module and annotation.
#

# At the moment, only from typing import ... is supported
# (not e.g. `typing.List`)
from typing import List, Tuple


# Assume we want a function that takes an integer and a tuple containing two
# Tensors as arguments and returns a Tensor:
@torch.jit.script
def mixed_function(x: int, tup: Tuple[torch.Tensor,
                                      torch.Tensor]) -> torch.Tensor:
    t0, t1 = tup
    return t0 + t1 + x


result = mixed_function(5, (torch.tensor(5), torch.tensor(5)))
print(f"mixed_function(5, (torch.tensor(5), torch.tensor(5))) -> {result}")

# Now we have left the flexibility of native Python and wrong data types will
# lead to exceptions:
# mixed_function(5.0, (torch.tensor(5), torch.tensor(5)))


#
# Annotation of lists
# Empty lists are assumed to be List[Tensor] and empty dicts Dict[str, Tensor].
# To instantiate an empty list or dict of other types, use annotation.
#

@torch.jit.script
def list_generator(x: torch.Tensor) -> List[Tuple[int, int]]:
    # Specify that empty `my_list` will hold tuples, where each tuple contains
    # two int objects
    my_list: List[Tuple[int, int]] = []
    
    # Create the list
    while x < 5:
        my_list.append((int(x), int(x+1)))
        x += 1
        
    # Return the list
    return my_list


result = list_generator(torch.tensor(0))
print(f"list_generator(torch.tensor(5)) -> {result}")


###############################################################################
# Scripting a torch.nn.Module instance
###############################################################################
# We can script instances of classes derived from torch.nn.Module. This will
# automatically script the forward() method and all functions called by
# forward(). See https://pytorch.org/docs/stable/jit.html for how to include or
# exclude methods from scripting explicitly.

#
# Scripting a torch.nn.Module instance
# Scripting of torch.nn.Module instances can be done using torch.jit.script
#

class MyModule(torch.nn.Module):
    def __init__(self, some_argument):
        super(MyModule, self).__init__()
        self.a = some_argument
        
    def forward(self, x: int, tup: Tuple[torch.Tensor,
                                         torch.Tensor]) -> torch.Tensor:
        """The forward method and all members and functions used by it will be
        scripted and need to follow the TorchScript restrictions for scripting
        functions. This also holds for members used here, e.g. self.a needs to
        remain static within the forward() method.
        """
        t0, t1 = tup
        return t0 + t1 + x + self.a


my_module_scripted = torch.jit.script(MyModule(5))
result = my_module_scripted(5, (torch.tensor(5), torch.tensor(5)))
print(f"my_module_scripted(5, (torch.tensor(5), torch.tensor(5))) -> {result}")


#
# Example: Scripting the simple RNN we used in Unit06
#
import timeit
from torch import nn


# Create a PyTorch module `RNN` for a simple RNN
class RNN(nn.Module):
    def __init__(self, n_input_features: int, n_hidden_units: int,
                 n_output_features: int):
        """Simple RNN consisting of one recurrent fully-connected layer with
        sigmoid activation function, followed by one fully-connected
        feed-forward output layer.
        
        Parameters
        ----------
        n_input_features: int
            Number of features in input tensor
        n_hidden_units: int
            Number of units in the hidden layer
        n_output_features: int
            Number of features in output tensor
        """
        super(RNN, self).__init__()
        # Create a fully-connected layer that expects the concatenated forward
        # features n_input_features and recurrent features n_hidden_units
        self.hidden_layer = nn.Linear(
                in_features=n_input_features + n_hidden_units,
                out_features=n_hidden_units)
        
        self.output_layer = nn.Linear(in_features=n_hidden_units,
                                      out_features=n_output_features)
        
        # We need some initial value for h_{t-1} at t=0. We will just use a
        # 0-vector for this:
        self.h_init = torch.zeros(size=(n_hidden_units,), dtype=torch.float32)
    
    def forward(self, x):
        """Apply RNN to `x`
        
        Parameters
        ----------
        x: torch.tensor
            Input tensor of shape (n_sequence_positions, n_input_features)
        
        Returns
        ----------
        torch.tensor
            Output tensor of shape (n_output_features,)
        """
        # Get initial h_{t-1} for t = 0
        h = self.h_init
        
        # We will use Python for-loop to loop over the sequence positions.
        for x_t in x:
            # Concatenate c_t and h_{t-1}
            inp_t = torch.cat([x_t, h])
            # Compute new h_t from c_t and h_{t-1}
            h = self.hidden_layer(inp_t)
            # Using sigmoid here is the reason why vanishing gradient will make
            # training very difficult (and probably will fail). In practice, we
            # use an LSTM.
            h = torch.sigmoid(h)
        
        # Last layer only sees h from last timestep
        output = self.output_layer(h)
        
        return output


# Create an instance of our RNN
rnn = RNN(n_input_features=8, n_hidden_units=32, n_output_features=1)

# Create a scripted version of our RNN instance:
rnn_scripted = torch.jit.script(rnn)

# Create some long input sequence with length 5000 and 8 features per position
input_tensor = torch.arange(5000*8, dtype=torch.float32).reshape((5000, 8))
print("\nRNN")
print(f"input tensor shape: {input_tensor.shape}")
output_tensor = rnn(input_tensor)
print(f"output tensor rnn: {output_tensor}")
output_tensor = rnn_scripted(input_tensor)
print(f"output tensor rnn_scripted: {output_tensor}")

# Compare speed (if you run into segmentation faults, decrease the sequence
# length)
rnn_time = timeit.timeit('rnn(input_tensor)', number=10,
                         setup="from __main__ import rnn, input_tensor")
rnn_scripted_time = timeit.timeit(
        'rnn_scripted(input_tensor)', number=10,
        setup="from __main__ import rnn_scripted, input_tensor")

# `rnn_scripted_time` version should be faster
print(f"Runtime original rnn: {rnn_time}")
print(f"Runtime scripted rnn: {rnn_scripted_time}")
# On my machine (CPU):
# Runtime original rnn: 3.075211859999399
# Runtime scripted rnn: 1.9866709439957049


###############################################################################
# Tracing
###############################################################################
# Tracing will execute Python code and create a graph from the execution, using
# the static program flow observed during execution. The traced code will
# therefore not reevaluate conditions or number of loop iterations of the
# Python code.


def foo(x, y):
    if x.max() > y.max():
        r = x
    else:
        r = y
    return r


tensor_1 = torch.arange(0, 10)
tensor_2 = torch.arange(2, 12)
result = foo(x=tensor_1, y=tensor_2)
print(f"Result of original foo function: {result}")

foo_traced = torch.jit.trace(foo, example_inputs=(tensor_1, tensor_2))
result = foo_traced(x=tensor_1, y=tensor_2)
print(f"Result of traced foo function: {result}")

# The if/else condition in the traced code is now static:
tensor_1 = torch.arange(5, 15)
tensor_2 = torch.arange(2, 12)
result = foo(x=tensor_1, y=tensor_2)
print(f"Result of original foo function: {result}")

result = foo_traced(x=tensor_1, y=tensor_2)
print(f"Result of traced foo function: {result}")


###############################################################################
# Saving, loading, and inspecting TorchScript code
###############################################################################

#
# Saving and loading of TorchScript code is straight-forward:
#
foo_traced.save("10_torch_script_module.pt")
loaded_script_module = torch.jit.load("10_torch_script_module.pt")
# We can also import this TorchScript program in Python-free high-performance
# environments, like C++: https://pytorch.org/tutorials/advanced/cpp_export.html

#
# We can inspect the additional information provided by the TorchScript objects
#
print()
print("#")
print("# Inspecting scripted function:")
print("#")
print("foo_scripted.graph:")
print(foo_scripted.graph)
print()
print("foo_scripted.code:")
print(foo_scripted.code)
print()
print("#")
print("# Inspecting traced function:")
print("#")
print("foo_traced.graph:")
print(foo_traced.graph)
print()
print("foo_traced.code:")
print(foo_traced.code)

# Further reading:
# https://pytorch.org/docs/stable/jit.html
# https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html
