# -*- coding: utf-8 -*-
"""07_solutions.py

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

Example solutions for tasks in file 07_tasks.py.

"""

#
# Task 01
#

# Implement gradient descent using the gradients computed via .backwards
# without using the PyTorch optimizer. The goal is to get the loss below
# 0.1 by optimizing the trainable weight tensor `weights`. The loss should be
# computed from the output and the target `target_tensor`. The output
# calculation should be done using
# output = input_tensor.matmul(weights)
# and the loss should be a mean squared error loss
# loss = ((target_tensor - output) ** 2.).mean()
# The `input_tensor` and `target_tensor` can be obtained by iterating over the
# `sample_generator()` generator:
# for update, (input_tensor, target_tensor) in enumerate(sample_generator()):
#   # Your code here...
#
# Hint: For gradient descent you will need to compute the gradients of the loss
# w.r.t. the trainable weight tensor. You then multiply the negative(!)
# gradients with a learning rate that is in range [0, 1], which gives you the
# update you have to apply to the trainable weight tensor. Adding the update
# to the trainable weight tensor is one "update step". Choose a number of
# update steps and a suitable learning rate. Don't forget to zero the gradients
# after calling .backward().
# More information gradient descent (and nice analogy for understanding):
# https://en.wikipedia.org/wiki/Gradient_descent

import torch
torch.random.manual_seed(0)  # Set a known random seed for reproducibility


def sample_generator():
    """Function returning a generator to generate random samples"""
    while True:
        input_tensor = torch.rand(size=(7,), dtype=torch.float32)
        target_tensor = torch.stack([input_tensor.sum(), input_tensor.sum()*2])
        yield input_tensor, target_tensor


# The trainable weight tensor we want to optimize
weights = torch.nn.Parameter(torch.rand(size=(7, 2), dtype=torch.float32),
                             requires_grad=True)

# Your code here #

learning_rate = 1e-1
n_updates = 50
# Iterate over samples
for update, (input_tensor, target_tensor) in enumerate(sample_generator()):
    # Compute output
    output = input_tensor.matmul(weights)
    # Compute loss
    loss = ((target_tensor - output) ** 2.).mean()
    # Compute gradients
    loss.backward()
    # Compute weight update
    weight_update = -weights.grad * learning_rate
    # Perform weight update
    weights.data[:] += weight_update
    # Reset gradients
    weights.grad.data.zero_()
    # Print progress
    if update % 5 == 0:
        print(f"Update {update}")
        print(f"  input_tensor: {input_tensor}")
        print(f"  weights: {weights}")
        print(f"  weights.grad: {weights.grad}")
        print(f"  target_tensor: {target_tensor}")
        print(f"  output: {output}")
        print(f"  loss: {loss}")
    # Stop when n_updates updates are reached
    if update >= n_updates:
        break
