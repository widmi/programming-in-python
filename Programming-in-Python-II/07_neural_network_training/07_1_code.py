# -*- coding: utf-8 -*-
"""07_1_neural_network_training.py

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

In this file we will learn how to use PyTorch to train our NN layers using
gradient-based methods.
"""

###############################################################################
# PyTorch - Creating a trainable parameter
###############################################################################
# As shown in Unit 06, we can create trainable parameters in PyTorch using
# the torch.nn.Parameter class. It will return a tensor with trainable values
# and by default keep track of the gradients of the tensor.

import os
import numpy as np
np.random.seed(0)  # Set a known random seed for reproducibility
import torch
import torch.nn as nn
torch.random.manual_seed(0)  # Set a known random seed for reproducibility

#
# Create a trainable PyTorch tensor
#
# Create tensor filled with random numbers from uniform distribution [0, 1)
param_values = torch.rand(size=(5, 1), dtype=torch.float32)
# Create trainable parameter from tensor values
trainable_param = nn.Parameter(data=param_values, requires_grad=True)


###############################################################################
# PyTorch - Updating a trainable parameter
###############################################################################
# We can use the automatic gradient computation via autograd to get the
# gradients of a computation w.r.t. our trainable parameter values.

#
# Computing the gradient
#

# Assume we compute a value from our trainable parameters
output = trainable_param.sum() * 2.

# We can get computational graph for gradient computation
print(f"output.grad_fn: {output.grad_fn}")

# We can compute the gradient of `output` w.r.t. a tensor with gradient
# information using autograd:
# (retain_graph=True if we want to compute the same gradients twice)
gradients = torch.autograd.grad(output, trainable_param, retain_graph=True)
print(f"trainable_param gradients: {gradients}")

# Alternatively, we can call the convenience function `.backward()`, which
# will automatically compute the gradients of a scalar tensor w.r.t. all leaves
# of the computational graph, e.g. trainable tensors.
# The gradient values will be accumulated in the `.grad` attribute of the graph
# leaves.
output.backward(retain_graph=True)
# ... the gradients that were computed are now accumulated in the nodes:
print(f"trainable_param.grad: {trainable_param.grad}")

# We have to reset the gradients explicitly, otherwise they will be
# accumulated further:
output.backward()
print(f"trainable_param.grad (2nd time): {trainable_param.grad}")

# Resetting gradient
trainable_param.grad.data.zero_()
print(f"trainable_param.grad (reset): {trainable_param.grad}")


###############################################################################
# PyTorch - Minimizing a loss (=optimizing our parameter values)
###############################################################################
# Having the gradient values of some computation result w.r.t. the contributing
# trainable tensors allows us to use gradient descent methods to minimize the
# result. If this computation result is computed using a loss function, we will
# minimize the loss.
# torch.optim provides different optimization functions, such as stochastic
# gradient descent (SGD) or the adam optimizer. We only have to supply a list
# of trainable parameters and specify optimizer-specific parameters such as the
# learning rate.

# Assume we want our output value to be 1:
output = trainable_param.sum() * 2.
target = torch.tensor(1, dtype=torch.float32)
loss = torch.abs(target - output)  # absolute error as loss function
print(f"Initial:")
print(f"  trainable_param: trainable_param: {trainable_param}")
print(f"  output: {output}; target: {target}; loss: {loss}")

# Assume we want to use the SGD optimizer to optimize our trainable parameter
optimizer = torch.optim.SGD([trainable_param], lr=0.01)

# We can compute the gradients and then perform an update step:
loss.backward()
optimizer.step()
# We can reset the gradients easily using the optimizer:
optimizer.zero_grad()
print(f"After update:")
print(f"  trainable_param: trainable_param: {trainable_param}")
output = trainable_param.sum() * 2.
target = torch.tensor(1, dtype=torch.float32)
loss = torch.abs(target - output)
print(f"  output: {output}; target: {target}; loss: {loss}")

# As you can see, we decreased the loss by optimizing the trainable_param
# values using SGD!

# We can perform multiple update-steps to further decrease our loss:
for update in range(5):
    # Again, we have to:
    # Compute the output
    output = trainable_param.sum() * 2.
    # Compute the loss
    loss = torch.abs(target - output)
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    print(f"Update {update}/5:")
    print(f"  trainable_param: trainable_param: {trainable_param}")
    print(f"  output: {output}; target: {target}; loss: {loss}")

# We can add arbitrary computations to the tensor that should be optimized, as
# long as PyTorch can compute a gradient. Let's say we want to have all values
# in trainable_param to be positive. We could simply add the absolute sum of
# its negative values to the loss.
for update in range(50):
    # Again, we have to:
    # Compute the output
    output = trainable_param.sum() * 2.
    # Compute the loss
    loss = torch.abs(target - output)
    # Add another loss term that minimizes negative values in trainable_param
    loss += trainable_param.clamp(max=0).sum().abs()
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if update % 10 == 0:
        print(f"Update {update}/50:")
        print(f"  trainable_param: trainable_param: {trainable_param}")
        print(f"  output: {output}; target: {target}; loss: {loss}")

# Note how we get closer to a loss of 0 while our main loss and additional loss
# term "compete", as the path we traverse using the SGD updates will depend on
# the values of the gradient computation. These gradient values will be higher
# the higher the losses are, i.e. the update steps will be highest in direction
# of the highest decrease in loss. We could put more emphasis on the term that
# should keep the parameter values positive by using something like:
# loss += trainable_param.clamp(max=0).sum().abs() * 1e2


###############################################################################
# PyTorch - Optimizing parameters of PyTorch modules
###############################################################################
# As shown in Unit 06, we can easily access the trainable parameter values of a
# PyTorch module using `.parameter()`. We can use this to train our model.


# Let's re-use the DSNN implementation from Unit 06:
class DSNN(nn.Module):
    def __init__(self, n_input_features: int, n_hidden_layers: int,
                 n_hidden_units: int, n_output_features: int):
        """Fully-connected feed-forward neural network, consisting of
        `n_hidden_layers` linear layers, using selu activation function in the
        hidden layers.
        
        Parameters
        ----------
        n_input_features: int
            Number of features in input tensor
        n_hidden_layers: int
            Number of hidden layers
        n_hidden_units: int
            Number of units in each hidden layer
        n_output_features: int
            Number of features in output tensor
        """
        super(DSNN, self).__init__()
        
        # We want to use `n_hidden_layers` linear layers, we can solve this
        # with a for-loop:
        hidden_layers = []
        for _ in range(n_hidden_layers):
            # Add linear layer module to list of modules
            layer = nn.Linear(in_features=n_input_features,
                              out_features=n_hidden_units)
            layer.weight.data.normal_(0.0, np.sqrt(1./np.prod(layer.weight.shape[1:])))
            hidden_layers.append(layer)
            # Add selu activation module to list of modules
            hidden_layers.append(nn.SELU())
            n_input_features = n_hidden_units
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Output layer usually is separated to allow easy access to features
        # before output layer
        self.output_layer = nn.Linear(in_features=n_hidden_units,
                                      out_features=n_output_features)
        self.output_layer.weight.data.normal_(0.0, np.sqrt(1./np.prod(self.output_layer.weight.shape[1:])))
    
    def forward(self, x):
        """Apply deep SNN to `x`
        
        Parameters
        ----------
        x: torch.tensor
            Input tensor of shape (n_samples, n_input_features, )
             or (n_input_features, )
        
        Returns
        ----------
        torch.tensor
            Output tensor of shape (n_samples, n_output_features, )
             or (n_output_features, )
        """
        # Apply hidden layers module
        hidden_features = self.hidden_layers(x)
        
        # Apply last layer (=output layer) without selu activation
        output = self.output_layer(hidden_features)
        
        return output


# Create an instance of our DSNN
dsnn = DSNN(n_input_features=5, n_hidden_layers=2, n_hidden_units=8,
            n_output_features=2)

# Create some input and target for our network
input_tensor = torch.arange(5, dtype=torch.float32)
target_tensor = torch.arange(2, dtype=torch.float32)

# .parameters() will return us all trainable parameters of the module,
# including the parameters of the submodules, by default (recurse=True).
# Let's plug them into the SGD optimizer:
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.01)

# Optimize our dsnn model using SGD:
for update in range(50):
    # Compute the output
    output = dsnn(input_tensor)
    # Compute the loss
    # Important: The .backward() method will only work on scalars, so our loss
    # needs to be a scalar:
    loss = torch.abs(target_tensor - output).sum()  # .sum() to create a scalar
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if update % 10 == 0:
        print(f"Update DSNN {update}/50:")
        print(f"  output: {output}; target: {target_tensor}; loss: {loss}")

#
# Perform computations on GPU
# Performing computations on different devices is as simple as showed in
# Unit 06.
#
device = torch.device("cuda:0")

# Create an instance of our DSNN
dsnn = DSNN(n_input_features=5, n_hidden_layers=2, n_hidden_units=8,
            n_output_features=2).to(device=device)

# Create some input and target for our network
input_tensor = torch.arange(5, dtype=torch.float32).to(device=device)
target_tensor = torch.arange(2, dtype=torch.float32).to(device=device)

# .parameters() will return us all trainable parameters of the module,
# including the parameters of the submodules, by default (recurse=True).
# Let's plug them into the SGD optimizer:
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.001)

# Optimize our dsnn model using SGD:
for update in range(50):
    # Compute the output
    output = dsnn(input_tensor)
    # Compute the loss
    # Important: The .backward() method will only work on scalars, so our loss
    # needs to be a scalar:
    loss = torch.abs(target_tensor - output).sum()  # .sum() to create a scalar
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if update % 10 == 0:
        print(f"Update DSNN {update}/50:")
        print(f"  output: {output}; target: {target_tensor}; loss: {loss}")


###############################################################################
# PyTorch - Loss functions
###############################################################################
# Pytorch offers different predefined optimizers and loss functions (see
# slides Unit 07). Always make sure to check the documentation of the optimizer
# and loss function for the correct usage.

#
# Mean squared error loss
# This loss would be used to e.g. have a NN predict numerical values
# (="regression task"). As output activation function for the NN the identity
# function can be used.
#
# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32)
# Let's assume each sample has 3 numerical target values:
target_tensor = torch.rand(size=(5, 3), dtype=torch.float32)
# Our network needs 3 output features to predict the 3 target values
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8,
            n_output_features=3)
# Define our MSE loss (reducing the loss of all samples to a scalar using the
# mean loss over the samples).
loss_function = torch.nn.MSELoss(reduction="mean")
# Use a SGD optimizer
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

# Optimize our dsnn model using SGD:
print("MSE example:")
for update in range(50):
    # Compute the output
    output = dsnn(input_tensor)
    # Compute the loss
    loss = loss_function(output, target_tensor)
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if update % 10 == 0:
        print(f"Update {update}/50:")
        print(f"  output: {output}; target: {target_tensor}; loss: {loss}")

#
# Binary classification task
# BCEWithLogitsLoss can be used to e.g. have a NN predict mutually exclusive
# binary class labels using sigmoid output activation function (="binary
# classification task").
#
# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32)
# Let's assume each sample belongs to class 0 or 1:
target_tensor = torch.tensor([0, 0, 1, 1, 0], dtype=torch.float32).reshape((5, 1))
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8,
            n_output_features=1)
# This BCE implementation expects the values before applying the sigmoid
# activation function for numerical stability:
loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")
# Use a SGD optimizer
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

# Optimize our dsnn model using SGD:
print("BCE example:")
for update in range(50):
    # Compute the output
    output = dsnn(input_tensor)
    # The prediction of our network would be after the sigmoid function
    prediction = torch.sigmoid(output)
    # Compute the loss
    loss = loss_function(output, target_tensor)
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if update % 10 == 0:
        print(f"Update {update}/50:")
        print(f"  output: {output}; prediction: {prediction}; "
              f"target: {target_tensor}; loss: {loss}")


#
# Multi-class classification task
# CrossEntropyLoss can be used to e.g. have a NN predict multiple mutually
# exclusive class labels using softmax output activation function
# (="multi-class classification task").
#
# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32)
# Let's assume each sample either belongs to class 0, 1, or 2:
target_tensor = torch.tensor([0, 2, 2, 1, 0], dtype=torch.long)
# We need 3 output features, one per class
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8,
            n_output_features=3)
# This CE implementation expects the values before applying the softmax
# activation function for numerical stability:
loss_function = torch.nn.CrossEntropyLoss(reduction="mean")
# Use a SGD optimizer
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

# Optimize our dsnn model using SGD:
print("CE example:")
for update in range(50):
    # Compute the output
    output = dsnn(input_tensor)
    # The prediction of the class probabilites of our network would be after
    # the softmax function
    prediction = torch.softmax(output, dim=-1)
    # Compute the loss
    loss = loss_function(output, target_tensor)
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if update % 10 == 0:
        print(f"Update {update}/50:")
        print(f"  output: {output}; prediction: {prediction}; "
              f"target: {target_tensor}; loss: {loss}")


#
# Multi-label classification task
# BCEWithLogitsLoss can be used to e.g. have a NN predict multiple not mutually
# exclusive class labels using a sigmoid output activation function
# (="multi-label classification task").
#
# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32)
# Let's assume each sample belongs to class 0, 1, or 2 or multiple classes:
target_tensor = torch.tensor([[0, 0, 1],
                              [1, 0, 1],
                              [0, 0, 0],
                              [1, 1, 1],
                              [1, 1, 0]], dtype=torch.float32)
# We need 3 output features, one per class
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8,
            n_output_features=3)
# This BCE implementation expects the values before applying the sigmoid
# activation function for numerical stability:
loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")
# Use a SGD optimizer
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

# Optimize our dsnn model using SGD:
print("BCE (2) example:")
for update in range(50):
    # Compute the output
    output = dsnn(input_tensor)
    # The prediction of the class probabilites of our network would be after
    # the sigmoid function
    prediction = torch.sigmoid(output)
    # Compute the loss
    loss = loss_function(output, target_tensor)
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    # Reset the accumulated gradients
    optimizer.zero_grad()
    if update % 10 == 0:
        print(f"Update {update}/50:")
        print(f"  output: {output}; prediction: {prediction}; "
              f"target: {target_tensor}; loss: {loss}")


###############################################################################
# Inspecting training - Tensorboard
###############################################################################
# Inspecting your models during training is very important to understand their
# dynamics and find good models!
# Tensorboard offers a convenient way to monitor the training of your models.
# It supports histograms, line plots, and other logging and visualization
# methods. It can be accessed via web-browser and stores results in a
# lossy manner.
# https://pytorch.org/docs/stable/tensorboard.html
# https://www.tensorflow.org/tensorboard/
from torch.utils.tensorboard import SummaryWriter
import tqdm  # Progress bar

#
# Example for using tensorboard to track losses, weights, and gradients during
# training.
#

# Let's assume a minibatch with 5 samples, 7 input features each
input_tensor = torch.rand(size=(5, 7), dtype=torch.float32).to(device=device)
# Let's assume each sample has 3 numerical target values:
target_tensor = torch.rand(size=(5, 3), dtype=torch.float32).to(device=device)
dsnn = DSNN(n_input_features=7, n_hidden_layers=2, n_hidden_units=8,
            n_output_features=3).to(device=device)
loss_function = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.SGD(dsnn.parameters(), lr=0.1)

# Define a tensorboard summary writer that writes to directory `log_dir`
writer = SummaryWriter(log_dir=os.path.join("results", "experiment_00"))

# Optimize our dsnn model using SGD:
print("Tensorboard example:")
for update in tqdm.tqdm(range(3000), desc="training"):
    # Compute the output
    output = dsnn(input_tensor)
    # Compute the main loss
    main_loss = loss_function(output, target_tensor)
    # Add l2 regularization
    l2_term = torch.mean(torch.stack([(param ** 2).mean()
                                      for param in dsnn.parameters()]))
    # Compute final loss
    loss = main_loss + l2_term * 1e-2
    # Compute the gradients
    loss.backward()
    # Preform the update
    optimizer.step()
    
    if update % 50 == 0:
        # Add losses as scalars to tensorboard
        writer.add_scalar(tag="training/main_loss",
                          scalar_value=main_loss.cpu(),
                          global_step=update)
        writer.add_scalar(tag="training/l2_term",
                          scalar_value=l2_term.cpu(),
                          global_step=update)
        writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(),
                          global_step=update)
        # Add weights as arrays to tensorboard
        for i, param in enumerate(dsnn.parameters()):
            writer.add_histogram(tag=f'training/param_{i}', values=param.cpu(),
                                 global_step=update)
        # Add gradients as arrays to tensorboard
        for i, param in enumerate(dsnn.parameters()):
            writer.add_histogram(tag=f'training/gradients_{i}',
                                 values=param.grad.cpu(),
                                 global_step=update)
    # Reset the accumulated gradients
    optimizer.zero_grad()


# You can now start tensorboard in a terminal using
# tensorboard --logdir results/ --port=6060
# and then open local:6060 in our web-browser

# For CNNs, it can be helpful to plot the CNN kernels (each channel as
# grayscale image). See imshow() in Unit 09 in Programming in Python I.


###############################################################################
# Hints
###############################################################################

#
# Weighting samples
#
# If the dataset is unbalanced (e.g. 10% positive and 90% negative samples), it
# can help to either sample more positive samples per minibatch or,
# alternatively, increase the weight of the positive sample losses. Many loss
# functions allow for weighting classes, such as `weight` in
torch.nn.CrossEntropyLoss()

#
# Learning rate and momentum
#
# The learning rate is a hyper-parameter, i.e. you may have to optimize it to
# find a good learning rate. Learning rates should be in range [0, 1] and will
# depend on the magnitude of the loss values, the task, and the optimizer
# algorithm. The default learning rates in PyTorch are good starting points.
# The same goes for the momentum, which also implicitly alters the learning
# rate and helps to overcome local minima and smooths gradients over samples.

#
# 16 bit computations
#
# If you use 16bit computations, you will probably have to increase the
# parameter for numerical stability if you use the adam optimizer.

#
# Clipping gradients
#
# If training is unstable due to too strong outliers in the gradients, you can
# use gradient clipping to increase stability (at the cost of altering the
# gradients). Clipping value and method are hyper-parameters.
clipping_value = 10
# You can either clip by norm:
torch.nn.utils.clip_grad_norm_(dsnn.parameters(), clipping_value)
# or clip the values directly:
_ = [param.grad.data.clamp_(-clipping_value, clipping_value)
     for param in dsnn.parameters()]

#
# Regularization
#
# PyTorch optimizers already include a parameter `weight_decay`, which is the
# scaling factor of the l2 weight penalty. Different optimizers might benefit
# different versions of l2 weight penalty, so the PyTorch implementation should
# be preferred. You can/should still compute the l2 penalty explicitly for
# plotting in tensorboard.
# Other common regularization methods are adding noise to inputs or features,
# l1 and l2 penalty, or dropout.
# See https://pytorch.org/docs/stable/nn.html#dropout-layers for dropout
# layers.
# Warning: Prefer the dropout option of individual PyTorch modules if it exists
# (some layer classes require specialized dropout versions). Use
# torch.nn.AlphaDropout for networks using SELUs.

#
# Finding good hyper-parameters
#
# Typically, hyper-parameters influence each other and cannot be optimized
# independently from each other. If you have a lot of resources at your
# disposal, you can perform a larger grid-search or random-search over
# different hyper-parameter combinations. However, in practice, you probably
# have to reduce the number of hyper-parameters and the search-space before you
# start a grid-search or random-search. To do this, you can manually check the
# training behaviour and identify settings that would not work (e.g. learning
# rate far too low or high) and exclude those values/use them as boundaries.
# You may also be able to identify which magnitudes of differences in the value
# of a hyper-parameter lead to performance differences and should be
# investigated.
# This is one of the things that make training NNs dependent on experience,
# since you will need to get a feeling for how certain hyper-parameter values
# perform in different settings and combinations (in addition to the
# theoretical backgrounds of the hyper-parameters).


###############################################################################
# Saving trained models
###############################################################################
# PyTorch offers convenient ways of saving and loading models:
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

# Saving trainable parameters of a model
torch.save(dsnn, os.path.join("results", "trained_dsnn.pt"))

# Loading trainable parameters of a model (the module must already be defined)
dsnn = torch.load(os.path.join("results", "trained_dsnn.pt"))


###############################################################################
# Putting it all together
###############################################################################
# We will now create a dataset with random samples that consist of 5 features
# with values between -1 and 1. The target for each sample is the mean of the
# squared feature values. We will train a SNN to solve this task. We will
# combine the materials of Units 05, 06, and 07.
import numpy as np


class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, n_features: int = 5):
        """Create random samples that consist of `n_features` features with
        values between -1 and 1. The target for each sample is the sum of the
        squared feature values.
        """
        self.n_features = int(n_features)
        self.n_samples = int(1e15)
    
    def __get_target__(self, values):
        target = (values ** 2).mean()
        return target
    
    def __getitem__(self, index):
        """ Get a random sample.
        
        Random samples consist of `n_features` features with values between -1
         and 1. The target for each sample is the sum of the squared feature
         values.
        """
        # While creating the samples randomly, we use the index as random seed
        # to get derministic behavior (will return the same sample for the
        # same ID)
        rnd_gen = np.random.RandomState(index)
        
        # Create the random sequence of features
        features = rnd_gen.uniform(low=-1, high=1, size=(self.n_features,))
        features = np.asarray(features, dtype=np.float32)
        target = self.__get_target__(features)
        
        # Let's say that our `index` is the sample ID
        sample_id = index
        # Return the sample, this time with label
        return features, target, sample_id
    
    def __len__(self):
        return self.n_samples


trainingset = RandomDataset(n_features=5)
training_loader = torch.utils.data.DataLoader(trainingset, shuffle=False,
                                              batch_size=4, num_workers=0)
dsnn = DSNN(n_input_features=5, n_hidden_layers=4, n_hidden_units=32,
            n_output_features=1).to(device=device)
loss_function = torch.nn.MSELoss(reduction="mean")
optimizer = torch.optim.Adam(dsnn.parameters(), lr=1e-3)

# Define a tensorboard summary writer that writes to directory `log_dir`
writer = SummaryWriter(log_dir=os.path.join("results", "experiment_01"))

n_updates = 10000  # number of updates to train for
update = 0  # update counter
update_progess_bar = tqdm.tqdm(total=n_updates, desc="updates")
while update < n_updates:
    for data in training_loader:
        mb_features, mb_targets, mb_ids = data
        mb_features = mb_features.to(device=device)
        mb_targets = mb_targets.to(device=device)
        
        # Compute the output
        output = dsnn(mb_features)[:, 0]
        # Compute the main loss
        main_loss = loss_function(output, mb_targets)
        # Add l2 regularization
        l2_term = torch.mean(torch.stack([(param ** 2).mean()
                                          for param in dsnn.parameters()]))
        # Compute final loss
        loss = main_loss + l2_term * 1e-2
        # Compute the gradients
        loss.backward()
        # Preform the update
        optimizer.step()
    
        if update % 50 == 0:
            # Add losses as scalars to tensorboard
            writer.add_scalar(tag="training/main_loss",
                              scalar_value=main_loss.cpu(),
                              global_step=update)
            writer.add_scalar(tag="training/l2_term",
                              scalar_value=l2_term.cpu(),
                              global_step=update)
            writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(),
                              global_step=update)
            writer.add_scalars(main_tag="training/output_target",
                               tag_scalar_dict=dict(output=output[0].cpu(),
                                                    target=mb_targets[0].cpu()),
                               global_step=update)
            # Add weights as arrays to tensorboard
            for i, param in enumerate(dsnn.parameters()):
                writer.add_histogram(tag=f'training/param_{i}', values=param.cpu(),
                                     global_step=update)
            # Add gradients as arrays to tensorboard
            for i, param in enumerate(dsnn.parameters()):
                writer.add_histogram(tag=f'training/gradients_{i}',
                                     values=param.grad.cpu(),
                                     global_step=update)
        # Reset the accumulated gradients
        optimizer.zero_grad()
        
        # Here we could also compute the scores on a validation set or store
        # the currently best model.
        
        # Break if n_updates is reached
        if update >= n_updates:
            break
        
        update_progess_bar.update()
        # Increment update counter
        update += 1
update_progess_bar.close()
torch.save(dsnn, os.path.join("results", "trained_dsnn.pt"))


###############################################################################
# Organization of files
###############################################################################
# How you organize your files is up to you. Often, the classes, functions, and
# scripts are split into multiple files to make the code more modular,
# reusable, and readable. In ML repos you often find a file containing dataset
# code, a file containing training code, a file containing the architectures,
# and a main file that imports and combines the code from the other files.
# It can also be a good choice to put hyper-parameter settings in configuration
# files, e.g. ".json" files.
# In file example_projects.zip you can find an example ML project, putting
# together what we learned so far and applying early stopping.
