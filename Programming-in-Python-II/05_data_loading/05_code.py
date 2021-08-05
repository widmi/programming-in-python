# -*- coding: utf-8 -*-
"""05_data_loading.py

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

In this file we will learn how to use PyTorch to read our dataset.
"""

###############################################################################
# Standardized interfaces, reserved methods, overloading, and chaining
###############################################################################

# A lot of the comfort of Python (and other higher-level languages) comes from
# using standardized interfaces between functions, classes, and operators. It
# allows us to write modular code and hide a lot of the details.

# Example: The len() function works on different types of objects (strings,
# lists, tuples, ...) - how does it do that? It simply calls the .__len__()
# method of the objects!

a = [1, 2, 3]
print(f"len(a): {len(a)}")
print(f"a.__len__() = {a.__len__()}")


# We can create our own class that has its own .__len__() method:
class AlwaysLength5:
    def __init__(self, string):
        self.string = string
    
    def __len__(self):
        """Our custom length function"""
        return 5


a = AlwaysLength5('abc')
print(f"len(a): {len(a)}")


# We can also create a class that supports the "with" statement:
class Book:
    def __init__(self, book_name):
        self.book_name = book_name
        
    def __enter__(self):
        """.__enter__() is called when starting the "with" block"""
        print(f"Opening book {self.book_name}")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """.__exit__() is called when leaving the block (also in case of
        exceptions, hence the exception arguments)"""
        print(f"Closing book {self.book_name}")


with Book('Pattern Recognition') as my_book:  # creates instance and calls .__enter__()
    print('reading')
    # Leaving block calls .__exit__()


# Operators also use this mechanism.
# Example: In Python the + operator uses the reserved method .__add__(other).
# .__add__(other) should return the result of "+" applied to the class instance
# and the object `other`:

a = 1
b = 2
print(f"{a}+{b}={a+b}")


# We can customize the reserved .__add__() method, which is called
# "overloading" for a new class:
class NewInt(int):
    """New class, derived from int class"""
    def __add__(self, other):
        """A custom add method"""
        return self.real + other - 1


a = NewInt(1)
b = 2

print(f"{a}+{b}={a+b}")
print(f"{b}+{a}={b+a}")
print(f"because type(a)={type(a)}")


# There are good practices for overloading, e.g. the '+' operator should
# return the same user-defined type to allow for chaining (=multiple operators
# in one line) of operators:
class NewInt(int):
    """New class, derived from int class"""
    def __add__(self, other):
        """A custom add method"""
        return NewInt(self.real + other - 1)


# Other operators: https://docs.python.org/3/library/operator.html

# PyTorch makes heavy use of standardized interfaces between objects and
# overloading, as we will see in the following Units.


###############################################################################
# PyTorch data handling - Introduction
###############################################################################

import numpy as np
np.random.seed(0)  # Set a known random seed for reproducibility
# More on random seed:
# https://en.wikipedia.org/wiki/Random_seed
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html#numpy.random.RandomState
# "A fixed seed and a fixed series of calls to ‘RandomState’ methods using the
# same parameters will always produce the same results up to roundoff error..."
# Example: The random number generated below should be the same for you because
# we set np.random.seed(0)
a = np.random.uniform()  # should be 0.5488135039273248
b = np.random.uniform()  # should be 0.7151893663724195
c = np.random.uniform()  # should be 0.6027633760716439

#
# Our dataset
#

# Let's say our dataset consists of 25 samples. Each sample is represented by
# a feature vector of shape (10,), meaning we have 10 features describing one
# sample.
# Create 25 random samples, each with 10 features, with values in [-1, 1]
our_samples = np.random.uniform(low=-1, high=1, size=(25, 10))


#
# torch.utils.data.Dataset
#

# The best way to utilize the convenient PyTorch data loading pipelines, is by
# representing your dataset as class derived from the PyTorch dataset class
# torch.utils.data.Dataset.

# Now we will represent our dataset using PyTorch
import torch
from torch.utils.data import Dataset


class Simple1DRandomDataset(Dataset):
    def __getitem__(self, index):
        """ Here we have to define a method to get 1 sample
        
        __getitem__() should take one argument: the index of the sample to get
        """
        # Now we have to specify how to get the sample at `index`:
        sample_features = our_samples[index]
        # It's a good idea to return the index/ID of the sample for debugging
        sample_id = index  # let's say that our `index` is the sample ID
        # And we have to return the sample
        return sample_features, sample_id

    def __len__(self):
        """ Optional: Here we can define the number of samples in our dataset
        
        __len__() should take no arguments and return the number of samples in
        our dataset
        """
        n_samples = len(our_samples)
        return n_samples


# Done! We have represented our dataset as PyTorch dataset!

# To use it, we have to create an instance
our_dataset = Simple1DRandomDataset()
print(f"our_dataset: {our_dataset}")
print(f"number of samples in our_dataset: {len(our_dataset)}")


#
# torch.utils.data.DataLoader
#

# Having our PyTorch compatible dataset, we can use the PyTorch DataLoader
# class to read the samples in minibatches
from torch.utils.data import DataLoader

our_dataloader = DataLoader(our_dataset,  # we want to load our dataset
                            shuffle=True,  # shuffle the order of our samples
                            batch_size=4,  # stack 4 samples to a minibatch
                            num_workers=0  # no background workers for now
                            )

# We can loop over our dataloader and it will loop over the dataset once,
# returning minibatches that contain our samples
for data in our_dataloader:
    mb_sample_features, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids} with features {mb_sample_features}\n")

print("Done loading every sample 1 time!")

# As we can see, the arrays have been stacked to mini-batches and converted to
# PyTorch tensors, following the original datatype of the arrays:
print(f"ID dtype: {mb_sample_ids.dtype}")
print(f"features dtype: {mb_sample_features.dtype}")

# This stacking is done by introducing a new first dimension. We will later see
# how to write custom stacking functions.


#
# torch.utils.data.Subset
#

# As we have already learned, it is important to split the training set from the
# test set (and optionally validation set). Unless you need something very fancy,
# the PyTorch subset does exactly that.
from torch.utils.data import Subset

# Let's assign 1/5th of our samples to a test set, 1/5th to a validation set, and
# the remaining 3/5th to a training set. We will use random splits.
n_samples = len(our_dataset)
# Shuffle integers from 0 n_samples to get shuffled sample indices
shuffled_indices = np.random.permutation(n_samples)
testset_inds = shuffled_indices[:int(n_samples/5)]
validationset_inds = shuffled_indices[int(n_samples/5):int(n_samples/5)*2]
trainingset_inds = shuffled_indices[int(n_samples/5)*2:]

# IMPORTANT: At this point, in a real ML project, you should save your subset
# indices to a file (e.g. .csv, .npz, or .pkl) for documentation and
# reproducibility!

# Create PyTorch subsets from our subset-indices
testset = Subset(our_dataset, indices=testset_inds)
validationset = Subset(our_dataset, indices=validationset_inds)
trainingset = Subset(our_dataset, indices=trainingset_inds)

# Create dataloaders from each subset
test_loader = DataLoader(testset,  # we want to load our dataset
                         shuffle=False,  # shuffle for training
                         batch_size=1,  # 1 sample at a time
                         num_workers=0  # no background workers
                         )
validation_loader = DataLoader(validationset,  # we want to load our dataset
                               shuffle=False,  # shuffle for training
                               batch_size=4,  # stack 4 samples to a minibatch
                               num_workers=2  # 2 background workers
                               )
training_loader = DataLoader(trainingset,  # we want to load our dataset
                             shuffle=True,  # shuffle for training
                             batch_size=4,  # stack 4 samples to a minibatch
                             num_workers=2  # 2 background workers
                             )

# Let's try out our data loaders
print(f"testset ({len(testset)} samples)")
for data in test_loader:
    mb_sample_features, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}")
print(f"validationset ({len(validationset)} samples)")
for data in validation_loader:
    mb_sample_features, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}")
print(f"trainingset ({len(trainingset)} samples)")
for data in training_loader:
    mb_sample_features, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}")


###############################################################################
# PyTorch data handling - Dataset details
###############################################################################

# In realistic scenarios, our dataset class might look a lot more complex. Not
# only can we add a custom __init__ function to it but reading a sample can be
# much more sophisticated than just indexing.
# Let's for example create a dataset class that creates a simulated dataset of
# 1e15 many random samples. Each sample is a sequence, described by a 2D
# array of input features of shape (sequence_length, n_features), with
# random values in range [-1, 1]. Each sample belongs to either a positive or
# negative class. In positive-class samples, we will implant the pattern
# [0, 1, 0, 1, 0, 1] in feature 0 at the beginning of the sequences.
# Since 1e15 is a large number, we do not want to hold that dataset in the
# RAM. Instead, we will create the random samples on-the-fly.


class RandomSeqDataset(Dataset):
    def __init__(self, sequence_length: int = 15, n_features: int = 9):
        """Here we define our __init__ method. In this case, we will take two
        arguments, the sequence length `sequence_length` and the number of
        features per sequence position `n_features`.
        """
        # super().__init__()  # Optional, since Dataset.__init__() is a no-op
        self.sequence_length = int(sequence_length)
        self.n_features = int(n_features)
        self.n_samples = int(1e15)
        # We'll stay in float32, as typical for GPU applications
        self.pattern = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)
        
    def __getitem__(self, index):
        """ Here we have to create a random sample and add the signal in
        positive-class samples. Positive-class samples will have a label "1",
        negative-class samples will have a label "0".
        """
        # While creating the samples randomly, we use the index as random seed
        # to get deterministic behavior (will return the same sample for the
        # same ID)
        rnd_gen = np.random.RandomState(index)
        
        # Create the random sequence of features
        sample_features = rnd_gen.uniform(low=-1, high=1,
                                          size=(self.sequence_length,
                                                self.n_features))
        # Stay in float32
        sample_features = np.asarray(sample_features, dtype=np.float32)
        # Set every 2nd sample to positive class
        sample_class_label = index % 2
        # Implant pattern in positive class
        if sample_class_label:
            sample_features[:len(self.pattern), 0] = self.pattern
            
        # Here we could add pre-processing pipelines and/or normalization
        # ...
        
        # Let's say that our `index` is the sample ID
        sample_id = index
        # Return the sample, this time with label
        return sample_features, sample_class_label, sample_id
    
    def __len__(self):
        """ Optional: Here we can define the number of samples in our dataset

        __len__() should take no arguments and return the number of samples in
        our dataset
        """
        return self.n_samples


# Again, we can use a PyTorch dataloader to load our dataset:
trainingset = RandomSeqDataset(sequence_length=15, n_features=9)
# Note: Shuffling would break with 1e15 sample indices
training_loader = DataLoader(trainingset, shuffle=False, batch_size=4,
                             num_workers=0)

for i, data in enumerate(training_loader):
    mb_sample_features, mb_sample_labels, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}, labels {mb_sample_labels}, "
          f"feature shape {mb_sample_features.shape}\n")
    if i > 500:
        # Ok, let's not do the full loop (unless you want to heat your CPUs)
        break


###############################################################################
# PyTorch data handling - Mini-batch stacking via collate_fn
###############################################################################

# By default, the PyTorch DataLoader class will convert the sample arrays to
# tensors and stack them by introducing a new first dimension.
# In some cases, stacking arrays to mini-batches will not work that easily. For
# example if you want to stack samples with different shapes, such as sequences
# of variable length or images of variable width/height.
# In other cases, you may want to perform stacking to a mini-batch in a very
# memory-efficient way, e.g. pre-allocate arrays for one-hot features.
# We can implement our own stacking function by passing the argument
# `collate_fn` to our dataloader, which should be our custom stacking function.

#
# Simple example: No stacking of mini-batch
#

def no_stack_collate_fn(batch_as_list: list):
    """Function to be passed to torch.utils.data.DataLoader as collate_fn
    
    collate_fn will receive one argument: A list of the samples in the
    minibatch, as they were returned by the __getitem__ method of a PyTorch
    Dataset. Keep in mind that each sample is represented by a tuple,
    containing e.g. the features, labels, and IDs.
    In this example, instead of stacking the samples and converting them to
    tensors, the samples will be individually converted to tensors and packed
    into a list instead.
    """
    # Number of entries per sample-tuple (e.g. 3 for features, labels, IDs)
    n_entries_per_sample = len(batch_as_list[0])
    # Go through all entries in all samples, convert to tensors and put them
    # in lists
    list_batch = [[torch.tensor(sample[entry_i]) for sample in batch_as_list]
                  for entry_i in range(n_entries_per_sample)]
    # Return the mini-batch
    return list_batch

# Let's try our `collate_fn` on our sequence dataset:

training_loader = DataLoader(trainingset, shuffle=False, batch_size=4,
                             num_workers=0, collate_fn=no_stack_collate_fn)

for i, data in enumerate(training_loader):
    mb_sample_features, mb_sample_labels, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}, labels {mb_sample_labels}\n")
    if i > 5:
        break

print("Our mini-batch is now a list of tensors instead of one stacked tensor!")


#
# Example: Stacking and conversion of numpy arrays only
#

# Let's assume we want to return a sample tuple that consists of
# (features, labels, ID) but ID cannot be converted to a tensor (e.g. because
# it is a string).

# As example, we can inherit from our RandomSeqDataset class and overwrite the
# __getitem__ method such that ID is a string:
class NewRandomSeqDataset(RandomSeqDataset):
    def __getitem__(self, index):
        """Overwrites __getitem__ from RandomSeqDataset class"""
        # Call original __getitem__ from RandomSeqDataset class
        original_sample = super().__getitem__(index)
        sample_features, sample_class_label, sample_id = original_sample
        # Return the sample entries but convert sample_id to string
        return sample_features, sample_class_label, str(sample_id)


# Since strings cannot be converted to tensors or stacked, we have to stack and
# convert the sample entries only if they are stack-able and convertible to
# tensors.
# We can start by building the logic of converting-and-stacking-if-possible:
def stack_or_not(something_to_stack: list):
    """This function will attempt to stack `something_to_stack` and convert it
    to a tensor. If not possible, `something_to_stack` will be returned as it
    was.
    """
    try:
        # Convert to tensors (TypeError if fails)
        tensor_list = [torch.tensor(s) for s in something_to_stack]
        # Try to stack tensors (RuntimeError if fails)
        stacked_tensors = torch.stack(tensor_list, dim=0)
        return stacked_tensors
    except (TypeError, RuntimeError):
        return something_to_stack


# And now, we use it in our collate_fn
def stack_if_possible_collate_fn(batch_as_list: list):
    """Function to be passed to torch.utils.data.DataLoader as collate_fn
    
    Will stack samples to mini-batch if possible, otherwise returns list
    """
    # Number of entries per sample-tuple (e.g. 3 for features, labels, IDs)
    n_entries_per_sample = len(batch_as_list[0])
    # Go through all entries in all samples and apply our stack_or_not()
    list_batch = [stack_or_not([sample[entry_i] for sample in batch_as_list])
                  for entry_i in range(n_entries_per_sample)]
    # Return the mini-batch
    return list_batch


# We create a dataset instance of our new NewRandomSeqDataset
trainingset = NewRandomSeqDataset(sequence_length=15, n_features=9)
# And load it via the PyTorch dataloader with our stack_if_possible_collate_fn
training_loader = DataLoader(trainingset, shuffle=False, batch_size=4,
                             num_workers=0,
                             collate_fn=stack_if_possible_collate_fn)

for i, data in enumerate(training_loader):
    mb_sample_features, mb_sample_labels, mb_sample_ids = data
    print(f"Loaded samples {mb_sample_ids}, labels {mb_sample_labels}, "
          f"feature shape {mb_sample_features.shape}\n")
    if i > 5:
        break

print("Our mini-batch is now a stacked tensors where possible and a list"
      " otherwise!")


###############################################################################
# PyTorch data handling - More examples
###############################################################################

# See 05_tasks.py/05_solutions.py files for:
# Stacking multiple sequences of different length into a minibatch, using
# padding.
# Sending one-hot indices to the GPU and unpacking them into a minibatch.


###############################################################################
# PyTorch data handling - Hints
###############################################################################

# Setting global random seeds via e.g. np.random.seed(0) will not make the
# order of samples or randomness in __getitem__() deterministic if the
# DataLoader is using multiple workers.
# Work-around for __getitem__() randomness: You can create a random generator
# object in __getitem__(), which is using the sample index as random seed to
# always get the same random behavior for samples with the same index.
#
# Don't forget: Pre-allocating memory is faster than first creating small
# arrays and them stacking them (unless PyTorch can optimize it using its
# magic).
#
# Using too many multiple worker processes will result in additional system
# overhead (managing the processes) and slow your program down! Always check
# the CPU utilization!
#
# It's good practice to save the indices of dataset splits in dedicated files.
#
# For more information on hdf5, numpy, and pickle containers, please see the
# materials of Programming in Python I.
