# -*- coding: utf-8 -*-
"""05_solutions.py

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

Example solutions for tasks in file 05_tasks.py.

"""

#
# Task 01
#

# Write a function `sequence_collate_fn(batch_as_list)`, that stacks samples
# containing sequences of varying lengths into mini-batches.
# `batch_as_list` is a list of samples, where each sample is a tuple
# `(sequence_array, label)`.
# `sequence_array` is a numpy array of shape `(seq_length, n_features=3)`,
# where `seq_length` can vary for each sequence but is >= 1.
# `label` is a numpy array of shape `(2, 2)`.
# The mini-batch entries for `sequence_array` should be stacked in the first
# dimension, padding the sequence at the sequence ends with 0-values to the
# same length.
# The mini-batch entries for `label` should be stacked in the first dimension.
# Your function should return a tuple (stacked_sequences, stacked_labels),
# where both tuple entries are PyTorch tensors of datatype torch.float32.
#
# Hint: Get the maximum sequence length within the current mini-batch and
# create a tensor that contains only 0 values and can hold all stacked 
# sequences. Then write the sequences into the tensor.

import numpy as np
import torch
np.random.seed(0)
batch_as_list_1 = [(np.random.uniform(size=(np.random.randint(low=1, high=10), 3)),
                    np.random.uniform(size=(2, 2))) for _ in range(4)]  # mb_size 4
batch_as_list_2 = [(np.random.uniform(size=(np.random.randint(low=1, high=10), 3)),
                    np.random.uniform(size=(2, 2))) for _ in range(3)]  # mb_size 3

# Your code here #


def sequence_collate_fn(batch_as_list: list):
    #
    # Handle sequences
    #
    # Get sequence entries, which are at index 0 in each sample tuple
    sequences = [sample[0] for sample in batch_as_list]
    # Get the maximum sequence length in the current mini-batch
    max_seq_len = np.max([seq.shape[0] for seq in sequences])
    # Allocate a tensor that can fit all padded sequences
    n_seq_features = sequences[0].shape[1]  # Could be hard-coded to 3
    stacked_sequences = torch.zeros(size=(len(sequences), max_seq_len,
                                          n_seq_features), dtype=torch.float32)
    # Write the sequences into the tensor stacked_sequences
    for i, sequence in enumerate(sequences):
        stacked_sequences[i, :len(sequence), :] = torch.from_numpy(sequence)

    #
    # Handle labels
    #
    # Get label entries, which are at index 1 in each sample tuple
    labels = [sample[1] for sample in batch_as_list]
    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32) 
                                  for label in labels], dim=0)
    
    return stacked_sequences, stacked_labels


print(f"batch_as_list_1:")
print(f"original shapes: {[(t[0].shape, t[1].shape) for t in batch_as_list_1]}")
# print(f"stacked: {sequence_collate_fn(batch_as_list_1)}")
print(f"stacked shapes: {[t.shape for t in sequence_collate_fn(batch_as_list_1)]}")
print(f"batch_as_list_2:")
print(f"original shapes: {[(t[0].shape, t[1].shape) for t in batch_as_list_2]}")
# print(f"{sequence_collate_fn(batch_as_list_2)}")
print(f"stacked shapes: {[t.shape for t in sequence_collate_fn(batch_as_list_2)]}")


#
# Task 02
#

# Write a function `one_hot_collate_fn(batch_as_list)`, that stacks samples
# containing one-hot features into mini-batches.
# `batch_as_list` is a list of samples, where each sample is a tuple
# `(one_hot_feat, label)`.
# `one_hot_feat` is a numpy array of shape `(n_features=3,)`,
# containing only the indices of the 1-entries in the one-hot feature vector.
# The full one-hot feature matrix should have shape `(3, 11)`.
# `label` is a numpy array of shape `(2, 2)`.
# The mini-batch entries for `one_hot_feat` should be stacked in the first
# dimension as full one-hot feature vectors.
# The mini-batch entries for `label` should be stacked in the first dimension.
# Your function should return a tuple (stacked_sequences, stacked_labels),
# where both tuple entries are PyTorch tensors of datatype torch.float32.
#
# Hint: First allocate a tensor filled with 0-values that can fit all stacked
# full one-hot feature vectors of the mini-batch. Then use `one_hot_feat` as
# indices to set elements to 1.
# See Programming in Python I for numpy fancy indexing, which also works on
# PyTorch tensors for a large part (08_numpy_pickle.py).

import numpy as np
import torch
np.random.seed(0)
batch_as_list_1 = [(np.random.randint(low=0, high=11, size=(3,)),
                    np.random.uniform(size=(2, 2))) for _ in range(4)]  # mb_size 4
batch_as_list_2 = [(np.random.randint(low=0, high=11, size=(3,)),
                    np.random.uniform(size=(2, 2))) for _ in range(3)]  # mb_size 3

# Your code here #


def one_hot_collate_fn(batch_as_list: list):
    #
    # Handle one-hot features
    #
    # Get one-hot feature entries, which are at index 0 in each sample tuple
    one_hot_indices = [sample[0] for sample in batch_as_list]
    # Allocate a tensor that can fit the stacked full one-hot features
    n_one_hot_features = one_hot_indices[0].shape[0]  # Could be hard-coded to 3
    stacked_one_hot_features = torch.zeros(size=(len(one_hot_indices),
                                                 n_one_hot_features, 11),
                                           dtype=torch.float32)
    
    # Write the indices into the tensor stacked_one_hot_features
    for i, one_hot_inds in enumerate(one_hot_indices):
        stacked_one_hot_features[i, torch.arange(n_one_hot_features),
                                 one_hot_inds] = 1
    
    #
    # Handle labels
    #
    # Get label entries, which are at index 1 in each sample tuple
    labels = [sample[1] for sample in batch_as_list]
    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32)
                                  for label in labels], dim=0)
    
    return stacked_one_hot_features, stacked_labels


print(f"batch_as_list_1:")
print(f"original: {batch_as_list_1}")
print(f"{one_hot_collate_fn(batch_as_list_1)}")
print(f"stacked shapes: {[t.shape for t in one_hot_collate_fn(batch_as_list_1)]}")
print(f"batch_as_list_2:")
print(f"original: {batch_as_list_2}")
print(f"{one_hot_collate_fn(batch_as_list_2)}")
print(f"stacked shapes: {[t.shape for t in one_hot_collate_fn(batch_as_list_2)]}")


#
# Task 03
#

# Same as task 02 but now send only the indices of the one-hot feature vector
# to the GPU and create the stacked full one-hot feature vector on the GPU.
#
# Hint: You can send a tensor to the GPU by specifying `device='cuda:0'`. If
# you do not have a GPU, set device to 'cpu'. See Programming in Python I, file
# 12_tensorflow_pytorch.py for more information about tensor basics.

import numpy as np
import torch
np.random.seed(0)
max_one_hot_ind = 11
batch_as_list_1 = [(np.random.randint(low=0, high=max_one_hot_ind, size=(3,)),
                    np.random.uniform(size=(2, 2))) for _ in range(4)]  # mb_size 4
batch_as_list_2 = [(np.random.randint(low=0, high=max_one_hot_ind, size=(3,)),
                    np.random.uniform(size=(2, 2))) for _ in range(3)]  # mb_size 3

# Your code here #
device = 'cuda:0'  # If you do not have a GPU, set device to 'cpu'


def one_hot_collate_fn(batch_as_list: list):
    """Create full one-hot array on GPU, only send indices"""
    #
    # Handle one-hot features
    #
    # Get one-hot feature entries, which are at index 0 in each sample tuple
    one_hot_indices = [sample[0] for sample in batch_as_list]
    # Allocate a tensor on GPU that can fit the stacked full one-hot features
    n_one_hot_features = one_hot_indices[0].shape[0]  # Could be hard-coded to 3
    stacked_one_hot_features = torch.zeros(size=(len(one_hot_indices),
                                                 n_one_hot_features,
                                                 max_one_hot_ind),
                                           dtype=torch.float32,
                                           device=device)
    
    # We will need the one-hot indices on the GPU. We could send them
    # one-by-one but packing them into one large array before sending them will
    # be faster.
    one_hot_indices = torch.stack([torch.tensor(i) for i in one_hot_indices])
    one_hot_indices = one_hot_indices.to(dtype=torch.long, device=device)
    # Write the indices into the tensor stacked_one_hot_features
    for i, one_hot_inds in enumerate(one_hot_indices):
        # Make sure all the tensors are on the GPU
        stacked_one_hot_features[i,
                                 torch.arange(n_one_hot_features, device=device),
                                 one_hot_inds] = 1
    
    #
    # Handle labels
    #
    # Get label entries, which are at index 1 in each sample tuple
    labels = [sample[1] for sample in batch_as_list]
    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32)
                                  for label in labels], dim=0)
    
    return stacked_one_hot_features, stacked_labels


# Note: For small arrays, the trade-off between GPU bandwidth vs. computation
# time on GPU will not justify this version. However, if you are working with
# large data, these tricks can make your code run-able in reasonable time.
# (See benchmark code below.)
print(f"batch_as_list_1:")
print(f"original: {batch_as_list_1}")
print(f"{one_hot_collate_fn(batch_as_list_1)}")
print(f"stacked shapes: {[t.shape for t in one_hot_collate_fn(batch_as_list_1)]}")
print(f"batch_as_list_2:")
print(f"original: {batch_as_list_2}")
print(f"{one_hot_collate_fn(batch_as_list_2)}")
print(f"stacked shapes: {[t.shape for t in one_hot_collate_fn(batch_as_list_2)]}")


#
# Benchmarking against "simple" implementation without sending indices only
#

def simple_one_hot_collate_fn(batch_as_list: list):
    """Create full one-hot array on CPU, send full one-hot array to GPU"""
    #
    # Handle one-hot features
    #
    # Get one-hot feature entries, which are at index 0 in each sample tuple
    one_hot_indices = [sample[0] for sample in batch_as_list]
    # Allocate a tensor on GPU that can fit the stacked full one-hot features
    n_one_hot_features = one_hot_indices[0].shape[0]  # Could be hard-coded to 3
    stacked_one_hot_features = torch.zeros(size=(len(one_hot_indices),
                                                 n_one_hot_features,
                                                 max_one_hot_ind),
                                           dtype=torch.float32)
    
    # Write the indices into the tensor stacked_one_hot_features
    for i, one_hot_inds in enumerate(one_hot_indices):
        # Make sure all the tensors are on the GPU
        stacked_one_hot_features[i,
                                 torch.arange(n_one_hot_features),
                                 one_hot_inds] = 1
    stacked_one_hot_features = stacked_one_hot_features.to(device=device)
    #
    # Handle labels
    #
    # Get label entries, which are at index 1 in each sample tuple
    labels = [sample[1] for sample in batch_as_list]
    # Convert them to tensors and stack them
    stacked_labels = torch.stack([torch.tensor(label, dtype=torch.float32)
                                  for label in labels], dim=0)
    
    return stacked_one_hot_features, stacked_labels


# Getting the computation times
import timeit
inds = timeit.timeit('(one_hot_collate_fn(batch_as_list_1), one_hot_collate_fn(batch_as_list_2))',
                     number=100, setup="from __main__ import one_hot_collate_fn, batch_as_list_1, batch_as_list_2")
simple = timeit.timeit('(simple_one_hot_collate_fn(batch_as_list_1), simple_one_hot_collate_fn(batch_as_list_2))',
                       number=100,
                       setup="from __main__ import simple_one_hot_collate_fn, batch_as_list_1, batch_as_list_2")
print(f"Sending indices: {inds}; Sending whole array: {simple}")

simple = timeit.timeit('(simple_one_hot_collate_fn(batch_as_list_1), simple_one_hot_collate_fn(batch_as_list_2))',
                       number=100,
                       setup="from __main__ import simple_one_hot_collate_fn, batch_as_list_1, batch_as_list_2")
inds = timeit.timeit('(one_hot_collate_fn(batch_as_list_1), one_hot_collate_fn(batch_as_list_2))',
                     number=100, setup="from __main__ import one_hot_collate_fn, batch_as_list_1, batch_as_list_2")

# `inds` version should be faster at max_one_hot_ind = 110000
print(f"Sending indices: {inds}; Sending whole array: {simple}")
