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

Tasks for self-study. Try to solve these tasks on your own and
compare your solution to the solution given in the file 05_solutions.py.
See 05_data_loading.py for more information on the tasks.

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
