# -*- coding: utf-8 -*-
"""04_solutions.py

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

Example solutions for tasks in file 04_tasks.py. See 04_data_analysis.py for
more information on the tasks.

"""

#
# Task 01
#

# Write a function that converts a color image, represented as numpy array of
# shape (H, W, 3) into a grayscale image, represented as numpy array of shape
# (H, W). The color channels are in order RGB. The numpy array is of type
# np.uint8 with values from 0 to 255.
# The function should take 4 arguments: the numpy array representing the image
# and the 3 contributions of the red, green, and blue channel in this order.
# Default values for contributions of the red, green, and blue channel should
# be r=0.2989, g=0.5870, b=0.1140.
# The function should return a numpy array of type np.unit8.
import numpy as np
example_image = np.random.randint(0, 256, size=(50, 40, 3), dtype=np.uint8)

# Your code here #


def rgb2gray(rgb_array: np.ndarray, r=0.2989, g=0.5870, b=0.1140):
    grayscale_array = (rgb_array[..., 0] * r +
                       rgb_array[..., 1] * g +
                       rgb_array[..., 2] * b)
    grayscale_array = np.round(grayscale_array)
    grayscale_array = np.asarray(grayscale_array, dtype=np.uint8)
    return grayscale_array


#
# Task 02
#

# Find our which images belong to the cluster!
# Hint: we know the cluster is somewhere around (mean=164, std=78). We can use
# this to compute a distance from (mean=164, std=78) to our data points. We
# can then sort the data point by that distance and locate the points close to
# (mean=164, std=78). If a lot of images from one folder are close to that
# point, we have found our culprit!
# You can use np.argsort() to get data point indices sorted by distance and
# np.unique(..., return_counts=True) to get how often individual elements occur
# in an array.

# Load the means, stds, folder_names, file_names from file
# "04_means_stds_precomputed.pklz"
import dill as pkl
import gzip
with gzip.open(f"04_means_stds_precomputed.pklz", 'rb') as fh:
    load_dict = pkl.load(fh)
means = load_dict['means']
stds = load_dict['stds']
folder_names = load_dict['folder_names']

# Your code here #
# Compute squared distance
distance_to_target = (means - 164) ** 2 + (stds - 78) ** 2
# Get indices of data points, sorted by distance
distance_to_target_inds = np.argsort(distance_to_target)
# Get folder_names of 100 closest data point indices
close_folder_names = folder_names[distance_to_target_inds[:100]]
# Check which folder name occurred a lot in the closest points
close_folder_names, counts = np.unique(close_folder_names, return_counts=True)
# Sort folder names by counts
sorted_inds = np.argsort(counts)
sorted_inds = sorted_inds[::-1]  # Indices of larger counts first
close_folder_names[:] = close_folder_names[sorted_inds]
counts[:] = counts[sorted_inds]
# Print folder names and counts
print(f"Close to cluster: {list(zip(close_folder_names, counts))}")
# Answer: Folder 196 is the cluster!


#
# Task 03
#

# Use UMAP to cluster the data instead of t-SNE.
# Follow the guide here:
# https://umap-learn.readthedocs.io/en/latest/clustering.html

#
# Task 04
#

# Normalize a numpy array to range [0, 1]
array = np.random.randint(0, 256, size=(50, 40, 3), dtype=np.uint8)

# Your code here #
array = np.array(array, dtype=np.float)  # uint8 will not work for divisions
array -= array.min()  # -> range [0, max]
array /= array.max()  # -> range [0, 1]

print(f"normalized array is in range [{array.min()}, {array.max()}]")

#
# Task 05
#

# Normalize a numpy array to range [-1, 1]
array = np.random.randint(0, 256, size=(50, 40, 3), dtype=np.uint8)

# Your code here #
array = np.array(array, dtype=np.float)  # uint8 will not work for divisions
array -= array.min()  # -> range [0, max]
array /= (array.max() / 2)  # -> range [0, 2]
array -= 1  # -> range [-1, 1]

print(f"normalized array is in range [{array.min()}, {array.max()}]")

#
# Task 06
#

# Convert an rgb image to grey-scale image using the function in task 01
# then standardize it (the standardized array has mean = 0 and std = 1).
# Hint: using the formula (x-mean)/std
array = np.random.randint(0, 256, size=(50, 40, 3), dtype=np.uint8)

# Your code here #
array = rgb2gray(array)
array = np.array(array, dtype=np.float)  # rgb2gray returns uint8 array which will not work for divisions
mean = np.mean(array)
std = np.std(array)
array -= mean  # to zero mean
array /= std  # to unit variance
print(f'min: {array.min()}, max: {array.max()}')
print(f'mean: {array.mean()}, std: {array.std()}')

