# -*- coding: utf-8 -*-
"""08_numpy_pickle.py

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

In this file we will learn how to perform fast (vector/matrix) calculations
with numpy and how to save Python objects to files, e.g. via the pickle module.

"""

###############################################################################
# numpy - numerical Python
###############################################################################
# Numpy is the go-to package for fast (matrix) computations in Python.
# It provides a broad range of tools and mathematical functions. The module
# abbreviation is 'np'.
# Homepage: http://www.numpy.org/
# Quick Tutorial: https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
import numpy as np

# Numpy's core datatype is the np.ndarray. These arrays can hold any datatype
# (it will fall back to Python's object datatype in the worst case). A
# np.ndarray can have multiple dimensions. Numpy arrays are supported by the
# pycharm debugger, so you may right-click and select 'view as array' on a
# numpy array in the variable-viewer to get a color-coded view of your array.


#
# Creation of arrays from Python lists
#

# Creation of a numpy array from a Python iterable
my_list = [1, 2, 3, 4]
my_array = np.array(my_list)  # Python list is converted to numpy array
print(my_array)

# Numpy arrays can have multiple dimensions, e.g. 2D matrices. In contrast to
# cumbersome/inefficient nested Python lists, numpy arrays are fast, more
# memory efficient, and provide a better interface to the arrays. However,
# numpy arrays have fixed and consistent array sizes, i.e. all rows have the
# same length and the shape stays the same, a consistent datatype, and are
# not as flexible as Python lists.
my_nested_list = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
my_2d_array = np.array(my_nested_list)
# - take a look at my_2d_array in the debugger!


#
# Datatypes
#

# Numpy arrays are consistent in terms of their datatype, i.e. all elements
# have the same datatype. Numpy will do its best to convert your list values
# into an efficient datatype. Take care that you do not end up with unwanted
# datatypes when converting Python lists to numpy arrays. To be on the safe
# side, you can specify the datatype explicitly. You can specify the number
# of bits in floats/integers using dedicated numpy datatypes:
# np.float16 -> 16bit float; np.int64 -> 64bit integer.

my_list = [1, 2, 3, 4]
int_array = np.array(my_list)
print(int_array, int_array.dtype)  # -> view the datatype of array via .dtype
float32_array = np.array(my_list, dtype=np.float32)  # force 32bit float
print(float32_array, float32_array.dtype)
int_array = np.array(float32_array, dtype=np.int)  # convert to default int
print(int_array, int_array.dtype)

my_list = [1, 2.5, 3, 4]
my_array = np.array(my_list)  # -> default float (=64bit float)
print(my_array, my_array.dtype)

# Strings in numpy can be in object datatype (slow, flexible length) or unicode
# characters of a fixed length (more efficient).
my_list = [1, 2.5, '1', 4]
my_array = np.array(my_list)  # -> unicode character array (default: length 32)
print(my_array, my_array.dtype)

# Numpy also supports boolean datatypes via np.bool:
my_list = [1, 2.5, -3, 0]
my_array = np.array(my_list, dtype=np.bool)  # -> only 0 is "False"
print(my_array, my_array.dtype)

# See https://docs.scipy.org/doc/numpy/user/basics.types.html for more
# information on numpy datatypes.


#
# Creation of arrays by shape
#

# Instead of creating numpy arrays from Python lists, they can be created
# based on a tuple that specifies the desired shape of the array dimensions
# and some initial value for the elements in the array:
ones_1d = np.ones(shape=(5,), dtype=np.int)
# -> 1D array, with 5 elements, initialized with 1
print(ones_1d, ones_1d.shape, ones_1d.dtype)

ones_2d = np.ones(shape=(5, 2), dtype=np.int)
# -> 2D array, with 5x2 elements, initialized with 1
print(ones_2d, ones_2d.shape, ones_2d.dtype)

zeros_2d = np.zeros(shape=(6, 3), dtype=np.float)
# -> 2D array, with 6x3 elements, initialized with 0
print(zeros_2d, zeros_2d.shape, zeros_2d.dtype)

zeros_5d = np.empty(shape=(5, 3, 4, 2, 1), dtype=np.int)
# -> 5D with dimensions (5, 3, 4, 2, 1), faster but not initialized!
print(zeros_5d, zeros_5d.shape, zeros_5d.dtype)

# Getting the shape
print('zeros_5d.shape', zeros_5d.shape)  # getting the shape as tuples
print('zeros_5d.ndim', zeros_5d.ndim)  # getting the number of dimensions
print('zeros_5d.size', zeros_5d.size)  # getting the number of elements


#
# Creation of arrays with ranges of values
#

# Naive and slow:
my_range = np.array(range(5))

# Faster: Creation of ranges of values via np.arange(start, stop, step)
print("np.arange(5)", np.arange(5))
print("np.arange(2, 5)", np.arange(2, 5))
print("np.arange(5, 0, -1)", np.arange(5, 0, -1))

# Creation of ranges of values via np.linspace(start, stop, num)
print("np.linspace(0, 5, 10)", np.linspace(0, 5, 10))


#
# Accessing elements
#

# Indexing is similar to Python lists:
one_dim = np.arange(25)

print('one_dim', one_dim)

# Indexing via integer index
print('one_dim[3]', one_dim[3])
print('one_dim[-3]', one_dim[-3])

# Indexing via slice
print('one_dim[3:10:2]', one_dim[3:10:2])
print('one_dim[3:6]', one_dim[3:6])
print('one_dim[:]', one_dim[:])

# "Fancy indexing": indexing via list/array of indices
print('one_dim[[3, 4, 6, 15]]', one_dim[[3, 4, 6, 15]])

# For more dimensions you can separate indices by commas:
two_dim = np.array([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4]])

print('two_dim[2, 3]', two_dim[2, 3])
print('two_dim[2]', two_dim[2])  # =two_dim[2, :]
print('two_dim[:, 3]', two_dim[:, 3])
print('two_dim[1:2, 3]', two_dim[1:2, 3])

# "Fancy indexing": indexing via boolean masks
mask = np.array([[True, False, False, True],
                 [False, True, False, True],
                 [False, False, True, True]])
print('two_dim[mask]', two_dim[mask])


# Important: Use commas to separate dimensions!
two_dim[:, 2]  # -> all elements along first axis, and index 2 in second axis
two_dim[:][2]  # -> what will this do?

# Writing to an array follows the same rules:
print(two_dim)
two_dim[1, 1] = 0.1
print(two_dim)

# Numpy "broadcasts" values to the target shape:
one_dim = np.arange(15)
print(one_dim)
one_dim[2:5] = 0
print(one_dim)

# Broadcasing for multiple dimensions has to be more specific:
two_dim = np.array([[1, 2, 3, 4],
                    [1, 2, 3, 4],
                    [1, 2, 3, 4]])
print(two_dim)
two_dim[1, 1:4] = 0
print(two_dim)
two_dim[:, 1:3] = np.array([[7, 8]])  # -> will be repeated along axes
print(two_dim)
two_dim[:] = 0
print(two_dim)

# More on indexing, e.g. via boolean masks:
# https://docs.scipy.org/doc/numpy/user/basics.indexing.html


#
# Reshaping arrays
#
a = np.arange(5 * 5)

# Reshaping:
new_shape = (5, 5)  # 2D shape, 5 elements along each dimension
a = a.reshape(new_shape)  # we may also use a=np.reshape(a, new_shape)
print(a.shape)

# Empty dimensions can be inserted via None (see indexing below)
# This will add an empty dimension between dimension 0 and 1:
a = a[:, None, :]
print(a.shape)

# We can let numpy determine one dimension by specifying a shape
# value of -1:
a = a.reshape((5, -1))
print(a.shape)


#
# Appending, concatenating, repeating, tiling
#
a = np.arange(5)

print(np.append(a, np.array([1, 2, 3])))
print(np.concatenate([a, a, a]))
print(np.repeat(a, repeats=5))
print(np.tile(a, reps=5))


#
# Memory consumption and performance
#

# Reshaping and slicing does not copy the original array, it only
# creates another "view" to the data.
# Casting an array to another datatypes via np.array(arrayname, dtype=...)
# will copy the array, as does np.copy(). np.asarray(arrayname, dtype=...)
# will not copy in all cases (e.g. if dtype does not change).

# Numpy functions are fast (in most cases), so prefer them over Python
# functions.
# Copying arrays takes time and memory due to memory allocation. Reusing
# the allocated arrays instead of creating new ones can lead to large
# speedups when dealing with larger arrays.


#
# Operations on arrays
#

# Common mathematical operators work element wise on numpy arrays:
two_dim = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
print('two_dim', two_dim)

new_two_dim = two_dim + 2  # creates a new array
print('new_two_dim', new_two_dim)

two_dim[:] = two_dim + 2  # overwrites content of old array
print('two_dim + 2', two_dim)

two_dim[:] = two_dim > 1 + two_dim * 3  # overwrites content of old array
print('two_dim + two_dim * 3', two_dim)

# Numpy provides a large range of functions, including matrix operations,
# common mathematical functions, saving and loading of numpy arrays, etc.:
# https://docs.scipy.org/doc/numpy


###############################################################################
# pandas - fast&flexible dataframes
###############################################################################
# Pandas allows you to use dataframes, which are arrays that can be indexed via
# keyword, integers, and fancy indexing. These dataframes are similar to
# R dataframes and offer multiple operations on the data, including fast
# reade/write operations on .csv files.
# Pandas is fast for most operations and uses numpy arrays for storing the
# values.
# Homepage: http://pandas.pydata.org/

import pandas as pd

test = pd.DataFrame(data=np.arange(5*5).reshape(5, 5))


###############################################################################
# pickle - saving Python objects to files
###############################################################################
# The pickle module allows you to save Python objects to files. The dill module
# provides additional functions to pickle. It works for many Python objects but
# can lead to errors when trying to pickle Python classes or exotic objects.
# In doubt, always verify if your object type was saved correctly by loading
# and checking it!
# Homepage and help: https://docs.python.org/3/library/pickle.html
import numpy as np
import dill as pickle

some_array = np.arange(20)
some_dict = dict(a=1, b=2, c=3)
some_list = [1, 2, 3]
some_tuple = (4, 5, 6)


def some_function(x):
    return x * 5


# If we want to store (='pickle') multiple objects, it is recommended to use
# a dictionary containing our objects:
my_objects = dict(some_array=some_array, some_dict=some_dict,
                  some_list=some_list, some_tuple=some_tuple,
                  some_function=some_function)

# We need to open files we want to pickle our data to in byte mode 'b'
with open('my_objects.pkl', 'wb') as f:
    # Pickle will store our object into the specified file
    pickle.dump(my_objects, f)

# Now we can load data from this file
with open('my_objects.pkl', 'rb') as f:
    # Pickle load the objects from the file into the dictionary 'data'
    data = pickle.load(f)

print("data['some_array']", data['some_array'])
print("data['some_function'](data['some_array'])",
      data['some_function'](data['some_array']))


###############################################################################
# h5py - Handling larger datafiles efficiently
###############################################################################
# h5py allows you to safe large arrays to your hard-disk or RAM in a compressed
# fashion. Access is similar to a dictionary containing numpy arrays. It offers
# multiple compression and performance settings. You can also use it to store
# compressed numpy arrays in RAM and only uncompress the part you currently
# need. I recommend it as the go-to filetype to store data.
# Homepage: https://www.h5py.org/
import h5py

filename = 'testfile.h5py'
some_larger_array = np.zeros((3, 400, 2, 4), dtype=np.float32)
# h5py files can be opened in read, (over)write, or append mode, as with other
# files. If you want to modify the content of an existing file use 'a' mode.
with h5py.File(filename, 'w') as h5file:
    # h5py files work like dictionaries. You can add entries (="datasets") via
    # create_dataset(name, shape, dtype) or
    # create_dataset(name, shape=None, dtype, data).
    
    # Create datasets based on existing data:
    h5file.create_dataset('some_larger_array', data=some_larger_array)
    # Create empty datasets based on shape:
    h5file.create_dataset('another_array', shape=(5000, 5000), dtype=np.int64)
    # Create datasets based on shape and initialize with some value:
    h5file.create_dataset('another_array_with_zeros', shape=(5000, 5000),
                          dtype=np.int64, fillvalue=0.)
    
    # You can retrieve the contents by using the dataset names as keys and using
    # slices. Slices follow the numpy style:
    
    # Using [:] to get all data:
    some_larger_array_copy = h5file['some_larger_array'][:]
    print(some_larger_array_copy.sum())
    
    # Reading only a part of the data (more memory efficient but slower
    # depending on usage case):
    another_array_slice = h5file['another_array'][0, 1:400]
    
    # You can get typical numpy-like information on the datasets without reading
    # them in (do not use [:] here -> no reading from disk -> fast!):
    print(h5file['some_larger_array'].shape)
    print(h5file['another_array'].shape)
    print(h5file['another_array_with_zeros'].shape)
    
    # You can modify the contents of datasets following numpy syntax (assuming
    # you did not open the file in 'r' mode!):
    print(np.sum(h5file['another_array_with_zeros'][:]))
    h5file['another_array_with_zeros'][1, 4:500] = 1
    print(np.sum(h5file['another_array_with_zeros'][:]))
    
    # Compression can be easily used and fine-adjusted when creating a dataset:
    h5file.create_dataset('some_larger_array_compressed', data=some_larger_array,
                          compression="lzf", chunks=True)
    some_larger_array_another_copy = h5file['some_larger_array_compressed'][:]
    print(some_larger_array_another_copy.sum())
    
    # Be careful when storing non-numpy objects in hdf5 files (e.g. Python
    # strings), as hdf5 might not support that object type or convert it.
    # See https://www.h5py.org/ for more information.
