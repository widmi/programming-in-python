# -*- coding: utf-8 -*-
"""08_solutions.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.10.2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Example solutions for tasks in file 08_tasks.py.

"""

###############################################################################
# 08 numpy
###############################################################################

#
# Task 1
#

# Assume we want to store 2-D array data in a Python list but we do not want
# to use a nested list.
# Create a 1-D Python list `my_1d_list` from the 2-D list `my_2d_list`. Use
# row-major order.
my_2d_list = [['a', 'b', 'c'],  # This is the first row
              ['d', 'e', 'f'],  # This is the second row
              ['g', 'h', 'i'],  # This is the third row
              ['j', 'k', 'l']   # This is the fourth row
              ]

# Your code here #
# Naive and slow solution:
my_1d_list = []
for row in my_2d_list:
    for element in row:
        my_1d_list.append(element)
        
# # List-comprehension solution:
# my_1d_list = [element for row in my_2d_list for element in row]

# # Making use of the + operator for list concatenation:
# my_1d_list = []
# for row in my_2d_list:
#     my_1d_list += row

# Now we want to index the 1-D list `my_1d_list` as if it were a 2-D list.
# Retrieve the element 'f' that would be at the second row and third column
# from the 1-D list. Assume that there are 4 rows and 3 columns.

# Your code here #
row_length = 3
my_1d_list[row_length*1+2]
# Verify with 2-D array:
my_2d_list[1][2]


#
# Task 2
#

# Create a numpy array 'myarray' with shape (8, 5) and 32 bit integer
# values that are initialized with value 1.
import numpy as np

# Your code here #

myarray = np.ones(shape=(8, 5), dtype=np.int32)


#
# Task 3
#

# Extract the values at the index 0 of the first dimension. Then
# overwrite them with ascending integer values (from 0 to 4).
# Afterwards print the array.
myarray = np.ones(shape=(8, 5), dtype=np.float)

# Your code here #
myarray[0]
myarray[0] = np.arange(5)
print(myarray)


#
# Task 4
#

# Multiply the values at the index 2 of the second dimension by 3. Then
# reshape the array to shape (5, 4, 2) and print it.
myarray = np.arange(8*5, dtype=np.float).reshape((8, 5))

# Your code here #
myarray[:, 2] *= 3.
myarray = myarray.reshape((5, 4, 2))
print(myarray)


#
# Task 5
#

# Create an array `values` that contains 64bit integer values from 0 to 10,
# including the value 10. Then create an array `squared` that contains the
# squared values of array `values`.

# Your code here #
values = np.arange(11, dtype=np.int64)
squared = values ** 2


#
# Task 6
#

# Get all indices where the values of array `long_array` are dividable by 3.
# Hint: np.where(boolean_array) will get all indices where boolean_array==True
long_array = np.arange(3*5*4, dtype=np.float).reshape((3, 5, 4))


# Your code here #
np.where((long_array % 3) == 0)


#
# Task 7
#

# Broadcast `subarray` over the first and last dimension of `array`.
# After broadcasting, any access array[n, :, m] should return
# the values [1, 2, 3].
# Hint: You will have to be explicit about which dimensions to broadcast
subarray = np.array([1, 2, 3])
array = np.zeros(shape=(5, 3, 3))

# Your code here #
# `array` has shape (5, 3, 3). To broadcast along first and last dimension,
# we need an array (1, 3, 1).
# Add empty dimension as first and last dimension:
subarray = subarray[None, :, None]
array[:] = subarray
print(f"array[0, :, 0] -> {array[0, :, 0]}")
print(f"array[0, :, 2] -> {array[0, :, 2]}")
print(f"array[2, :, 1] -> {array[2, :, 1]}")
