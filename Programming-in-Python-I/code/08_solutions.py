# -*- coding: utf-8 -*-
"""08_solutions.py

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

Example solutions for tasks in file 08_tasks.py.

"""

###############################################################################
# 08 numpy
###############################################################################

#
# Task 1
#

# Create a numpy array 'myarray' with shape (8, 5) and 32 bit integer
# values that are initialized with value 1.

# Your code here #
import numpy as np

myarray = np.ones(shape=(8, 5), dtype=np.int32)


#
# Task 2
#

# Extract the values at the index 0 of the first dimension. Then
# overwrite them with ascending integer values (from 0 to 4).
# Afterwards print the array.

# Your code here #
myarray[0]
myarray[0] = np.arange(5)
print(myarray)


#
# Task 3
#

# Multiply the values at the index 2 of the second dimension by 3. Then
# reshape the array to shape (5, 4, 2) and print it.

# Your code here #
myarray[:, 2] *= 3
myarray = myarray.reshape((5, 4, 2))
print(myarray)
