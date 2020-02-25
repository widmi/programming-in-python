# -*- coding: utf-8 -*-
"""01_solutions.py

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

Example solutions for tasks in file 01_tasks.py.

"""

###############################################################################
# 01 Tuples, Lists, Indices, Dictionaries, Slices
###############################################################################

#
# Task 1
#

#
# Create a list a with values 5, 'a', 7. Then create a variable b that
# points to the same object like the second element in list a.
#

# Your code here #
a = [5, 'a', 7]  # Creating the list
b = a[1]  # Get b to point to the same object as the second element in a


#
# Task 2
#

#
# Append the integer 5 to the end of list a and store the result in
# variable b. Afterwards change the first element in b to integer 0.
#
a = [1, 2, 3, 4]

# Your code here #
b = a + [5]
b[0] = 0


#
# Task 3
#

# Append the integer 5 to the end of tuple a and store the result in
# variable b. Afterwards try to change the first element in b to integer 0.
# Why  will it not work to overwrite the element in the tuple? And what can
# you do to create a new tuple like b where the first element is 0?
a = (1, 2, 3, 4)

# Your code here #
b = a + (5,)
# b[0] = 0  # this will fail because tuple are immutable
# However, we can always stitch together a new tuple from other tuples:
b = (0,) + b[1:]


#
# Task 4
#

# Add an entry with key 'c' and value 3 to the dictionary a.
a = dict(a=1, b=2)  # or a = {'a':1, 'b':2}

# Your code here #
a['c'] = 3


#
# Task 5
#

# Create a list with numbers from 0 to 100, excluding 100 (use range()). Then
# extract every third element starting at index 50 until index 70 and store it
# in a new list. You can solve this using slicing.

# Your code here #
l = list(range(100))
l2 = l[50:70:3]  # start=50, stop=70, stepsize=3


#
# Task 6
#

# String 'a' contains elements that are separated by either a ',' or a ';'
# character. Reverse the order of elements and replace the ',' or ';'
# characters by ':'.
a = 'element1,element2;element3;element4;element5,element6'

# Your code here #

# One way of solving this is to split string 'a' at characters ',' and ';',
# which will give us a list with the elements we are interested in. Then we
# can reverse the list and join the elements with a ':' character to a string.
# To split the string we need to find a common split character. We can replace
# ';' by ',', then all elements are separated by ','.
common_a = a.replace(';', ',')  # Now all elements are separated by ','
split_a = common_a.split(',')  # Get a list of all ','-separated elements
reversed_a = split_a[::-1]  # Now we reverse the order of elements in the list
a = ':'.join(reversed_a)  # Join the list elements by ':' to new string

