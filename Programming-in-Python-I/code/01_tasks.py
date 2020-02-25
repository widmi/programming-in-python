# -*- coding: utf-8 -*-
"""01_tasks.py

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

Tasks for self-study. Try to solve these tasks on your own and
compare your solution to the solution given in the file 01_solutions.py.

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


#
# Task 2
#

#
# Append the integer 5 to the end of list a and store the result in
# variable b. Afterwards change the first element in b to integer 0.
#
a = [1, 2, 3, 4]

# Your code here #


#
# Task 3
#

# Append the integer 5 to the end of tuple a and store the result in
# variable b. Afterwards try to change the first element in b to integer 0.
# Why  will it not work to overwrite the element in the tuple? And what can
# you do to create a new tuple like b where the first element is 0?
a = (1, 2, 3, 4)

# Your code here #


#
# Task 4
#

# Add an entry with key 'c' and value 3 to the dictionary a.
a = dict(a=1, b=2)  # or a = {'a':1, 'b':2}

# Your code here #


#
# Task 5
#

# Create a list with numbers from 0 to 100, excluding 100 (use range()). Then
# extract every third element starting at index 50 until index 70 and store it
# in a new list. You can solve this using slicing.

# Your code here #


#
# Task 6
#

# String 'a' contains elements that are separated by either a ',' or a ';'
# character. Reverse the order of elements and replace the ',' or ';'
# characters by ':'.
a = 'element1,element2;element3;element4;element5,element6'

# Your code here #
