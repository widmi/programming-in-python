# -*- coding: utf-8 -*-
"""03_tasks.py

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
compare your solution to the solution given in the file 03_solutions.py.

"""

###############################################################################
# 03 Functions and Printing
###############################################################################

#
# Task 1
#

# Write a function that divides two numbers and returns the result. The
# numbers should be taken as keyword arguments with defaults of 1.0 for both
# values.

# Your code here #


#
# Task 2
#

# Write a function that takes one argument that can be assumed to be a list.
# The function should add up the last two elements in the list, print the sum,
# and append it at the end of the list without any return value.

a = [1, 2, 3, 5]

# Your code here #


#
# Task 3
#

# Write a function my_function() that takes an arbitrary number of positional
# arguments as input. You can assume these arguments are either strings or
# nested lists of strings.
# Create a flat (=not nested) list from these arguments that holds only the
# strings. Add 'image' to the beginning of each strings if the string ends
# with '.png' ('file1.png' -> 'imagefile1.png'), 'text' if the string ends
# with '.txt', and 'data' otherwise. You can use a recursive
# function to solve this task.
# Useful functions:
# 'string'.endswith('substring') to check if a string ends with a certain
# substring.
# isinstance(a, list) to check if variable "a" is of type list

# Example function call:
some_list = ['file2.txt', 'file3.txt']
some_nested_list = ['file4.png', 'file5.txt', ['file6.txt', 'file7.dat']]
# my_list = my_function('file0.txt', 'file1.png', some_list, some_nested_list)

# Your code here #


#
# Task 4
#

# Write a function next_element(my_iterable) that returns an iterable that
# yields the next element in my_iterable.
# If the list is exhausted (i.e. if the last element has been reached), the
# next element should be the first list element.
# Write a loop that calls next_element() 25 times and prints the currently
# returned value of next_element().
# Example function call:
# next_element('abcdefg')

# Your code here #


#
# Task 5
#

# Write a function next_element(my_iterable, max_iter) that returns an iterable
# that yields the next element in my_iterable.
# If the list is exhausted (i.e. if the last element has been reached), the
# next element should be the first list element. A maximum of max_iter elements
# shall be returned.
# Write a loop that calls next_element(my_iterable, max_iter) until max_iter is
# reached and prints the currently returned value of next_element().
# Example function call:
# next_element('abcdefg', max_iter=20)

# Your code here #
