# -*- coding: utf-8 -*-
"""06_tasks.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.10.2019

Add tasks 4 by Van Quoc Phuong Huynh -- WS2020

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
compare your solution to the solution given in the file 06_solutions.py.

"""

###############################################################################
# 06 Command line arguments and subprocess
###############################################################################

#
# Task 1
#

# Create a Python script that expects an integer as command line argument. It
# should then print this integer.


# Your code here #


#
# Task 2
#

# Save the script from task 1 into a python file with name "myprog.py".
# Call this program using subprocess module with command line arguments
# from 0 to 10 (including 10). Collect the output of myprog.py and print
# it to the console.


# Your code here #


#
# Task 3
#

# Take the program "echo", which will just print what you pass to it, and
# apply it to values from 0 to 500. Use 8 parallel worker threads for the
# task via the Pool.map function from the multiprocessing module. Look at
# the outputs - since it is asynchronous the values might be printed in
# different orders (but the returned list of results would be in the same
# order as the list of arguments).

# Your code here #


#
# Task 4
#

# You are given 2D data as nested list 'data'
data = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [2, *range(5000)],
        [3, *range(6000)],
        [4, *range(7000)]]
# , where [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] is the first row and [1, 2, 3, 4] is
# the first column.
# Your task is to compute the sum of each row in parallel, by using
# multiprocessing.Pool to spawn a set of worker processes. Afterwards your code
# should print the result. For the example above, the output should be:
# [55, 65, 75, 85]
#
# Hint: You will need a function that computes the sum of 1 row, i.e. the sum
# over 1 list. You can either write your own function or use a certain built-in
# function.
from multiprocessing import Pool

# Your code here #
