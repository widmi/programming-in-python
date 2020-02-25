# -*- coding: utf-8 -*-
"""00_tasks.py

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
compare your solution to the solution given in the file 00_solutions.py.

"""

###############################################################################
# 00 Variables
###############################################################################

#
# Task 1
#

# Create a variable 'a' with the content 1 and a variable 'b' with the
# content False.

# Your code here #


#
# Task 2
#

# Convert the following variable 'a' into an integer. Take a look at the
# variable in the debugger. What will happen to the floating point number if
# it is converted to an integer?
a = 5.356

# Your code here #


#
# Task 3
#

# Create a variable 'a' with content "test" and a variable 'b' that points
# to the same content as 'a'. Afterwards let 'a' point to 2 and delete 'b'.
# What will happen to the string "test" after deleting variable 'b'?

# Your code here #


#
# Task 4
#

# Create a variable 'a' with content 123 and create a variable 'b'
# pointing to a string that consists of the sentence
# "this is a test " followed by the contents of variable 'a'.
# Finally, create a variable 'c' that repeats this sentence 3 times.

# Your code here #


#
# Task 5
#

# Take a look at the following lines of code. Why does variable 'a' have a
# different value than variable 'b'?
# And which of the two version should be used if you need the precise value?

a = 123456789123456789
b = int(123456789123456789.0)
print(f"value a: {a}")
print(f"value b: {b}")

# No code required for the solution of this task #
