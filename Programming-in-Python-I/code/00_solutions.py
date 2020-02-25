# -*- coding: utf-8 -*-
"""00_solutions.py

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

Example solutions for tasks in file 00_tasks.py.

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
a = 1
b = False


#
# Task 2
#

# Convert the following variable 'a' into an integer. Take a look at the
# variable in the debugger. What will happen to the floating point number if
# it is converted to an integer?
a = 5.356

# Your code here #
a = int(a)
# Conversion from float to int means we lose the information after the
# floating point!


#
# Task 3
#

# Create a variable 'a' with content "test" and a variable 'b' that points
# to the same content as 'a'. Afterwards let 'a' point to 2 and delete 'b'.
# What will happen to the string "test" after deleting variable 'b'?

# Your code here #
a = "test"
b = a
a = 2
del b
# The garbage collector will at some point remove the string "test"
# because no variable is pointing to it anymore.


#
# Task 4
#

# Create a variable 'a' with content 123 and create a variable 'b'
# pointing to a string that consists of the sentence
# "this is a test " followed by the contents of variable 'a'.
# Finally, create a variable 'c' that repeats this sentence 3 times.

# Your code here #
a = 123
b = f"this is a test {a}"
c = b * 3


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

# For variable 'a', an integer is used, which is precise. Since Python3
# uses variable-length integers, we will not run out of bits to encode the
# value.
# For variable 'b' a float is used and then converted to an integer. However,
# the float 123456789123456789.0 can not be stored in the 64bit Python3
# float, so we lose precision and effectively end up with a different value.
# If you want the precise value, you should use the integer version.
