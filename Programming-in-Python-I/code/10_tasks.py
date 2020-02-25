# -*- coding: utf-8 -*-
"""10_tasks.py

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
compare your solution to the solution given in the file 10_solutions.py.

"""

###############################################################################
# 10 classes
###############################################################################

#
# Task 1
#

# Create a class 'Face' with an attribute 'orientation' and two methods
# 'look(new_orientation)' and 'show()'.
# look() shall take a string new_orientation as argument. If new_orientation is
# the string "left", let the attribute 'orientation' point to string "left". If
# new_orientation is the string "right", let the attribute 'orientation' point
# to string "right".
# 'show()' should print a face in 'right' or 'left' orientation, based
# on the current value of 'orientation'.

# Example usage:
# face = Face()
# face.look('left')
# face.show()  # prints output '<.<'
# face.look('right')
# face.show()  # prints output '>.>'


# Your code here #


#
# Task 2
#

# Create a class 'OwlFace' that is derived from class Face from task 1.
# look() shall take a string new_orientation as argument. If new_orientation is
# the string "left", let the attribute 'orientation' point to string "left". If
# new_orientation is the string "right", let the attribute 'orientation' point
# to string "right". If new_orientation is the string "ahead", let the
# attribute 'orientation' point  to string "ahead".
# 'show()' should print a face in 'right', 'left', or ahead orientation, based
# on the current value of 'orientation'.

# Example usage:
# face = Face()
# face.look('left')
# face.show()  # prints output '(O<O)'
# face.look('right')
# face.show()  # prints output '(O>O)'
# face.look('ahead')
# face.show()  # prints output '(OvO)'


# Your code here #
