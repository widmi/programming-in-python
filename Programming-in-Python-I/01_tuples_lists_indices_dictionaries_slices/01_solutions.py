# -*- coding: utf-8 -*-
"""01_solutions.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.10.2019

Add tasks 7, 8, 9 by Van Quoc Phuong Huynh -- WS2020

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


#
# Task 7
#

# Add two elements 7 and 9 to a nested list 'a' so that it becomes
# [1, 2, 3, [4, 5, [6, 7, 8, 9], 10, 11], 12, 13, 14]
# Afterwards, flatten it to become a one-dimensional list
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
a = [1, 2, 3, [4, 5, [6, 8], 10, 11], 12, 13, 14]

# Your code here #
a[3][2].append(9)
a[3][2].insert(1, 7)
print(a)

sub_list = a[3]  # get sub list at index 3
sub_sub_list = sub_list[2]  # get sub list at index 2 of sub_list
sub_list[2:3] = sub_sub_list  # replace element at index 2 of sub_list by elements of sub_sub_list
a[3:4] = sub_list  # replace element at index 3 of list a by elements of sub_list
print(a)

# or alternatively
a = [1, 2, 3, [4, 5, [6, 8], 10, 11], 12, 13, 14]
a[3][2].append(9)
a[3][2].insert(1, 7)
sub_list = a[3]  # get sub list at index 3
sub_sub_list = sub_list[2]  # get sub list at index 2 of sub_list
a = a[:3] + sub_list[:2] + sub_sub_list + sub_list[3:] + a[4:]
print(a)


#
# Task 8
#

# Create a dictionary d3 that combines the two given dictionaries d1 and d2.
# Then replace the entry {6: "five"} by {5: "five"}.
# Desired result: {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
d1 = {1: "one", 2: "two", 3: "three"}
d2 = {4: "four", 6: "five"}

# Your code here #
d3 = d1.copy()  # d3 is a copy of d1
d3.update(d2)  # all items of d2 are added to d3
d3[5] = d3.pop(6)  # remove item of key 6 from d3, the value "five" is returned and used as the value of new key 5
# # Alternatively:
# d3[5] = d3[6]
# del d3[6]
print(d3)


#
# Task 9
#

# You are given a group of employees as nested dictionary s.
# Given this dictionary d, get the supervisor's name of the employee with id "e2".
d = {
    "sale department": {
        "e1": {
            "name": "Alice",
            "role": "Sale Manager",
            "salary": 5000
        },
        "e2": {
            "name": "Bob",
            "role": "Sale Employee",
            "salary": 4000,
            "supervisor": "e1"
        }
    }
}

# Your code here #
sale_dept = d["sale department"]  # get sale department info
sup_id = sale_dept["e2"]["supervisor"]  # get supervisor id of employee "e2"
print(sale_dept[sup_id]["name"])  # get name of the supervisor
