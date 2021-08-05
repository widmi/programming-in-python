# -*- coding: utf-8 -*-
"""02_tasks.py

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

Tasks for self-study. Try to solve these tasks on your own and
compare your solution to the solution given in the file 02_solutions.py.

"""

###############################################################################
# 02 Conditions and loops
###############################################################################

#
# Task 1
#

# Loop through list fnames and count how many elements in the list end with
# '.png'. You can use 'string'.endswith('substring') to check if a string ends
# with a certain substring. Use a for loop to solve this task.
fnames = ['file0.txt', 'file1.png', 'file2.txt', 'file3.txt',
          'file4.png', 'file5.txt', 'file6.txt']

# Your code here #


#
# Task 2
#

# Create a list "some_list" with values from 3 to 100 (100 is excluded). Loop
# through the list some_list and compute the sum of the square root of all
# elements.

# Your code here #


#
# Task 3
#

# Create a counter "counter" and loop through list "some_list". For each
# element 'not good' add -1, for each element 'okay' add 0, for each element
# 'good' add 1, and for each element 'very good' in the list add 2 to the
# counter.
some_list = ['okay', 'not good', 'okay', 'okay', 'good', 'good', 'okay',
             'very good',  'not good', 'very good']

# Your code here #


#
# Task 4
#

# Create a new list from list fnames but add 'image' to the beginning of the
# element if the string ends with '.png' ('file1.png' -> 'imagefile1.png'),
# 'text' if the string ends with '.txt', and 'data' otherwise. You can use
# 'string'.endswith('substring') to check if a string ends with a certain
# substring. Use a for-loop for the task.
fnames = ['file0.txt', 'file1.png', 'file2.txt', 'file3.txt',
          'file4.png', 'file5.txt', 'file6.txt', 'file7.dat']

# Your code here #


#
# Task 5
#

# Create a new list from list fnames but add 'image' to the beginning of the
# element if the string ends with '.png' ('file1.png' -> 'imagefile1.png'),
# 'text' if the string ends with '.txt', and 'data' otherwise. You can use
# 'string'.endswith('substring') to check if a string ends with a certain
# substring. Use a list-comprehension for the task.
fnames = ['file0.txt', 'file1.png', 'file2.txt', 'file3.txt',
          'file4.png', 'file5.txt', 'file6.txt', 'file7.dat']

# Your code here #


#
# Task 6
#

# Create an empty string 'some_string'. Use the line
user_input = input('Input:\n')
# to get a string as input from the user. Use a while loop to get input
# from the user and append the input to the end of 'some_string'. Escape the
# loop if the user types "end".

# Your code here #


#
# Task 7
#

# Merge two sorted lists, list1 and list2, to become a new sorted list
list1 = [1, 4, 7, 8, 12, 30]
list2 = [3, 5, 8, 9, 18, 20, 21, 25]

# Your code here #


#
# Task 8
#

# Use while ... else ... statement to implement a pseudo-login scenario in
# which a user is asked to enter a password. Use
# password = input("Password: ")
# to get the input of the user as string.
# If the password is correct, print "Login success" and and exit the program.
# Otherwise, let the user try again.
# In case of entering 3 wrong passwords, print
# "Contact the administrator to recover password"
# and exit the program.
# The choice of the correct password is up to you.

# Your code here #


#
# Task 9
#

# Given lists of grades of students, make a summarized grade list of students
# following the scheme: <student name>, <grade sum>, <tuple of component grades>
exam1s = [6, 10, 9, 8]
exam2s = [10, 7, 8, 8]
assg1s = [20, 15, 18, 17]
assg2s = [25, 20, 30, 28]
assg3s = [26, 25, 30, 25]
students = ["Alice", "Malice", "Sam", "May"]

# Expected result:
# [['Alice', 87, (6, 10, 20, 25, 26)],
#  ['Malice', 77, (10, 7, 15, 20, 25)],
#  ['Sam', 95, (9, 8, 18, 30, 30)],
#  ['May', 86, (8, 8, 17, 28, 25)]]

# Your code here #
