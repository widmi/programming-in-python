# -*- coding: utf-8 -*-
"""02_solutions.py

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

Example solutions for tasks in file 02_tasks.py.

"""

###############################################################################
# 02 Conditions, for-loops, while-loops, and list comprehensions
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
counter = 0
for fname in fnames:
    if fname.endswith('.png'):
        counter += 1
        
# Alternative solution, since True evaluates to the integer 1:
counter = 0
for fname in fnames:
    counter += fname.endswith('.png')


#
# Task 2
#

# Create a list "some_list" with values from 3 to 100 (100 is excluded). Loop
# through the list some_list and compute the sum of the square root of all
# elements.

# Your code here #
result = 0
for value in range(3, 100):
    result += value ** (1/2)


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
counter = 0
for element in some_list:
    if element == 'not good':
        counter += -1
    elif element == 'okay':
        counter += 0
    elif element == 'good':
        counter += 1
    elif element == 'very good':
        counter += 2

# Alternative (ignoring 'okay' since we should add 0):
counter = 0
for element in some_list:
    if element == 'not good':
        counter += -1
    elif element == 'good':
        counter += 1
    elif element == 'very good':
        counter += 2


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
new_list = []
for fname in fnames:
    if fname.endswith('.png'):
        new_list.append('image' + fname)
    elif fname.endswith('.txt'):
        new_list.append('text' + fname)
    else:
        new_list.append('data' + fname)


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
new_list = ['image' + fname if fname.endswith('.png') else
            'text' + fname if fname.endswith('.txt') else
            'data' + fname
            for fname in fnames]


#
# Task 6
#

# Create an empty string 'some_string'. Use the line
user_input = input('Input:\n')
# to get a string as input from the user. Use a while loop to get input
# from the user and append the input to the end of 'some_string'. Escape the
# loop if the user types "end".

# Your code here #
some_string = ''
while True:
    user_input = input('Input:\n')
    some_string += user_input
    if user_input == 'end':
        break

# Better solution (here we first collect the strings in a list and then
# combine them instead of creating a new longer string every iteration):
temp_list = []
while True:
    user_input = input('Input:\n')
    temp_list.append(user_input)
    if user_input == 'end':
        break
some_string = ''.join(temp_list)
