# -*- coding: utf-8 -*-
"""02_conditions_loops.py

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

In this file we will learn about if, elif, and else conditions, for loops,
while loops, and list comprehensions. They will give us control over which
lines of code are executed and for how many repetitions certain lines of code
are executed.

"""

###############################################################################
# Code blocks and indentation
###############################################################################
# Python uses indentation to form blocks of code and statements, in contrast to
# e.g. {} in other languages. Indentation can be done via tabs or spaces. It is
# common to use 4 spaces (in most editors this is the default or you can set
# tab to equal 4 spaces).
# Example:
# block 1
# block 1
    # block 2
    # block 2
    # block 2
        # block 3
        # block 3
    # block 2
        # block 4
        # block 4
    # block 2
# block 1
# block 1
    # block 5
    # block 5
# block 1
# block 1


###############################################################################
# if, elif, else (1)
###############################################################################
# Code within an "if" statement is executed only if the condition is True.
condition = True
if condition:
    print("This code block is executed since 'condition' is 'True'!")

condition = False
if condition:
    print("This code block is not executed since 'condition' is 'False'!")
    
if not condition:  # Here we use "not False" as condition
    print("'not False' evaluates to 'True', so this code block is executed!")

# We can combine the "if" statement with an "else" statement. This will
# execute the "if" part only if the condition is True and otherwise execute
# the "else" part.
is_true = True
if is_true:
    print("This code block is executed!")
else:
    print("This code block is not executed since the 'if'-part was already "
          "executed!")

if not is_true:
    print("This code block is not executed!")
else:
    print("This code block is executed since the 'if'-part was not executed!")

# We can also combine the "if" statement with "elif" statements. Doing this,
# the conditions in the if-elif statements are checked one after the other.
# Only the part where the first condition is True is executed and the rest
# is ignored.
is_true = True
if not is_true:
    print("This code block is not executed!")
elif not is_true:
    print("This code block is not executed!")
elif is_true:
    print("Only this code block is executed!")
elif is_true:
    print("This code block is not executed since a previous if-part was "
          "already executed!")
elif not is_true:
    print("This code block is not executed since a previous if-part was "
          "already executed!")

# We can also combine the "if", "elif", and "else" statements. The conditions
# in the if-elif-else statements are checked one after the other. Only the
# part where the first condition is True is executed and the rest is ignored
# but if none of the "if"-"elif" parts are executed, the "else" part will be
# executed.
is_true = True
if not is_true:
    print("This code block is not executed!")
elif not is_true:
    print("This code block is not executed!")
elif not is_true:
    print("This code block is not executed!")
else:
    print("This code block is executed since no other part was executed!")


###############################################################################
# Comparisons
###############################################################################
# We already saw that we need boolean values as conditions to decide which
# parts of code to execute in the if-elif-else statements. Comparisons in
# Python allow us to compare two objects and return a boolean value. You can
# use comparisons on objects of different datatypes.

# Let us assume we have two variables a and b that refer to float objects:
a = 1.
b = 2.

# There are 8 comparison operations in Python:
equal = a == b  # Check if a and b have equal values
larger = a > b  # Check if a larger than b
larger_equal = a >= b  # Check if a larger than or equal to b
lesser = a < b  # Check if a lesser than b
lesser_equal = a <= b  # Check if a lesser than or equal to b
same_object = a is b  # Check if a and b refer to same object
not_same_object = a is not b  # Check if a and b do not refer to same object
# Important: '==' will compare the numerical value of objects, 'is' will
# compare the objects themselves!

# Numerical values of 1 and 1. are equal even if they have different datatypes
is_equal = 1 == 1.
# ... but int 1 and float 1. are not the same object
is_not_same_object = 1 is 1.

# Boolean True defaults to numerical 1 and False to numerical 0. Any numerical
# value other than 0 defaults to True if used as boolean:
is_equal = 1 == True  # Is True since numerical values are equal
is_equal = 1 is True  # Is False since objects are not the same
is_equal = False > 2.3  # Is False since 0 is not greater than 2.3

# You can chain comparisons using boolean operations like "and" or "or".
# "and" will be True if and only if both of the two statements that it combines
# evaluate to True:
is_true = True and True
is_false = True and False
is_false = False and False

# "or" will be True if and only if one or both of the two statements that it
# combines evaluate to True:
is_true = True or True
is_true = True or False
is_false = False or False

# The priority of comparisons is higher than that of boolean operations but lower
# than mathematical operators:
is_true = 1 == True and 1 == 1.
is_true = (1 == True) and (1 == 1.)  #  Is the same as "1 == True and 1 == 1."
is_true = (1 == True) and True  #  Is the same as "1 == True and True"
is_true = (1 > -4) and (1 is 1)
is_true = (1 > -4) and (1 is 1) and (1 < 4)
is_false = (1 > -4) and not (1 < 4)

# You can also chain comparisons without "and" or "or":
is_true = 1 < 4 <= 5  # Is the same as "1 < 4 and 4 <= 5"

# All non-numerical objects support the "==" and "!=" operator, which is then
# equivalent to "is" and "is not".
is_false = () == []  # Is equivalent to "() is []"
is_true = dict() != []  # Is equivalent to "dict() is not []"

# Other comparison operators are only supported by non-numerical objects
# "if they make sense". E.g. you can use them on strings:
is_true = 'abc' == 'abc'
is_true = 'abc' >= ''
is_true = 'abc' >= 'a'
is_true = 'abc' >= 'ab'
is_true = 'abc' >= 'abc'
is_false = 'abc' >= 'abd'
is_false = 'abc' >= 'abd'

# Here we combine comparisons and if-elif-else statements:
a = 2
b = 3
if a > b:
    print('a > b !')
elif a < b:
    print('a < b !')
elif a == b:
    print('a == b !')
else:
    print('How did I get here?')


###############################################################################
# for loops
###############################################################################
# For loops in python iterate over a so-called iterable. This can be e.g. a
# list, dictionary, tuple, string, or function return (see yield statement
# later).

some_iterable = [1, 2, 'three', 4.0, None, 0]
for current_element in some_iterable:
    # This code block will be execute for each element in "some_iterable",
    # where "current_element" takes the value of the current element in the
    # iterable "some_iterable".
    print(current_element)

a = 0
print(f"Before loop: a is {a}")
for i in [1, 2, 3, 10]:
    a += i
    print(f"Current value for i is {i}")
    print(f"Current value for a is {a}")
print(f"After loop: a is {a}")

# If you want to iterate over iterable elements and get the number of the
# current loop iteration, you can use the enumerate() function.
for iteration, current_element in enumerate(range(5, 10)):
    print("loop iteration: {} element: {}".format(iteration, current_element))

# If you want to iterate over two iterables simultaneously, you can either
# iterate over the indices:
for s in range(len("test")):
    print('test'[s] == "nest"[s])

# Or you can use zip() to combine the iterables:
for s1, s2 in zip('test', '123456'):
    print(s1 == s2)

# Important: Changing the content of the iterable while you iterate over it
# is tricky and might lead to unintended results. Use a copy of the iterable
# instead, if want to modify it during the loop.

# Tricky and potentially unsafe:
# some_iterable = list(range(10))
# for element in some_iterable:
#     print(element)
#     del some_iterable[0]


###############################################################################
# while loops
###############################################################################
# While loops continue as long as the condition is True:
i = 0
while i < 10:
    print(i)
    i += 1

# This would not be executed:
i = 0
while i < 0:
    print(i)
    i += 1

# This would run forever:
# i = 0.
# while True:
#     print(i)
#     i += 1.


###############################################################################
# break, continue, and else
###############################################################################
# The break and continue keywords can be used to either escape the loop or to
# jump to the next element
for i in range(10):
    if i == 4:
        continue
    elif i == 8:
        break
    print(i)

# The else keyword put after a loop has a special meaning in Python. It will
# only get executed if the loop has finished properly, i.e. the for loop
# iterable was exhausted or the while condition was False.
for i in range(5):
    if i == 3:
        break
    print(i)
else:
    print("Made it to the end!")


###############################################################################
# list comprehensions
###############################################################################
# List comprehensions are a fast(er) and compact way to write loops and
# conditions applied to an iterable.
# List comprehensions with () will be evaluated dynamically, list
# comprehensions with [] will immediately generate a full list.
# The syntax is: [code-block-in-loop for current_element in iterable]
old_list = [1, 2, 3, 4, 5]
new_dynamic_iterable = (str(i) for i in old_list)
new_list = [str(i) for i in old_list]
# This returns the same results as:
new_list = []
for i in old_list:
    new_list.append(str(i))
    

# You can use an if statement at the end of a list comprehension to execute
# the code-block in the loop conditionally. Note that there is no element
# appended to the list if the condition is not True:
new_list = [str(i) for i in old_list if i < 3]
# This returns the same results as:
new_list = []
for i in old_list:
    if i < 3:
        new_list.append(str(i))

# You can also use if-else statements in the loop code-block:
new_list = [str(i) if (i < 5) else 'end' for i in old_list]
# This returns the same results as:
new_list = []
for i in old_list:
    if (i < 5):
        new_list.append(str(i))
    else:
        new_list.append('end')

# Something fancier:
new_list = [str(i) if (i > 1) and (i < 5) else 'start' if (i == 1) else 'end'
            for i in old_list]
# This returns the same results as:
new_list = []
for i in old_list:
    if (i > 1) and (i < 5):
        new_list.append(str(i))
    else:
        if (i == 1):
            new_list.append('start')
        else:
            new_list.append('end')

# You can iterate over indices and use multiple iterables in one list
# comprehension:
old_list = [1, 2, 3, 4, 5]
old_list2 = list(range(5))

new_list = [old_list[i] + old_list2[i] for i in range(len(old_list))]

# You may also use zip() to combine 2 lists
new_list = [a + b for a, b in zip(old_list, old_list2)]

# You can even nest list comprehensions:
old_list = [1, 2, 3, 4, 5]
new_list = [str(element)
            for current_tuple in zip(old_list, old_list[::-1])
            for element in current_tuple]
# This returns the same results as:
new_list = []
for current_tuple in zip(old_list, old_list[::-1]):
    for element in current_tuple:
        new_list.append(str(element))
