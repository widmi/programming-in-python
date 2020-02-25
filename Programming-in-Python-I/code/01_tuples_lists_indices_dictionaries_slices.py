# -*- coding: utf-8 -*-
"""01_tuples_lists_indices_dictionaries_slices.py

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

In this file we will learn about tuples, lists, how to index them, how
to create dictionaries, and how to use slices to retrieve multiple elements.

"""

###############################################################################
# Lists
###############################################################################

# list: object that can store multiple objects in a sorted manner. You can
# create a list with square brackets [].
a_list = [1, 2.3456, 3, 'abc']

# Each list element is like a variable. You can access it via an integer index
# via "listname[index]" (note that indices start at 0):
first_element = a_list[0]  # Here we get the 1st element from list a_list

# Only integers (and slices, which we will learn about later) are allowed as
# list indices:
#not_possible = a_list[0.0]  # This will not work


# Lists can store all kinds of objects, even other lists. If a list contains
# another list as element, it is referred to as a "nested" list.
# Here we create a list, i.a. containing a list, i.a. containing a list:
nested_list = ['eg', 3, [23, 5, 65, ['aeg', 35]]]
# This will return the element at index 2, which is another list inner_list:
inner_list = nested_list[2]
# We can index this list inner_list again to get an element:
inner_list_element_zero = inner_list[0]
# We can navigate through the nested lists using indices like this:
nested_list_element = nested_list[2][3][0]

# We can modify the content of the list by assigning an object to the
# indexed list:
a_list = [1, 2.3456, 3, 'abc']
a_list[0] = 5  # Change the first element to 5
a_list[0] = a_list[0] * 2  # Change first element to first element times 2
# We can also use the abbreviated notations for "in-place" operations:
a_list[0] *= 3  # Change first element to first element times 3

# You may also use negative indices to start from the end of the list
# (using -1 for the last element, -2 for the 2nd last, etc.)
last_element = a_list[-1]
second_last_element = a_list[-2]

# List elements can be deleted by index using del
old_first_element = a_list[0]
del a_list[0]  # removes first element from list
new_first_element = a_list[0]

# You can append and remove elements from a list and you can use many
# operations you already know from strings:
a_list.append('new_element')  # Append object to the end of list
a_list.remove('abc')  # Remove first reference of object 'abc' from list
a_list *= 3  # Repeat list 3 times
detect_2 = 'abc' in a_list  # Check if object 'abc' is referenced to in list

# Sorting can be done in-place (will not create a new list)
numeric_list = [1, 5, 3, -5]
numeric_list.sort()

# Or via the sorted() function to get a sorted copy of the list
numeric_list = [1, 5, 3, -5]
sorted_list = sorted(numeric_list)

# Iterable objects, e.g. strings, can be converted to lists using list():
string_as_list = list("somestring")

# Information on sorting: https://wiki.python.org/moin/HowTo/Sorting

# Important:
# Unlike strings, lists are mutable objects. Modifying a string means creating
# and allocating a completely new string. In contrast, modifications to a list
# will only change the modified elements. Changing, appending, or removing
# elements will not create an entirely new list but reuse the old one.
long_string = "abc" * 500
long_string = long_string.replace('a', '')  # This will create a new string
long_list = ['a', 'b', 'b'] * 500
long_list.remove('a')  # This will not create a new list

# We can use indexing also for strings:
a_character = 'abc'[0]
# But since strings are immutable, overwriting elements in-place is not
# possible:
#long_string[0] = 5


###############################################################################
# Tuples
###############################################################################

# tuple: an immutable sorted array/list. Its content can not be changed after
# creation. You can create them via optional brackets (). They are similar to
#strings.
a_tuple = 1, 2, 3  # you can do this but I wouldn't recommend it
a_tuple = (1, 2, 3)  # this is more readable and better

# Available operations are similar to strings and lists
joined_tuples = a_tuple + a_tuple  # Concatenating 2 tuples
multiple_tuples = a_tuple * 5  # Repeating a tuple 5 times

# As with strings and lists, you can use an index to extract an element
first_tuple_element = a_tuple[0]

# As with strings, this is not possible:
#a_tuple[0] = 2

# Tuples can be converted to lists and vice versa:
now_a_tuple = tuple(a_list)
now_a_list = list(a_tuple)


###############################################################################
# Dictionaries
###############################################################################

# dict: a mutable unsorted list, accessed via key words. You  can create them
# via dict(key=value, key=value, ...) or via {key: value, key: value}. Note
# that every hashable object can be used as key. However, the dict() statement
# will convert the keys to strings, the {} statement uses the key-objects
# directly.
some_key = 'abc'
dictionary = dict(some_key=3.24, other_key='twh')
dictionary2 = {'stringkey': 55, some_key: 'someitem', 23: 4}

# Indexing is done via the key objects
element = dictionary['some_key']
element2 = dictionary2[some_key] * dictionary2[23]

# Get all keys as list (order of keys might change since unsorted!)
keys = list(dictionary.keys())

# Get all values as list (order of values might change since unsorted!)
values = list(dictionary.values())

# Get key-value pairs as tuples
pair_tuples = list(dictionary.items())

# Check for key
is_key_there = 'some_key' in dictionary

# Get a value and use a default value if key does not exist
value = dictionary.get('nonexistingkey', 'defaultvalue')

# Create dictionary from list with keys and list with values:
keys = ['a', 'b', 'c']
values = [1, '2', '3']
zip_dictionary = dict(zip(keys, values))

# Hashable objects, that also includes tuples and functions, can be used as
# keys:
other_dict = dict()  # Empty dictionary
other_dict[(1, 2, 3)] = 'test'  # Using a tuple (1, 2, 3) as key
other_dict_element = other_dict[(1, 2, 3)]

# For dictionaries with ordered entries see OrderedDict
# (from collections import OrderedDict)


###############################################################################
# Slicing
###############################################################################

# slice: index object that allows to select multiple entries based on a simple
# pattern. Can be created via slice(...) or directly in place of the indices as
# somelist[start:end:stepsize]. Note that the element at index "end" is not
# included. Omitting start, end, or stepsize defaults to the first, and last
# sequence element and a stepsize of 1, respectively. You may omit the last
# double dots if you do not want to specify a stepsize.

some_list = [0, 1, 2, 3, 4]
some_elements = some_list[2:3]  # this is the same as some_list[2:3:]
first_2_elements = some_list[:2]
last_2_elements = some_list[-2:]
reversed_list = some_list[::-1]
all_elements = some_list[:]

string_slice = "blablabla"[::-1]

# Slice objects can be created explicitly (and reused later)
some_slice = slice(2, 5, 2)  # Notation: slice(start, stop, stepsize)
some_other_elements = some_list[some_slice]

# You can use slices to replace the sliced elements by new elements
original_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
original_list[3:7] = [10, 10]  # Replace sub-list [3:6] by [10, 10]


###############################################################################
# Useful functions
###############################################################################
# range(): generates a sequence of integer numbers as generator object,
# following similar syntax like slicing. range() creates a dynamic iterable
# but we can convert it to a list via list(range(...))
zero_to_four = list(range(5))
two_to_six = list(range(2, 7))

# len(): get the number of elements (=length) of an iterable
zero_to_four_length = len(zero_to_four)
stringlength = len('123')


# More information on lists, slicing, and range():
# https://docs.python.org/3/library/stdtypes.html#sequence-types-list-tuple-range
