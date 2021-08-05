# -*- coding: utf-8 -*-
"""03_code.py

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

In this file we will learn about functions, passing arguments to and
returning values from functions, printing to and reading from the
console, and how to import python modules.

"""

###############################################################################
# Functions
###############################################################################
# Functions can be used to create re-usable parts of code. Python functions
# should have lower case names, optionally take arguments, and are created
# with the "def" keyword and the syntax
# def function_name(arguments):
#     code in function
#     return returned-variables

# This would be a function that takes no arguments, computes 3*5, converts the
# result to a string and returns it:
def no_arguments():
    a = 3 * 5
    a = str(a)
    return a


# We can then call this function whenever we need it:
result = no_arguments()
another_result = no_arguments()


# Note that variables created inside a function only exist inside the function!
def add(a, b):
    # This code is executed when calling the function. Variables created here
    # are not available outside of the function.
    c = a + b  # This variable "c" exists only within this function
    return c

# Arguments can be passed to functions using keyword-arguments:
result = add(b=2, a=1)

# Or using positional arguments (following the position of the arguments in
# the function definition):
result = add(1, 2)

# Or a mix of both - as long as the positional arguments are in front:
result = add(1, b=2)

# If you do not want to make use of the returned values, use an underscore:
_ = add(1, 2)

# The variable "c" in the function does only exist within the function, so
# we can use another different variable with the same name outside the
# function:
c = 15
print(f"c before function: {c}")
# This will not alter the variable c outside the function:
result = add(b=2, a=1)
print(f"c after function: {c}")


#
# Function arguments
#

# You can assign default values to the arguments and make them optional:
def add(a, b=1):  # we set a default value of 1 for argument b
    c = a + b
    return c


result = add(1)


# You may use colons to indicate the type of the arguments (this is only
# for indication and does not force anything!). You can indicate the return
# type using "->".
# See the module "typing" for more type indications:
# https://docs.python.org/3.7/library/typing.html
def add(a: int, b: int = 1) -> int:
    c = a + b
    return c


# This still works but PyCharm warns us about an integer being expected:
result = add(1, 2.123)


# Make use of docstrings to show what your function is about:
def add(a: int, b: int = 1) -> int:
    """Adding two variables (small description)

    This function adds two variables via the + operator. (long description)

    Parameters
    -------------
    a : int
        First argument
    b : int
        Second argument

    Returns
    -------------
    int
        Returns a+b
    """
    c = a + b
    return c


# Left-click on the function "add" and Ctrl+q or go to View->QuickDocumentation
# to display the information in the docstring as documentation in PyCharm

# If you do not know how many arguments you will get, you can use *arg to collect
# all positional arguments and **kwargs to collect all keyword arguments.
# If no return statement is used, None is returned by default.
def unknown_arguments(*args, **kwargs):
    print(f"Function got args: {args}")
    print(f"Function got kwargs: {kwargs}")


unknown_arguments(1, 2, 3, a=4, b=5, c=6)


# If you want to pass list or dictionary elements to a function as separated
# arguments, you can use * or ** respectively to unpack the values
def add(a, b=1):
    c = a + b
    return c


arg_list = [1, 2]
unknown_arguments(*arg_list)

kwarg_list = dict(a=1, b=2)
unknown_arguments(**kwarg_list)


#
# Function return values
#

# Returning a value is optional in Python functions. If you do not return
# anything, None will be returned. If you return multiple values, they are
# packed into a tuple:
def return_values():
    return 1, 2, 3, 4


# Here we just leave the return values as a tuple:
result_tuple = return_values()
# But we can also unpack the values:
a, b, c, d = return_values()
# Or we can unpack only the first values and leave the rest as list
# using *listname:
first, second, *others = return_values()


# The return statment not only returns values but also exits the function
def return_values(a: bool, b: int):
    """Will return b**2 if a is False, otherwise returns b"""
    if a:
        return b
    b **= 2
    return b


#
# Recursions
#

# Functions are allowed to call themselves recursively
def add(*args):
    """This function adds up 2 or more arguments recursively"""
    a = args[0]
    # Check if there is only 1 element after the first element. If there is,
    # use the add() function to reduce it to 1 element. Then add the result to
    # the first element.
    if len(args[1:]) == 1:
        b = args[1]
    else:
        b = add(args[1], *args[2:])
    c = a + b
    return c


# Explanation for recursion example:
# Assume we know how to add 2 values. Let's start at the simplest case, where
# args has 2 elements:
def add_two_values(*args):
    a = args[0]
    b = args[1]
    c = a + b
    return c


result = add_two_values(1, 2)

# Now assume that args has 2 or 3 elements but we only know how to add 2
# elements:
def add_two_values(*args):
    a = args[0]
    b = args[1]
    if len(args[2:]):  # If there is another (third) element in the list...
        b2 = args[2]
        b = b + b2  # ... then add the third element to b
    c = a + b  # ... and then add the resulting b to a
    return c


result = add_two_values(1, 2)
result = add_two_values(1, 2, 3)  # Also works with 3 arguments now

# b = b + b2 looks familiar - we solved this with our add_two_values() function
# already! Let's use our add_two_values() function:
def add_two_values(*args):
    a = args[0]
    b = args[1]
    if len(args[2:]):  # If there is another (third) element in the list...
        b2 = args[2]
        b = add_two_values(b, b2)  # ... then add the third element to b
    c = a + b
    return c

result = add_two_values(1, 2)
result = add_two_values(1, 2, 3)

# If we write this in a slightly more general way, we end up with a recursive
# function that can deal with 2 or more arguments:
def add_two_values(*args):
    """This function adds up 2 or more arguments recursively"""
    a = args[0]
    if len(args[1:]) == 1:  # Only 1 element in list left after first element?
        b = args[1]
    else:  # If no, pass the remaining values to add_two_values to add them:
        b = add_two_values(args[1], *args[2:])
    c = a + b
    return c

result = add_two_values(1, 2)
result = add_two_values(1, 2, 3, 4, 5)  # Also works with 2 or more arguments


#
# Mutable objects as function arguments
#

# If you pass a variable that references a list object to a function, the
# argument now references the same list object as the passed variable (same as
# with assigning a list to multiple variables using
# a = [1,2,3]  # If we pass "a" to a function argument "b"...
# b = a  # ...this is what happens at the function call
# . The list is not copied, we still only have 1 list object!
# So if you modify the list object in the function, this also affects the
# variable that references the same list object outside the function.
# The same goes for dictionaries and other mutable objects.
# Note how the example below changes the list elements also outside of the
# function:
def modlist(somelist):
    somelist[0] = 55


mylist = [1, 2, 3]
modlist(mylist)  # this changes the first element in list mylist!


#
# Using a function as an iterable with 'yield'
#
def iterable_function(l):
    # Code we write here will be executed only once
    l *= 2
    for a in range(l):
        # Code we write here will be executed once by iteration because of the
        # yield statement
        print('In function:', a)
        # Variable 'a' will be returned at every iteration
        yield a


for b in iterable_function(5):
    print('Function returned:', b)


#
# Using variables from the outer namespace
#
# Variables created within the function only exist within the function.
# However, variables created outside the function may be used within the
# function.
a = 0


def some_function(b):
    c = a + b
    # This would now no longer allowed since we used the name "a" from outside
    # the function:
    # a = 0
    return c


result = some_function(4)


###############################################################################
# Printing to and reading from console
###############################################################################
# print() can be used to print one or multiple objects to the console
print("Printing a string")
print("Printing multiple items", 34, 245.2)
print(f"Printing multiple items ({34} {245.2}) nicely")
# The end argument states how the printed text should be ended. It defaults to
# a line break \n
print("Printing without a newline at the end", end='')
print(" - There was no linebreak!")
print("Printing without 2 newlines at the end", end='\n\n')

# input() is used to get (and wait for) input from the console
value = input("Give me some input")
# input() reads everything as string!
integer = int(input("Give me an integer!"))


###############################################################################
# Modules
###############################################################################

#
# Importing modules
#

# You can import existing modules (=python files, classes, functions) via
# the import ... or import ... as ... statement. Afterwards you can use its
# contents. With this, the full module will be imported.
import sys
python_executable = sys.executable

# Many python modules have suggested nicknames, You can choose a different
# name of the imported module using the 'as' keyword:
import sys as system
encoding_in_this_python = system.getdefaultencoding()

# Use from ... import ... to only import a specific part of a module
from os import path
correct_path = path.join('some', 'directory')
# ...to be precise, this imports names from the module into our local symbol
# table without importing the module name (e.g. "os" is undefined).

# You can import multiple modules with one statement using commas
import os, sys
from os import path, makedirs

# If you import a module multiple times, python will usually not import it anew


#
# Creating and using custom modules
#

# If you want to reuse content from your own files, you can simply import it
# from there. Always make sure that the PYTHONPATH is set correctly or the
# module is within your working directory. Python will search within the
# PYTHONPATH and the working directory for the module.

# The following line imports function 'add' from file "my_module.py" and
# assigns it the name 'my_add'
from my_module import add as my_add

argument_list = list(range(10))
print(my_add(*argument_list))

# When importing from a file, the content of the file will be executed. Often
# we want to include code that should only be executed when using the
# file as main file (= execution only when calling the file but not when
# importing from it). This can be done using the following syntax:

print("This code will be executed when this file is imported")

if __name__ == '__main__':
    print("This code will not be executed when this file is imported")

# You can check the example in my_module.py by executing my_module.py directly

# You can access directories as "classes" to import files from by placing an
# empty __init__.py file in the directory. (See exercise 05_classes.py later.)

# More information on modules:
# https://docs.python.org/3.7/tutorial/modules.html#more-on-modules
