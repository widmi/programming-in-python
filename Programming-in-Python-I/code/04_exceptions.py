# -*- coding: utf-8 -*-
"""04_exceptions.py

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

In this file we will learn how to raise and catch exceptions

"""

###############################################################################
# Exceptions
###############################################################################
# Python supports a trial-and-error mentality. You can dynamically try to
# execute code and if something goes wrong, you handle the problem by either
# applying different code or reporting the error to the user. So instead of
# checking for possible errors in if-else conditions before executing the code,
# the code is executed and if an error arises, the error can be handled
# dynamically. "Errors" during code execution are referred to as "exceptions"
# in Python. Exceptions can be of different types and they can provide
# additional information about the error that occurred.

# Exceptions can be raised (activated) by using the 'raise' keyword with this
# syntax:
# raise TypeOfError("Some error message as string")
a = False  # Set this to True to raise the following exception
if a:
    raise ValueError(f"Variable a was {a} which equals True!")

# Once an exception is raised, the execution of the program jumps to the end
# of the program or until the exception is caught with the 'except' keyword.
# Python enforces "good" programing habits by only allowing this in
# try-except blocks:
try:
    # Here we put our "normal" code. If an exception should occur here, we can
    # "catch" it in the following 'except' blocks.
    a = False  # Set this to True to raise the exception
    if a:
        # If something goes wrong an exception will be raised. We can also do
        # this ourselves by raising it directly:
        raise ValueError(f"Variable a was {a} which equals True!")
except ValueError as ex:
    # We will land here if a ValueError was raised. We can use this to execute
    # code after the exception was raised.
    print(f'A ValueError brought us here! The error was "{ex}"')
    # Important: if we still want the program to terminate, we have to raise
    # the exception again:
    raise ex
except TypeError:
    # We'll land here if a TypeError was raised.
    print("A TypeError brought here!")
    # Since we do not raise any further exception here, the program execution
    # would continue from here.
finally:
    # You can use the 'finally' keyword instead of 'except' to execute this code,
    # independent of raised exception. This can e.g. be used for clean-up.
    print("This will be executed anyway.")

#
# It is also possible to catch multiple exceptions at once:
#
try:
    a = 1 + 'f'  # This will raise a "TypeError"
except (ValueError, TypeError) as ex:
    # We will land here if a ValueError or TypeError was raised.
    print(f"We caught the exception {ex}!")

#
# Try-Except blocks may be nested arbitrarily
#
try:
    a = 1 + 'f'  # This will raise a "TypeError"
except (ValueError, TypeError) as ex1:
    # We'll land here if a ValueError or TypeError was raised.
    print(f"We caught the exception {ex1}!")
    try:
        # We can add other try-except blocks here
        a = 1 + [4, 5]
    except (ValueError, TypeError) as ex2:
        print(f"We caught the exception {ex2}!")
        # We could also raise ex1 or ex2 here
        print(f"ex1: {ex1}")
        print(f"ex2: {ex2}")
    # Here we could raise ex1 but we will just print it
    print(f"ex1: {ex1}")


#
# Exceptions can be used to create smooth program flows and are common in Python
#

def add(a, b):
    try:
        c = a + b
    except TypeError:
        print("a or b not numbers. Trying to convert to float...")
        c = float(a) + float(b)
    return c


# This will work now because we added an try-except block in the function:
result = add(1, '2')
print(f"add(1, '2') -> {result}")
