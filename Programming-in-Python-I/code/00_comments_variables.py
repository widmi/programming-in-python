# -*- coding: utf-8 -*-
"""00_comments_variables.py

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

In this file we will learn about syntax, comments, and docstrings in Python,
common Python variables, variable operations, and datatype conversions.

Python offers the option for a description at the beginning of each file,
function, or class which is called "docstring" and enclosed in 3 double quotes.
What you are reading now is such a docstring. Docstrings are for documentation
purposes and not executed as Python code (i.e. don't do anything).

The content in docstrings can range from short notes to extensive documentation
with input/output arguments etc.. It will automatically be parsed (if possible)
and displayed in the help for the specific function or class.
In PyCharm you can toggle this help by left-clicking on a function or class and
pressing Ctrl+q or selecting "Quick Documentation" in the "View" menu bar.

For more information and one of the commonly used docstring-styles please refer
to https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
and http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html.
The general conventions for docstrings can be found here
https://www.python.org/dev/peps/pep-0257/.

For more details on Python functionality in general see the official
tutorial/documentation https://docs.python.org/3/tutorial/index.html.

"""

###############################################################################
# Comments
###############################################################################

# Comments are part of a Python file that are not executed (i.e. don't do
# anything).
# Comments in Python start with a hashtag. You can also place comments after
# non-comment statements using a hashtag.

# There are conventions and advices regarding code style in Python, such as
# https://www.python.org/dev/peps/pep-0008/ . It will help you -a lot- to keep
# your code readable and clean, however, Python does explicitly not force you
# to follow it. We will stick to this style as good as possible in this
# lecture. PyCharm tries to highlight code that goes against PEP standards.

# "A universal convention supplies all of maintainability, clarity,
# consistency, and a foundation for good programming habits too. What it
# doesn't do is insist that you follow it against your will. That's Python!"
#  —Tim Peters on comp.lang.python, 2001-06-16


###############################################################################
# General
###############################################################################

# Python program code is stored in text files.
# A Python code file is executed line by line (from top to bottom of the file).
# The standard filename-suffix indicating a Python file is ".py", e.g.
# "myfile.py".
# It is good practice to limit the width of the code lines to e.g. 80
# characters for better readability on smaller screens.


###############################################################################
# Shebang line
###############################################################################

# You might encounter the line <#!/usr/bin/env python3> at the first line of
# Python files. This is the so-called Shebang line, which can help to increase
# compatibility with UNIX/Linux systems.


###############################################################################
# Variables
###############################################################################

#
# General
#

# Variables in Python are just references to objects stored and generated
# automatically in the background. Variable names must start with characters
# that are not digits and not operators. Python variables are case sensitive.
# Use all lower-case names with underscores for variables.

# In the next line, a variable var is created that points to a string object
# with content "hello, world!", which is created and stored somewhere in
# the background.
# This is done via the assignment operator "=", where the variable name is on
# the left and the content to assign is on the right side of the equal sign.
var = "hello, world!"

# We can change the type and content of a variable at any time, since the
# variable is just a reference to some object stored in the background.
# We can e.g. let it point to the integer 5 now:
var = 5

# You can remove a variable with the del statement but the storage it points to
# will only be freed some time after no more variables are pointing there.
# This happens automatically via the "garbage collector".
del var

# It is not common to end statements with a semicolon in Python. You can use
# it to write multiple statements in one line but it is not encouraged to do so.
var = 5;  # This notation also works but is not recommended

#
# Error messages
#

# Variable names must not start with digits. Try to run this line in your
# Python console without the hashtag:
#1var = 5
# You will see that you receive an error message (SyntaxError). Such error
# messages can help you a lot to find out what went wrong. Try to get familiar
# with reading and using them.

#
# Memory usage
#

# A variable pointing to an object takes up 16+x bytes (in native Python)
# for a 64bit Python3 installation.
# This results from:
# (type pointer (8 bytes) + refcount (8 bytes) + object bytes (x bytes))
# = 16+x bytes
# So we pay for convenient code by inefficient memory usage because we have a
# 16 bytes overhead for every (CPython) object, in addition to the x bytes it
# takes to store the object itself.
# However, you can write memory efficient code in Python, if you use one of
# the many Python packages we will see later.


###############################################################################
# Datatypes (1)
###############################################################################

#
# Booleans ("bool")
#

# Booleans can take the values True or False.
# They are used for binary information, like checking if a condition is
# fulfilled or not, as we will see later.

# Here we create a boolean variable with content True
true_variable = True
# Here we create another boolean variable with content False
false_variable = False

# The 'not' statement can be used to negate a boolean
another_true_variable = not false_variable

# Booleans in native Python3 are stored like integers (see below) and thereby
# use up 24 (for False) or 28 (for True) bytes of memory (16 bytes overhead
# + 8 or 12 bytes for integer).

#
# Integer numbers ("int")
#

# Integer numbers can be created using digits.

# Here we create a variable an_integer that points to an integer 5.
an_integer = 5

# Integers in native Python3 are variable-length objects. This means that
# you do not have to worry about overflow problems with integers with large
# values.
# In practice, a base integer using 4 bytes (=32bits) is stored with an
# additional 8 bytes counter. One integer can use multiple base integer
# memories and the counter counts how many base integer memories are used.
# For an integer, Python3 would use 12 bytes and allocate more memory as
# needed. Combined with the 16 bytes overhead, this results in 28 bytes.
# The integer 0 is an exception that only uses the 8 bytes counter an no 4
# bytes base memory.
# Note: This allocation scheme will not be important for us. All that matters,
# is that we do not have to worry about overflow problems and are aware that
# our memory usage might be inefficient.

#
# Floating point numbers ("float")
#

# Floating point numbers can be created using digits combined with a dot.

# Here we create a variable a_float that points to a float value 3.563
a_float = 3.563

# Alternatively, we can use the character e between digits to indicate decimal
# powers for float numbers. Syntax: multiplier e decimal_power
another_float = 2e1  # Corresponds to 20
yet_another_float = 1e-3  # Corresponds to 0.001

# Floating point numbers in native Python3 are stored as 8 bytes float
# (=64bits float). For storing a CPython float, Python3 would use 8 bytes.
# Combined with the 16 bytes overhead, this results in 24 bytes in total.
# Keep in mind that floating point values are stored via a formula (see slides)
# with a fixed number of bits and therefore can lose precision.


###############################################################################
# Mathematical operations
###############################################################################

# Python supports all sorts of mathematical operations using the operators
# + for addition
# - for subtraction
# * for multiplication
# / for division
# ** for power of (syntax: a ** b corresponds to a to the power of b)

# Addition of ints produces an int
sum_integer = an_integer + an_integer

# Addition of floats produces a float
sum_float = a_float + a_float

# Combining ints and floats produces a float
combination = an_integer + a_float

# Mathematical operations can be combined arbitrarily (common rules for
# order of mathematical operations, including brackets, apply)
combination = ((an_integer + a_float * an_integer) / a_float) ** 5

# Use the shortcuts +=, -=, etc. if you want to modify a variable:
combination += 4  # Equals combination = combination + 4
combination /= 2.5
combination **= 2

# If you combine boolean and other numbers, True will be 1 and False will be 0
combination -= True

# You can break long statements into multiple lines with a backslash \
combination += 4 + 6 - \
    3 + 5

# Or you can just put them into brackets and have linebreaks
combination -= (1 + 2 +
                3)


###############################################################################
# Datatypes (2)
###############################################################################

#
# Strings ("str")
#

# A string is an immutable object consisting of characters.
# "Immutable" means that if you modify a string, you actually create a
# completely new one and will use up memory for both strings.
a_string = "I am a string"

# Strings can be created using double quotes, single quotes, 3 double quotes,
# or 3 single quotes. The outer quote is always the relevant one.
a_string = "I am a string"
a_string = 'I am also a string'
a_string = 'You can use the other quotes " inside a string'
a_string = "... and the other way around: ' "
a_string = "Line one.\nLine 2.\nLine 3."  # \n is the newline character

# 3 quotes may include linebreaks (adds newline-character '\n' automatically)
a_string = """I am a string...
and I include a linebreak,
as I span multiple lines."""

# you can escape the effect of quotes by using the escape-character backslash \
a_string = 'I want to use " and \' in a string without ending it.'

# If you want to use backslashes in strings, you have to 'escape' their
# escape-function:
windows_path = "C:\\Windows\\System32"

# Alternatively, you can use a 'raw' string by putting a r in front of the
# first quote to ignore the special meaning of characters:
windows_path = r"C:\Windows\System32"


#
# There are multiple convenience operations on strings:
#

# Concatenation
conc_strings = "I am " + "3 concatenated" + ' strings!'
conc_strings += " Plus one more!"

# Case transformation
upper_case_string = conc_strings.upper()
lower_case_string = "SILENCE!".lower()

# Repetition
repeated_string = "bla" * 3

# This will not work but try it and try to understand the error message:
#not_possible = "bla" / 3

# Length
string_length = len(conc_strings)

# Counting substrings
number_of_bla = repeated_string.count('bla')

# Splitting or joining strings (from and to lists, which we will learn about
# later)
separated_elements = "my-string-elements".split('-')
joined_elements = ";".join(['my', 'string', 'elements'])

# Replacing and finding substrings (keep in mind that this creates a
# completely new string in the background!)
replacement = "This is a placeholder".replace('placeholder', 'string')
check_for_substring = 'string' in conc_strings
locate_substring = replacement.find('is a')

# You can include all kinds of objects in strings with f"{variablename}" using
# formatted strings. Formatted strings start with a f character
a = 1
b = 2.2
c = a + b
string = f"We calculated c = a + b = {a} + {b} = {c}"

# There are many types of formatting options available, like
long_number = 100/3
formated_string = "You can format floating point " + \
                  "values as :total_minimum_digits.total_max_digits " + \
                  f"{long_number:10.5}"

# Strings can be indexed and sliced like lists, which will be shown later.

# More information:
# Common string operations:
#  https://docs.python.org/3/library/string.html#formatstrings
# Common string methods:
#  http://docs.python.org/3/library/stdtypes.html#string-methods

#
# None
#

# None is the sole value of the datatype NoneType in Python. It typically
# represents the absence of a value. This will play a role later, when e.g. a
# function does not return anything.

a = None


###############################################################################
# Datatype conversions
###############################################################################

# If you want to convert an object to another datatype, be careful since
# you might lose information! You can convert to datatypes like this:
# Conversion of "combination" to float (it actually already was a float here):
as_float = float(combination)
# Conversion of "combination" to int. Note how we lose the information after
# the floating point (not rounded to nearest integer!):
as_int = int(combination)
# Conversion of "combination" to bool. 0 translates to False, everything else
# to True:
as_bool = bool(combination)
# Conversion of "combination" to str. Python will do its best to convert
# meaningfully:
as_string = str(combination)
# NoneType can no be converted to int or float but can be "converted" to str:
#not_possible = int(None)
none_string = str(None)


###############################################################################
# Details on evaluation order
###############################################################################

# Multiple assignments in one line are possible. Assignment is performed from
# right to left.
# The following line will create var3 with content 8, then var2 with content
# of var3, then var1 with content of var2:
var1 = var2 = var3 = 2 * 4


print("End of Unit 00. Check out the tasks in file 00_tasks.py.")
