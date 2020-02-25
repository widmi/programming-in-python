# -*- coding: utf-8 -*-
"""11_decorators_numba.py

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

In this file we will learn how to speed up your Python code by using the numba
package. We will also briefly learn about decorators in Python.

"""

###############################################################################
# Excursion: Decorators
###############################################################################
# Python supports the use of "decorators", which provide a more condense way
# of wrapping functions by other functions or classes. Decorators are just
# alternative syntax, so you do not need to use them. However, the numba
# package provides handy decorators, so we will cover the basics here.


# Let's assume we have a function original_f(a)...
def original_f(a):
    print(a)
    

# ... and we want to wrap this function by a another function wrapper_f().
def wrapper_f(function_to_wrap):
    """This function will return a wrapped version of function_to_wrap"""
    def wrapped_f():
        """This function is the wrapped version of function_to_wrap"""
        return function_to_wrap(5)
    print(f"Wrapping {function_to_wrap} to function with default argument 5")
    return wrapped_f  # return wrapped function


# We could wrap original_f by wrapper_f like this:
original_f = wrapper_f(original_f)
original_f()


# Decorator achieve the same behavior using a different syntax:
def original_f(a):
    print(a)


@wrapper_f  # this is a decorator
def original_f(a):
    print(a)

original_f()


# So
#
# def original_f(a):
#     print(a)
# original_f = wrapper_f(original_f)
#
# is equivalent to
#
# @wrapper_f
# def original_f(a):
#     print(a)


# Decorators can become quite complex and powerful, when used with classes
# instead of functions. There are many different supported syntax versions for
# decorators. For more information on the advanced usage of decorators see
# https://www.python.org/dev/peps/pep-0318/
# https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html
# https://www.python-course.eu/python3_decorators.php


###############################################################################
# Performance in Python: General hints
###############################################################################
# Pure Python code is interpreted code and therefore typically very slow,
# which especially concerns Python loops. To write fast/efficient code in
# Python, one can use Python modules that provide efficient implementations.
# For example, Numpy, pandas, and similar modules are highly optimized and
# -in most cases- fast when their functions are used.
# However, not every operation is covered by these modules and this is where
# performance optimization modules like Cython, numba, theano, TensorFlow,
# and PyTorch can help.
# theano, TensorFlow, and PyTorch are of special importance in
# Machine Learning, so we will learn about them in a dedicated Unit.
# IMPORTANT: In case your code is slow, you should first set some timers to
# locate the parts in your code that actually decrease performance. Then you
# can try to solve these operations via numpy or other packages.


###############################################################################
# Cython
###############################################################################
# The Cython module offers the possibility to write C or pseudo-C in Python. As
# numba is -in my opinion but that is subjective- closer to Python code, I will
# refer to the Cython homepage at this point. Performance-wise, Cython and
# numba are at par, depending on the individual use-cases.
# Homepage: http://cython.org/
# Comparison Cython/Numba:
# https://jakevdp.github.io/blog/2013/06/15/numba-vs-cython-take-2/


###############################################################################
# Numba
###############################################################################
# The numba module utilizes the just-in-time (jit) compiler to speed up your
# Python code. It does so via decorators, that allow you to use different
# levels of optimization for CPU or GPU.
# Homepage: https://numba.pydata.org/numba-doc/dev/index.html
# Quickguide: https://numba.pydata.org/numba-doc/dev/user/5minguide.html
import time
import numpy as np
from numba import jit


#
# Slow Python version
#

# Let us assume we have a function with a Python for-loop (=slow)
def python_function():
    """Some Python function we want to optimize"""
    b = 0
    for a in range(int(1e6)):
        b += a
    return b


# We perform 10 runs and log the average time it took (1000 runs would take too long)
start = time.clock()
for _ in range(10):
    _ = python_function()
native_python = (time.clock() - start) / 10
print("time for Python version: {}".format(native_python))


#
# Similar numpy version
#

# We could write this as numpy operations but it would require more memory
# (because range() is a generator and np.arange() instantly creates the whole array):
start = time.clock()
for _ in range(1000):
    _ = np.sum(np.arange(int(1e6)))
numpy_version = (time.clock() - start) / 1000
print("time for numpy version: {}".format(numpy_version))


#
# Simple numba usage scenario (might not yield performance gain)
#

# Use the jit decorator provided by numba to optimize your function. You
# do not need to worry about Python datatypes and objects, numba supports
# almost anything in this mode. BUT: numba can not make full usage of
# optimization (it still needs to deal with dynamic typing etc.) and
# might actually fall back to native Python ('object mode') in some cases.
@jit  # this is the numba decorator
def python_function():
    """Some Python function we want to optimize"""
    b = 0
    for a in range(int(1e6)):
        b += a
    return b

# We perform 1000 runs and log the time it took
start = time.clock()
for _ in range(1000):
    _ = python_function()
simple_jit_version = (time.clock() - start) / 1000
print("time for simple jit version: {}".format(simple_jit_version))


#
# Advanced numba usage scenario (some restriction but better performance)
#

# To get better optimization, you may use the jit decorator provided by numba
# while explicitly specifying the datatypes and avoiding some Python functions
# (e.g. dicts).
# You may set nopython=True to True to disable falling back to the potentially
# slow 'object mode'.
# Syntax: @jit(return_type(arg_type, arg_type, ...), nopython=True)
# More details: https://numba.pydata.org/numba-doc/dev/user/jit.html
from numba import int64


@jit(int64(), nopython=True)  # here we specify the numba datatype explicitly
def python_function():
    """Some Python function we want to optimize"""
    b = 0
    for a in range(int(1e6)):
        b += a
    return b
    
# We perform 1000 runs and log the time it took
start = time.clock()
for _ in range(1000):
    _ = python_function()
nopython_jit_version = (time.clock() - start) / 1000
print("time for nopython jit version: {}".format(nopython_jit_version))


# The jit decorator allows you to disable the 'global interpreter lock' (gil)
# via 'nogil=True' but you have to be careful about race conditions then. This
# will only work in non-object mode (nopython=True) and result in the function
# not locking the objects it accesses (multiple functions can work in
# parallel).
from numba import int64


@jit(int64(), nopython=True, nogil=True)
def python_function():
    """Some Python function we want to optimize"""
    b = 0
    for a in range(int(1e6)):
        b += a
    return b

# We perform 1000 runs and log the time it took
start = time.clock()
for _ in range(1000):
    _ = python_function()
nogil_jit_version = (time.clock() - start) / 1000
print("time for nogil jit version: {}".format(nogil_jit_version))


print("Our timings:\n\tpure_python: {}\n\t".format(native_python) +
      "~numpy: {}\n\t".format(numpy_version) +
      "simple_jit: {}\n\t".format(simple_jit_version) +
      "nopython_jit: {}\n\tnogil_jit: {}".format(nopython_jit_version,
                                                 nogil_jit_version))
# On my machine I got:
# Our timings:
# 	pure_python: 0.046863452
# 	~numpy: 0.0013845139999999958
# 	simple_jit: 5.141400000000118e-05
# 	nopython_jit: 1.7799999999823513e-07
# 	nogil_jit: 2.3000000000195087e-07
# note that nogil did not have any effect as we only ran 1 numba function at a time


#
# Working with arrays in numba
#

# Numba supports numpy arrays as datatypes but you will have to set the
# number of dimensions, e.g. float32[:,:], for a float32 numpy array with
# 2 dimensions.
from numba import void, float32
my_array = np.arange(10*10, dtype=np.float32).reshape((10, 10))
print(my_array)


@jit(void(float32[:, :]), nopython=True)
def python_function(array):
    """Some Python function we want to optimize"""
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            # Using numba in nopython mode -> for loops are fast
            prod = array[x, y] * array[y, x]
            if prod >= 50:
                # We remember from previous Units that accessing a mutable
                # object via indices changes the array we passed to the
                # function:
                array[x, y] = 0.


_ = python_function(array=my_array)
print(my_array)

# You can view the compiled code as assembly code using
_ = [print(code) for code in python_function.inspect_asm().values()]

# More performance hints:
# https://numba.pydata.org/numba-doc/dev/user/performance-tips.html

# Numba provides many (partly experimental) features, such as caching the
# compilation results to files, automatic parallel computations, GPU support,
# etc. See the documentation at https://numba.pydata.org/ for more information.

# Some nerdy entertainment:
# https://www.youtube.com/watch?v=-4tD8kNHdXs
