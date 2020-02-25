# -*- coding: utf-8 -*-
"""06_os_sys_subprocess.py

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

In this file we will look into how to use the os/sys modules to access OS
operations and to get the arguments passed to our Python script using argparse.
We will also see how we can start external programs in the background using
the subprocessing module and how to call functions or external programs in a
parallel fashion using the multiprocessing module.
Using the modules shown in this unit, Python can be used as powerful
alternative to shell-/bash-scripts to call and communicate with other programs
and to use multiple processes, while being largely independent of the OS.
"""

###############################################################################
# os - using the operating system in Python
###############################################################################
# The os module allows you to use the operating system terminal/command line
# and call operating system functions independently of the operating system.
# It also provides functions to make path-handling independent from the
# operating system.
# Documentation: https://docs.python.org/3/library/os.html

import os

# We can use os.system() to use the OS terminal
os.system('echo hi')  # This is the same as typing "echo hi" into the terminal

# The OS module also allows us to parse and join file-paths and names
# independent of the operating systems syntax:
# Join multiple paths together
filename = os.path.join('some',' directory',' filename.py')
print(filename) # filename might look different on different OS
# Get directory name
directory = os.path.dirname(filename)
# Get name of file itself
name = os.path.basename(filename)

# We can create directories:
os.makedirs("new_directory", exist_ok=True)

# We can rename files or directories:
os.rename("new_directory", "my_new_new_directory")

# We can remove files:
# os.remove("filetoremove")

# We can remove directories:
os.removedirs("my_new_new_directory")

# We can get the number of CPUs in our machine:
print(os.cpu_count())

# ... and there are a lot more functions provided by the os module.


###############################################################################
# sys - accessing system parameters
###############################################################################
# The sys modules allows you to access system parameters and functions related
# to them. For example, it allows you to access the command line arguments
# passed to your Python script.
# Documentation: https://docs.python.org/3/library/sys.html

import sys

# Here we will get our command line arguments
command_line_args = sys.argv
print(f"I received these arguments: {command_line_args}")
print(f"With types: {[type(a) for a in command_line_args]}")

# Note that first element in sys.argv is always the program name itself.
# The arguments are always read in as strings, if you want numbers, you will
# have to convert them.

# The sys module will also let you view the Python version, the current
# PYTHONPATH variable, and the Python executable:
print(f"Python version: {sys.version}")
print(f"PYTHONPATH: {sys.path}")
print(f"Python executable: {sys.executable}")


###############################################################################
# argparse - getting command line arguments easier
###############################################################################
# The argparse module lets you easily and safely access the command line
# arguments that were passed to your Python program. I recommend this module
# over using the sys module for getting the command line arguments.
# Documentation: https://docs.python.org/3/library/argparse.html

# The usage of argparse is as follows:

# Import argparse module and create a parser instance
import argparse

# Create a parser instance
parser = argparse.ArgumentParser()

# Specify arguments you want to receive. The help text will be displayed when
# your program is called with '-h' (e.g. Python3 my_program -h). The
# argument "type" specifies the data type you want to accept (will be checked
# and converted automatically!).
parser.add_argument('filename', help='some filename', type=str)
parser.add_argument('number', help='some float number', type=float)

# Parse the arguments
args = parser.parse_args()

# -> you can now use args.argumentname to access your arguments, e.g.:
my_filename = args.filename
my_floatnumber = args.number


###############################################################################
# subprocess - spawning and managing subprocesses
###############################################################################
# The subprocess module allows you to execute other programs in the
# background/in parallel, catch their output and error messages, communicate
# with them, and manage them. There are many (partly redundant) functions
# available.
# More information: https://docs.Python.org/3/library/subprocess.html

import subprocess

# call(): call a program via the OS and get its exit-code.
# Usage: exit_code = subprocess.call(["some_program", "arguments"])
exit_code = subprocess.call(["echo", "test"])

# You can also write to the shell directly. This can be a security hazard.
exit_code = subprocess.call("echo test", shell=True)


# shlex.split() can translate a shell line into separate arguments:
import shlex

args = shlex.split("echo test")
print(args)

#
# Running programs in the background
#

# subprocess.Popen allows you to run programs in background:
p = subprocess.Popen(['echo', 'test'])  # this will not wait for echo to finish

# This will wait 15sec for the program to terminate or raise a TimeoutExpired
# exception.
try:
    _ = p.wait(timeout=15)
except subprocess.TimeoutExpired:
    # What should we do if the process doesn't finish in time?
    # We could kill it, give it more time, etc. but for now we kill it
    print("Process couldn't finish in time - killing it!")
    p.kill()

# If we want to access the output of the program, we have to use pipes. Here
# we will get the standard output (stdout) and the error messages (stderr)
# during the execution of the program in the background:
p = subprocess.Popen(['echo', 'test'], stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)

# This will wait 15sec for the program to terminate or raise a TimeoutExpired
# exception. Output or error messages will be sent to variables outs, errs.
try:
    outs, errs = p.communicate(timeout=15)
    print(f"outs: {outs}")
    print(f"errs: {errs}")
except subprocess.TimeoutExpired:
    # What should we do if the process doesn't finish in time?
    # We could kill it, give it more time, etc. but for now we kill it
    print("Process couldn't finish in time - don't forget to kill it!")
    p.kill()
    outs, errs = p.communicate()
    print(f"outs: {outs}")
    print(f"errs: {errs}")

# You can use with-as to access programs running in background in a safe manner
with subprocess.Popen(['echo', 'test'], stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE) as p:
    try:
        outs, errs = p.communicate(timeout=15)
        print(f"outs: {outs}")
        print(f"errs: {errs}")
    except subprocess.TimeoutExpired:
        # What should we do if the process doesn't finish in time?
        # We could kill it, give it more time, etc. but for now we kill it
        print("Process couldn't finish in time - we'll just ignore it and the "
              "with-as block will take care of cleaning up.")


###############################################################################
# multiprocessing - easy and powerful parallelization
###############################################################################
# The multiprocessing module provides functions for easy and safe distribution
# of a task to a pool of worker processes.
# Documentation: https://docs.python.org/3/library/multiprocessing.html

# You can use multiprocessing.Pool.map to apply a function to a list of
# arguments in a prallel fashion. Pool.map will automatically manage the
# processes and work through the list in parallel.

from multiprocessing import Pool


def f(x):
    """This is the function we want to run in parallel. You can also use call
     os.system() or subprocess.call() here to run non-Python programs!"""
    return x*x


# Now we will create a pool of 5 worker processes in the background. Then the
# function f() will be applied to each element in the list list(range(100))
# using all of the worker processes.

# This if-condition is important to only execute the code in the main process
if __name__ == '__main__':
    # This will be the list we want to process
    arguments = list(range(100))
    # Here we create the pool of 5 worker processes
    with Pool(5) as p:
        # And now we use .map() to use the 5 workers to process our list
        pool_returns = p.map(f, arguments)
    print(pool_returns)


# We can also use Pool.map to call os.system() or subprocess.call() in
# parallel:
def f(x):
    """This is the function we want to run in parallel. You can also use call
     os.system() or subprocess.call() here to run non-Python programs!"""
    os.system("echo " + str(x))


# This if-condition is important to only execute the code in the main process
if __name__ == '__main__':
    # This will be the list we want to process
    arguments = list(range(100))
    # Here we create the pool of 10 worker processes
    with Pool(10) as p:
        # And now we use .map() to use the 10 workers to process our list
        _ = p.map(f, arguments)

#
# Multiprocessing using a dynamic iterable
#

# .map() will process the list of arguments at once and store the result in
# a list. This can consume large amounts of memory and we do not see the
# status of the processes.
# The .imap() function addresses these issues. It returns a dynamic iterable
# over which we can iterate in a loop. As with .map(), the pool of workers
# is used to process the arguments but we can access the results as soon as
# they are ready.

def f(x):
    return x ** x


# This if-condition is important to only execute the code in the main process
if __name__ == '__main__':
    # This will be the list we want to process
    arguments = list(range(10))
    # Here we create the pool of 3 worker processes
    with Pool(3) as p:
        # And now we use .imap() to use the 3 workers to process our list
        for return_value in p.imap(f, arguments):
            print(return_value)

# Note that .imap() returns the number of results in the same order as the
# arguments. A faster, but unordered, version is the .imap_unordered()
# function.

# The multiprocessing module also provides tools for shared memory,
# asynchronous pooling, semaphores, etc.

#
# Practical example
#

# Let's combine some of the tools we have seen so far with the tqdm module,
# which creates progress bars. Assume we have some files and want to apply
# an external program to them. Furthermore we want to do some post-processing
# using our function some_postprocessing(). We can easily apply our pipeline
# of calling the external program and using our post-processing in parallel
# using multiprocessing:


def some_postprocessing(preprocessed_input):
    """This function pretends to do some post-processing"""
    # This will just do some calculations that require CPU power for some
    # time
    a = 0
    for i in range(int(1e7)):
        a += 1
    return a


def pipeline(filename):
    """Let's assume this is a function that processes a file with some external
    program and then applies our custom some_postprocessing() function to it"""
    
    # We pretend to call some program that processes the file with name
    # filename. Actually we call the program "sleep" with argument "1", which
    # does nothing for 1 seconds
    with subprocess.Popen(['sleep', '0'], stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE) as p:
        try:
            outs, errs = p.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            # What should we do if the process doesn't finish in time?
            # We could kill it, give it more time, etc. but for now we kill it
            print("Process couldn't finish in time - we will return None to indicate"
                  " there was an issue in this subprocess.")
            return None
    
    # Now we will use the output of the external program an apply some
    # post-processing
    final_output = some_postprocessing(preprocessed_input=outs)
    
    # We will return the filename and the final_output
    return filename, final_output


import tqdm  # This will provide us with a nice progressbar

# This if-condition is important to only execute the code in the main process
if __name__ == '__main__':
    # This will be the list we want to process
    arguments = list(range(1000))
    # Here we create the pool of 8 worker processes
    with Pool(8) as p:
        returns = []
        # And now we use .imap() to use the 8 workers to process our list
        for return_tuple in tqdm.tqdm(p.imap(pipeline, arguments), total=len(arguments),
                                      desc="processing files"):
            returns.append(return_tuple)
    print(f"Collected {len(returns)} returns")
    print("Done!")
