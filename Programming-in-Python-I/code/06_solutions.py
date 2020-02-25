# -*- coding: utf-8 -*-
"""06_solutions.py

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

Example solutions for tasks in file 06_tasks.py.

"""

###############################################################################
# 06 Command line arguments and subprocess
###############################################################################

#
# Task 1
#

# Create a Python script that expects an integer as command line argument. It
# should then print this integer.


# Your code here #

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('myarg', type=int)

# Parse the arguments
args = parser.parse_args()

print(args.myarg)


#
# Task 2
#

# Save the script from task 1 into a python file with name "myprog.py".
# Call this program using subprocess module with command line arguments
# from 0 to 10 (including 10).


# Your code here #
import sys
import subprocess

python = sys.executable
for i in range(11):
    output = subprocess.call([python, "myprog.py", str(i)])
    print(output)


#
# Task 3
#

# Take the program "echo", which will just print what you pass to it, and
# apply it to values from 0 to 500. Use 8 parallel worker threads for the
# task via the Pool.map function from the multiprocessing module. Look at
# the outputs - since it is asynchronous the values might be printed in
# different orders (but the returned list of results would be in the same
# order as the list of arguments).

from multiprocessing import Pool

# Your code here #
n_worker_processes = 8
values = list(range(501))


def f(x):
    _ = subprocess.call(["echo", str(x)])


if __name__ == '__main__':
    with Pool(n_worker_processes) as p:
        _ = p.map(f, values)
