# -*- coding: utf-8 -*-
"""05_solutions.py

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

Example solutions for tasks in file 05_tasks.py.

"""

###############################################################################
# 05 Files
###############################################################################

#
# Task 1
#

# Create a File 'myfile.txt' and write some text to it. Then open it again and
# append some text. Finally, open it only to read all the content and print it
# to the console. Use "with" blocks to open the files.

# Your code here #

with open('myfile.txt', 'w') as wf:
    wf.write('This will create or overwrite the file!')
    print("Pay attention to the default newline at the end of the text when using print().", file=wf)

with open('myfile.txt', 'a') as af:
    af.write('This text will be appended to the existing file without overwriting it.\n')

with open('myfile.txt', 'r') as rf:
    # Read file at once (entire content will be stored in memory!)
    print(rf.read())


#
# Task 2
#

# Use the glob module to print a list of all files ending in ".txt" in the
# current working directory and all subdirectories.

# Your code here #

import os
import glob
found_files = glob.glob(os.path.join('**', '*.txt'), recursive=True)
print(f"{found_files}")
