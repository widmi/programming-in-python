# -*- coding: utf-8 -*-
"""05_files_glob.py

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

In this file we will learn how to open, close, and read from files.
We will also see that the glob module is handy for finding files in
directories.

"""

###############################################################################
# Files
###############################################################################
# Files can be opened with the open() function, which returns a file handle.
# A file handle is an object representing the file but not the content of the
# file. The syntax of the open() function is
# filehandle = open('filepath_and_name', mode)
# with 'mode' being
# '"r"' for read only (only reading from file, no write access),
# '"w"' for (over-)write (write to file and overwrite file if it exists), and
# '"a"' for append (write access where writing will append at
# the end of the file if file already exists).
# If you want to handle a file not as text but as bytes, use 'rb', 'wb', 'ab'.

# This will open the file "my_module.py" in read mode:
f = open('my_module.py', 'r')

# Now we can use the file handle to e.g. read its content as string
content = f.read()
print(content)

# And finally we need to close the file, otherwise it might be blocked
f.close()

#
# Reading from files
#

# Python encourages you to write "safe" code. Needing to close the file by
# hand is problematic (e.g. if exceptions occur and close() is not executed).
# Therefore I STRONGLY recommend to use the with-as statement, which closes
# the file automatically:
with open('my_module.py', 'r') as f:
    # We now have 'f' as file handle available until the code block is left
    print(f.read())
# Here the file handle 'f' would be closed and not be available anymore

# We can iterate over lines in a file using a for loop (using common line-
# breaks such as "\n").
with open('my_module.py', 'r') as f:
    # The for-loop automatically reads the file content line by line:
    for line in f:
        print(line)
# Iterating over a file line by line is slower than reading the whole content
# at once. But it allows us to use less memory since we only have to hold
# parts of the file in the RAM.

#
# Writing to files
#

# The file handle allows us to write to a file using the syntax
# f.write("somestring\n")
# which will write a string, e.g. "somestring", to the file.
# .write() does NOT add a newline character at the end by default.

# If we use 'w' mode, we will either overwrite or create a file and write to
# it.
with open('somefile.txt', 'w') as f:
    f.write("This overwrites an existing file\n"
            "or creates a new one with this text!\n")
    f.write("This adds another line to the file without a newline character "
            "at the end.")
    f.write(" ... but now we add a newline character.\n")

# If we use 'a' mode, we will either append to an existing file or create a new
# file and write to it.
with open('somefile.txt', 'a') as f:
    f.write("And here this line is appended.\n")
    f.write("And another line is appended.\n")

# We can also use the print() function to write to files by passing the
# filehandle as argument "file". Keep in mind that the print() function does
# add a newline character at the end by default.
with open('somefile.txt', 'a') as f:
    print('And another line via print()!', file=f)
    print('Note that print adds a newline by default.', file=f)


###############################################################################
# csv files
###############################################################################

# Comma separated values (csv) files are a common data file format if the data
# shall be stored in text format. In csv files a special character, e.g.
# a comma ",", semicolon ";", or tab "\t", is used as separator between
# columns in the file. A csv file is still just a text file, so using the
# separator is not enforced and we could still print normal text to it.

# The csv module allows you to read from and write to csv files in a flexible
# and convenient manner.
# Documentation: https://docs.python.org/3/library/csv.html

import csv

# Here we write to a csv file. We first open the file as we would for a normal
# text file and specify that '\n' (newline) is the row-separator:
with open('somefile.csv', 'w', newline='\n') as csvfile:
    # Now we create a writer object using the csv module for a tab ('\t')
    # separated file
    csvwriter = csv.writer(csvfile, delimiter='\t')
    
    # This which allows us to use the function .writerow(row) to print
    # the list row to the file, separated by the delimiter we specified
    # earlier.
    for i in range(5):
        _ = csvwriter.writerow([f'col1 row{i}', f'col2 row{i}', f'col3 row{i}'])
    
    # Alternatively, we could use the join() function without the csv module:
    for i in range(5, 10):
        print('\t'.join([f'col1 row{i}', f'col2 row{i}', f'col3 row{i}']),
              file=csvfile)

# Here we read a tab ('\t') separated file line by line. We first open the
# file and specify that '\n' (newline) is the row-separator:
with open('somefile.csv', 'r', newline='\n') as csvfile:
    # We then create our reader object with '\t' as delimiter between columns
    csvreader = csv.reader(csvfile, delimiter='\t')
    
    # We now we can iterate over the rows in csvreader, where each 'row' is
    # a list of separated column elements.
    for row in csvreader:
        # Here we print the list of elements in this line
        print(f"{row}")

#
# Handling large csv files
#

# Modules such as numpy (see unit 08) or pandas (see unit 08) have their own
# built-in functions for reading/writing csv-like files. Pandas e.g. uses a
# C-backend and is much faster than pure Python code - if you're dealing with
# larger data files, this can provide a large speed-up.


###############################################################################
# glob - searching for files in (sub)directories
###############################################################################
# The glob module allows you to (recursively) search for files and folders in
# directories and subdirectories.
# Documentation: https://docs.python.org/3/library/glob.html

import os
import glob

# The syntax for using glob for searching for files is
# found_files = glob.glob("searchpattern")
# where "searchpattern" is a string that decides what files patterns should be
# searched for.

# Glob supports regex-like syntax (regex will be explained in unit 07) in the
# search pattern. We can for example use the '*' character, which matches
# anything within a folder (excluding subfolders).

# This would return a list of all files and folders in the working directory:
found_files = glob.glob("*")
print(f"Files and folders in the current working directory:\n"
      f"{found_files}")

# This would search the working directory directory for files that end in
# '.py':
found_files = glob.glob('*.py')
print(f"Files and folders ending in '.py' in the current working directory:\n"
      f"{found_files}")

# To search for files and folders recursively (=in all subdirectories), you can
# use two '*' characters and set the argument recursive=True. "**" will then
# match all subdirectory names.
found_files = glob.glob('**', recursive=True)
print(f"Files and folders in the working directory and its subdirectories:\n"
      f"{found_files}")

# Search in a directory and in all subdirectories for files ending in ".py":
found_files = glob.glob(os.path.join('**', '*.py'), recursive=True)
print(f"Files ending in '.py' in the working directory and its subdirectories:\n"
      f"{found_files}")
# Always use os.path.join() to join paths. We will learn more about this in
# unit 06.

# Search in folder 'some_folder' and in all its subdirectories for files ending
# in ".py":
dirname = 'some_folder'
found_files = glob.glob(os.path.join(dirname, '**', '*.py'), recursive=True)
