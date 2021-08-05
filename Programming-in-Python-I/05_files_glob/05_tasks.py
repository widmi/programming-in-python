# -*- coding: utf-8 -*-
"""05_tasks.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.10.2019

Add task 3 by Van Quoc Phuong Huynh, Michael Widrich -- WS2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Tasks for self-study. Try to solve these tasks on your own and
compare your solution to the solution given in the file 05_solutions.py.

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


#
# Task 2
#

# Use the glob module to print a list of all files ending in ".txt" in the
# current working directory.

# Your code here #


#
# Task 3
#

# Step 1:
# Create a csv file 'datafile.csv', which uses the character ',' as column
# separator and '\n' as row delimiter.
# Write the column names from list 'headers' as first row to the file.
# Write the first 3 lists in list 'df' as rows into the file (i.e. the second
# row in the csv file should be "1,Alice,20,30,30\n").
# Step 2:
# Open the csv file 'datafile.csv' to write the remaining 2 lists in list 'df'
# as rows to the file. These rows should be written to the end of the file
# without deleting the prior content of the file.
# Step 3:
# Open the file to read data from the csv file 'datafile.csv' to a list.
# Calculate the sum of points in the last 3 columns for each row except for
# the first row (the one with the column names).
# Add the resulting sums as new column with column name "Sum" as last column.
# Write the resulting table to the file 'datafile2.csv'.
col_names = ['ID', 'Name', 'Assignment1', 'Assignment2', 'Assignment3']
df = [['1', 'Alice', 20, 30, 30],
      ['2', 'Malice', 15, 30, 25],
      ['3', 'Bob', 20, 25, 30],
      ['4', 'Buddy', 20, 25, 25],
      ['5', 'Mary', 20, 30, 25]]

# Desired file content for 'datafile.csv' (without the # characters):
# ID,Name,Assignment1,Assignment2,Assignment3
# 1,Alice,20,30,30
# 2,Malice,15,30,25
# 3,Bob,20,25,30
# 4,Buddy,20,25,25
# 5,Mary,20,30,25

# Desired file content for 'datafile2.csv' (without the # characters):
# ID,Name,Assignment1,Assignment2,Assignment3,Sum
# 1,Alice,20,30,30,80
# 2,Malice,15,30,25,70
# 3,Bob,20,25,30,75
# 4,Buddy,20,25,25,70
# 5,Mary,20,30,25,75

# Your code here #
