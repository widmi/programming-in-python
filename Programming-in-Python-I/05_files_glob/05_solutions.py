# -*- coding: utf-8 -*-
"""05_solutions.py

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

import csv

# Step 1
# Open file for writing, over-write if exists
with open("datafile.csv", "w", newline="\n") as datafile:
    # Prepare csv writer
    csv_writer = csv.writer(datafile, delimiter=",")
    # Write first row with column names
    csv_writer.writerow(col_names)
    # Write next 3 rows from list 'df'
    for row in df[0:3]:
        csv_writer.writerow(row)

# Step 2
# Open file for appending
with open("datafile.csv", "a", newline="\n") as datafile:
    # Prepare csv writer
    csv_writer = csv.writer(datafile, delimiter=",")
    # Write remaining rows from list 'df'
    for row in df[3:]:
        csv_writer.writerow(row)

# Step 3
# Prepare a list to store the data from the csv file
read_df = []
# Open file for reading
with open("datafile.csv", "r", newline="\n") as datafile:
    # Prepare csv writer
    csv_reader = csv.reader(datafile, delimiter=",")
    # Read all rows into our list 'read_df'
    for row in csv_reader:
        read_df.append(row)
    # Here we can close the file, since we already read its contents

# Add 'Sum' as last element in the list of names:
col_names = read_df[0]
col_names.append('Sum')
# Compute the sum of the last 3 columns
for row_i, row in enumerate(read_df):
    if row_i == 0:
        # Skip the first row (=column names)
        continue
    
    # Compute sum over last 3 columns in current row
    assignment_sum = sum([int(i) for i in row[-3:]])
    
    # Alternative (no list comprehension):
    # assignment_sum = 0
    # for element in row[-3:]:
    #     assignment_sum += int(element)
    
    # Append sum to row
    row.append(assignment_sum)

# Open file for over-writing
with open("datafile2.csv", "w", newline="\n") as datafile:
    # Prepare csv writer
    csv_writer = csv.writer(datafile, delimiter=",")
    # Write read_df to file
    for row in read_df:
        csv_writer.writerow(row)
