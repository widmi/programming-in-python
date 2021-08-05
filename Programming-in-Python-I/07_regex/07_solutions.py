# -*- coding: utf-8 -*-
"""07_solutions.py

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

Example solutions for tasks in file 07_tasks.py.

"""

###############################################################################
# 07 Regex
###############################################################################

#
# Task 1
#

# You are given a string "some_header". Use a regex to extract the ID-value,
# that is the content of the line starting with "# id: " but without the
# "# id: ". Regex can also match "\n" characters, so can search for some
# pattern that starts with "# id: " and ends with "\n".

some_header = """
# alpha: 55
# beta: 62
# some stuff
# id: A523B
# some stuff
"""

# Result should be:
# "A523B"

# Your code here #
import re
# This regex will match a string starting with " id:", followed by any number
# of white space characters " *", and any number of characters "(.*)" followed
# by a "\n". It will return as group 1 the characters in the group "(.*)".
pattern = "# id: *(.*)\n"
matchobject = re.search(pattern, some_header)
result = {matchobject.group(1)}
print(f'{result}')


#
# Task 2
#

# You are given a string "some_string". In the string there are 2 words that
# are separated by any number of whitespace characters. Extract the two words
# in the string without the whitespace characters and put them in a list
# "words".

some_string = "first_word          second_word"

# Result should be:
# words = ["first_word", "second_word"]

# Your code here #
import re
# This regex will match a string starting with any number larger than 0 of
# characters "(.+?)", followed by any number larger than 0 of white space
# characters " +",
# followed by any number larger than 0 of characters "(.+)".
# Since we make "(.*?)" non-greedy and strings are scanned from start to end,
# we will search for the shortest string followed by whitespace characters.
# After that the " +" pattern will match as many whitespace characters as it
# can (because it is greedy). Finally, "(.+)" will collect all other
# characters after the whitespace characters.
pattern = "(.+?) +(.+)"
matchobject = re.search(pattern, some_string)
matches = matchobject.groups()
words = list(matches)
print(f'{words}')


#
# Task 3
#

# You are given a list of strings "some_strings". In each string there are 2
# words that are separated by any number of whitespace characters. Extract
# the two words for each string in the list.
# Create 2 lists, "first_words" and "second_words" that each contain the
# collected first and second words.

some_strings = ["first_word second_word",
                "first_other_word   second_other_word",
                "other_first_word          other_second_word"]

# Result should be:
# first_words = ["first_word", "first_other_word", "other_first_word"]
# second_words = ["second_word", "second_other_word", "other_second_word"]


# Your code here #
import re

pattern = "(.+?) +(.+)"
first_words = []
second_words = []

for some_string in some_strings:
    matchobject = re.search(pattern, some_string)
    matches = matchobject.groups()
    first_words.append(matches[0])
    second_words.append(matches[1])

print(f'first_words: {first_words}')
print(f'second_words: {second_words}')
