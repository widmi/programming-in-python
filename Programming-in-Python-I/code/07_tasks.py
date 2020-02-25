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

Tasks for self-study. Try to solve these tasks on your own and
compare your solution to the solution given in the file 07_solutions.py.

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
