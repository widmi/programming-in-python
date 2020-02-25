# -*- coding: utf-8 -*-
"""07_regex.py

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

In this file we will look into how to use the re module to search for
more complex patterns in strings via regular expressions ("regex").
"""

###############################################################################
# re - searching for complicated patterns in text via regular expressions
###############################################################################
# The re module allows you to search for complex patterns in text. If you are
# only looking for simple patterns, the native python string functions (e.g.
# 'stringwithsubstring'.find('substring') are simpler and faster and should be
# preferred.
# Regex syntax can become quite complex. You can find the documentation at
# https://docs.python.org/3/library/re.html .
# You can use tools like https://www.debuggex.com/r/gj5buG9fdS-UJQHu to
# debug a regex.
# If you are struggling to build a specific pattern, it might pay off to
# google it first. Chances are that someone already created a similar pattern.

import re

#
# Searching for one occurrence of a pattern
#

# There are two important functions for finding a pattern or a group of
# patterns in a string: re.search() and re.match().

# re.search() will search for the first occurrence of a pattern and return a
# MatchObject object if it found a pattern. If no pattern is found, None will
# returned.
# Strings are scanned from start (left) to end (right).
pattern = 'Elm Street'
text = 'Ross McFluff: 155 Elm Street'
matchobject = re.search(pattern, text)

print(f'{text} + {pattern} -> {matchobject}')

# A MatchObject evaluates to True for conditions if a pattern was found. We can
# use if conditions to check if the pattern was found:
if matchobject:
    # This will only be executed if a pattern was found
    print(f'{text} + {pattern} -> {matchobject}')

# re.match() will search for patterns only at the beginning of the string (even
# if it is a multi-line string).
pattern = 'Ross Mc'
text = 'Ross McFluff: 155 Elm Street'
matchobject = re.match(pattern, text)

print(f'{text} + {pattern} -> {matchobject}')

pattern = 'Elm Street'
text = 'Ross McFluff: 155 Elm Street'
matchobject = re.match(pattern, text)

# Note that this will return None because no pattern was found
print(f'{text} + {pattern} -> {matchobject}')


#
# Returning groups within patterns
#

# You can use groups to only return sub-patterns within a search-pattern.
# Groups are created using brackets ().

# This will match the string 'Elm Street' and return 'Elm' and 'Str'
# separately in groups:
pattern = '(Elm) (Str)eet'
text = 'Ross McFluff: 155 Elm Street'
matchobject = re.search(pattern, text)
print(f'{text} + {pattern} -> {matchobject}')

# You can access the found pattern(s) with MatchObject.group(i),
# where i is the group-number of the found pattern you want to retrieve.
# MatchObject.group() or MatchObject.group(0) will return the complete
# pattern, MatchObject.group(1) the first group, MatchObject.group(2) the
# second group and so on.
# MatchObject.groups() will return all found pattern groups.
print(f'{text} + {pattern} -> .groups() -> {matchobject.groups()}')
print(f'{text} + {pattern} -> .group() -> {matchobject.group()}')
print(f'{text} + {pattern} -> .group(0) -> {matchobject.group(0)}')
print(f'{text} + {pattern} -> .group(1) -> {matchobject.group(1)}')
print(f'{text} + {pattern} -> .group(2) -> {matchobject.group(2)}')

# You can also nest groups:
pattern = 'Elm ((Str)eet)'
text = 'Ross McFluff: 155 Elm Street'
matchobject = re.search(pattern, text)
print(f'{text} + {pattern} -> .groups() -> {matchobject.groups()}')
print(f'{text} + {pattern} -> .group() -> {matchobject.group()}')
print(f'{text} + {pattern} -> .group(0) -> {matchobject.group(0)}')
print(f'{text} + {pattern} -> .group(1) -> {matchobject.group(1)}')
print(f'{text} + {pattern} -> .group(2) -> {matchobject.group(2)}')


#
# Getting additional data from MatchObject objects
#

# MatchObject objects contain more than just the found pattern. They also
# let you access information like the start and end position, the width, etc.
# for the individual groups:
print(f'{text} + {pattern} -> .group(1) -> {matchobject.group(1)}\n'
      f'  start, group 1: {matchobject.start(1)}\n'
      f'  end, group 1: {matchobject.end(1)}\n'
      f'  span, group 1: {matchobject.span(1)}')

# Note that the end position is the index+1 to allow for slicing:
print(f'{text[matchobject.start(1):matchobject.end(1)]}')
print(f'{text[matchobject.start(1)]}')
print(f'{text[matchobject.end(1) - 1]}')


#
# Searching for (non-overlapping) multiple occurrences of a pattern
#

# re.findall() will search for all patterns in the text and returns a list.
# re.finditer() also does this but one item at a time. It returns a
# MatchObject.
# Strings are scanned from start (left) to end (right).

pattern = 'bla'
text = 'blablablibla'
match_list = re.findall(pattern, text)
# Note that the function will return None if no pattern was found
print(f'{text} + {pattern} -> {match_list}')

for i, p in enumerate(re.finditer(pattern, text)):
    print(f'{i}. pattern: {p.group()} start: {p.start()} end: {p.end()}')

# You can again use groups to return sub-patterns:
pattern = '(bl)(a)'
text = 'blablablibla'
match_list = re.findall(pattern, text)
# Note that the function will return None if no pattern was found
print(f'{text} + {pattern} -> {match_list}')


#
# Making a pattern flexible
#

# Patterns can include meta-characters in regex syntax to search for flexible
# patterns. The regex syntax uses the special characters {}[]()^$.|*+?
# If you want to use them as normal characters in a string, you need to escape
# their special function with a preceding backslash "\". For example "\?".

# [] will specify a set of characters to match.
pattern = '[cbr]at'
text = 'cat bat rat dog'
match_list = re.findall(pattern, text)
print(f'{text} + {pattern} -> {match_list}')

# You can use [0-2] to match integers from 0 to 2. [0-9a-fA-F] will e.g. match
# a hexadecimal number (it will match all integers from 0 to 9 and all
# characters from a to f, both upper and lower case).
pattern = '[0-5a-c]at'
text = 'cat bat rat dog 3at 7at'
match_list = re.findall(pattern, text)
print(f'{text} + {pattern} -> {match_list}')

# You can use ^ to negate character patterns. [^0-9] will match all
# non-numerical characters
pattern = '[^0-9]'
text = 'a1b2c3'
match_list = re.findall(pattern, text)
print(f'{text} + {pattern} -> {match_list}')

# There exist predefined groups of characters, such as \d for numerical or \D
# for non-numerical characters. Important: You need to write \ in the string,
# meaning you need to escape the \ or use a raw string:
pattern = r'\d'  # equivalent to '\\d'
text = 'a1b2c3'
match_list = re.findall(pattern, text)
print(f'{text} + {pattern} -> {match_list}')

# See https://docs.python.org/3/library/re.html for an exhaustive list of
# special characters and their meaning.


#
# Searching for alternative patterns
#

# The | character can be used to search for alternative patterns
pattern = '[bcr]at|dog'
text = 'cat bat rat dog'
match_list = re.findall(pattern, text)
print(f'{text} + {pattern} -> {match_list}')

# This can be combined with the brackets () to group search-patterns:
pattern = '(ai|ml) student'
text = 'this matches ai student and ml students and returns "ai" or "ml"'
match_list = re.findall(pattern, text)
print(f'{text} + {pattern} -> {match_list}')


#
# Repetitions in patterns
#

# * will match any number of repetitions and is by default greedy (ie. searches
# for the largest pattern).
# + will match 1 or more repetitions and is also by default greedy.
pattern = '([bcr]at|dog)*'
text = 'catcat batbat ratratrat dog'
matchobject = re.search(pattern, text)
print(f'{text} + {pattern} -> {matchobject.group()}')

# findall will only report the set of captured groups:
pattern = '([bcr]at|dog)+'
text = 'catcat batbat ratratrat dog'
matchobject = re.search(pattern, text)
print(f'{text} + {pattern} -> {matchobject.group()}')

# To be non-greedy, you need to add the suffix ?
pattern = '([bcr]at|dog)+?'
text = 'catcat batbat ratratrat dog'
matchobject = re.search(pattern, text)
print(f'{text} + {pattern} -> {matchobject.group()}')

match_list = re.findall(pattern, text)
print(f'{text} + {pattern} -> {match_list}')


#
# Substituting and splitting strings
#

# There are many more functions available via the re module, such as split
# and sub (substitution).
# Please see https://docs.python.org/3/library/re.html for more information.
