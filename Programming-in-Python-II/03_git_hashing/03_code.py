# -*- coding: utf-8 -*-
"""03_hashing.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.02.2021

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

In this file we will learn how to create hash values in Python.
"""

###############################################################################
# Excursion: Hashing in Python
###############################################################################
# If we want to check for duplicates of data points, i.e. duplicates of files,
# we can use "hash functions" to map the file content to a fixed-size vector
# (the "hash value") and then search for duplicates of these vectors. Hash
# functions are designed to be fast to compute (in the average case) and to
# have a minimal number of collisions (=multiple inputs resulting in the same
# hash value).
# https://docs.python.org/3/library/hashlib.html

# In Python hashing can be done using the module "hashlib":
import hashlib
import numpy as np

# hashlib provides many different hash functions, we will use sha256 here:
hashing_function = hashlib.sha256()

# These hash function objects are class instances that can be fed
# bytes-like objects. Their method .update() is be used to feed them data and
# the .digest() method is used to compute the hash from all the data fed via
# .update() so far.

# This is what we want to hash
some_data = 'A string'
# We first need to encode the characters as bytes (=values in
# range 0 <= x < 256). For this we must specify the encoding of the
# characters. We will use the UTF encoding.
some_data = bytes(some_data, encoding='utf')
# Let's feed it to our hash object
hashing_function.update(some_data)
# And compute the hash-value
first_hash = hashing_function.digest()
print(f"hash-value for 'A string': {first_hash}")

# Let's check if the hash function is consistent
hashing_function = hashlib.sha256()
some_other_data = 'A string'
hashing_function.update(bytes(some_other_data, encoding='utf'))
second_hash = hashing_function.digest()
print(f"hash-function returns same output for same input: "
      f"{first_hash == second_hash}")

# Let's check if the hash function is returning different output for different
# inputs
hashing_function = hashlib.sha256()
some_data = 'Another string'
hashing_function.update(bytes(some_data, encoding='utf'))
some_data = '... and add some more'
hashing_function.update(bytes(some_data, encoding='utf'))
third_hash = hashing_function.digest()
print(f"hash-function returns same output for different input: "
      f"{first_hash == third_hash}")
print(f"But hash-values have same length: "
      f"{len(first_hash) == len(third_hash)}")

#
# Computing hashes of numpy arrays
# A fast way to compute hash values for numpy arrays, is to first convert
# the array to bytes using the .tostring() method and then hashing the array.
#
some_array = np.arange(1000)
some_array_bytes = some_array.tostring()
hashing_function = hashlib.sha256()
hashing_function.update(some_array_bytes)
array_hash = hashing_function.digest()
print(f"hash-value for some_array: {array_hash}")
print(f"hash-values still have same length: "
      f"{len(first_hash) == len(array_hash)}")

#
# Salty hashes
# For sensitive applications, e.g. password hashing, salt (=secret byte offset)
# is applied before hashing to increase resistance against brute-force attacks.
# For our purpose, we do not need (and do not want) salt in our hash-values.
#
# Compute hash function with salt
some_array = np.arange(1000)
some_array_bytes = some_array.tostring()
hashing_function = hashlib.blake2b(salt=b'some salt')  # Our salt
hashing_function.update(some_array_bytes)
array_hash_1 = hashing_function.digest()

# Compute hash with different salt
some_array = np.arange(1000)
some_array_bytes = some_array.tostring()
hashing_function = hashlib.blake2b(salt=b'some salt 2')  # Different salt
hashing_function.update(some_array_bytes)
array_hash_2 = hashing_function.digest()
print(f"hash-values for arrays with different salt equal: "
      f"{array_hash_1 == array_hash_2}")

#
# Python hash() built-in function
# Python provides a built-in hash() function, that is e.g. used for hashing
# dictionary keys. This hash() function will add random salt that is constant
# within an individual Python session.
#

# This hash-value will be different for different Python sessions!
python_hash = hash(some_array_bytes)
print(f"Python built-in hash of array: {python_hash}")
