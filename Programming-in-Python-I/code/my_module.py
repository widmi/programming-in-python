# -*- coding: utf-8 -*-
"""my_module.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.10.2019

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This  material,  no  matter  whether  in  printed  or  electronic  form,
may  be  used  for personal  and non-commercial educational use only.
Any reproduction of this manuscript, no matter whether as a whole or in parts,
no matter whether in printed or in electronic form, requires explicit prior
acceptance of the authors.

###############################################################################

Example for 03_functions_print_input_modules.py

"""


def add(*args):
    """This function adds up any arbitrary number of arguments recursively"""
    a = args[0]
    if len(args[1:]) == 1:
        b = args[1]
    else:
        b = add(args[1], *args[2:])
    c = a + b
    return c


if __name__ == '__main__':
    print("This code will not be executed when this file is imported")
    print(f"Example: add([1, 2, 3, 4]) -> {add([1, 2, 3, 4])}")
