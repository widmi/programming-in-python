# -*- coding: utf-8 -*-
"""useful_statements_modules.py

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

In this file contains some useful statements and modules for various
tasks.

"""
###############################################################################
# assert - checking conditions in debug runs
###############################################################################
# The assert keyword allows you to check for conditions in debug runs
# Homepage: https://docs.python.org/3/reference/simple_stmts.html
a = 1
b = 2
assert a == b,  'This raises an exception!'


###############################################################################
# pandas - fast&flexible dataframes
###############################################################################
# Pandas allows you to use dataframes, which are arrays that can be indexed via
# keyword, integers, and fancy indexing. Dataframes are similar to R dataframes
# and offer multiple operations on the data, even covering reading or writing
# to .csv files. Pandas is fast for most operations.
# Homepage: http://pandas.pydata.org/


###############################################################################
# PyQt and PySide - GUI creation in Python
###############################################################################
# PyQt allows you to create GUIs in Python but has a more restricitve (GPL)
# licence. If you want to create commercial programs, PySide is a good
# alternative.
# QtDesigner offers an easy-to-use editor to create GUIs.
# PyQT: https://wiki.python.org/moin/PyQt
# PySide: https://wiki.qt.io/PySide
# QtDesigner: https://wiki.ubuntuusers.de/Qt_Designer/
