# -*- coding: utf-8 -*-
"""09_solutions.py

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
compare your solution to the solution given in the file 09_solutions.py.

"""

###############################################################################
# 09 matplotlib
###############################################################################

#
# Task 1
#

# You are given the following pixel data as numpy array:
import numpy as np
a = np.zeros(shape=(10, 10), dtype=np.uint8)
a[0, 0] = 255  # 255 is the highest pixel value for uint8 (only 2^8 values)
# The code below plots the image. Take a look at it. You should see 1 bright
# pixel, the other pixels should have value 0.
# Create a loop in which the bright pixel moves from left to right and top to
# bottom of the image. In detail, it should move 1 column to the right at each
# iteration. If the bright pixel would move out of the image, move it to the
# first (=left-most) column, one row below the previous row. Plot the image at
# each iteration and use plt.pause(0.001) after plotting (otherwise the pixel
# will move too fast).
# Hint: For efficient plotting, re-use the handle from .imshow() and update
# the axis data instead of creating a new figure or axis:
# fig, ax = plt.subplots()  # Create figure and axis
# imshow_handle = ax.imshow(a)  # Plot image on axis
# imshow_handle.set_data(a)  # Update axis data
# plt.pause(0.001)  # Pause for 0.001 seconds (we could also use time.sleep())
# imshow_handle.set_data(a)  # Update axis data again

from matplotlib import pyplot as plt
plt.ion()
fig, ax = plt.subplots()
plt.show()
imshow_handle = ax.imshow(a)

# Your code here #


#
# Task 2
#

# Load the pixel data from file "cdc_6600_low_resolution.jpg" into a numpy array.
# The code blow plots the image. Take a look at it. What you are seeing is the
# fastest memory device of its time (1965), it could hold 1156 bit and the
# state of the bits is visible to the bare eye (the darker grid-points are
# high (1) bits, others are low (0)).
# Create a loop in which the each pixel moves from left to right and top to
# bottom of the image (as in task 1 but for every pixel). In detail, they
# should move 1 column to the right at each iteration. If the pixels would move
# out of the image, move them to the first (=left-most) column, one row below
# the previous row. Pixel that reached the bottom-right of the pixel shall be
# placed at the top-left of the image instead of moving out of the image.
# Plot the image at each iteration and use plt.pause(0.00001) after plotting
# (otherwise the pixel will move too fast).
# Hint: Work with slices where possible and reuse the numpy array instead of
# creating new arrays. We have 2 spatial dimensions and one
# color channel. Maybe this task will get easier when the 2 spatial dimensions
# are seen as one flat array?
import numpy as np
from matplotlib import pyplot as plt
image_data = plt.imread("cdc_6600_low_resolution.png")
image_data = np.array(image_data)
fig, ax = plt.subplots()
plt.show()
imshow_handle = ax.imshow(image_data)
original_image_shape = image_data.shape
print(f"Image shape is: {original_image_shape}")

# Your code here #


#
# Task 3
#

# Extend the example for the animation from the file 09_matplotlib.py to RGB
# instead of grayscale.

# Your code here #
