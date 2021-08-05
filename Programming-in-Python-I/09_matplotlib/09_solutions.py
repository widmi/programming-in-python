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

Example solutions for tasks in file 09_tasks.py.

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

# Setting only the pixel at current position to 1 while looping over all
# positions:
for row_i in range(a.shape[0]):
    for column_i in range(a.shape[1]):
        print(f"Pixel at ({row_i}, {column_i})")
        # Set pixel to 255 at current row/column position
        a[row_i, column_i] = 255
        imshow_handle.set_data(a)
        plt.pause(0.01)
        plt.show()
        # Set to 0 after plotting so that we can re-use the array
        a[row_i, column_i] = 0

#
# Task 2
#

# Load the pixel data from file "cdc_6600_low_resolution.png" into a numpy
# array. The code blow plots the image. Take a look at it. What you are seeing
# is the fastest memory device of its time (1965), it could hold 1156 bit and
# the state of the bits is visible to the bare eye (the darker grid-points are
# high (1) bits, others are low (0)). (See "cdc_6600.jpg" for better quality.)
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
# Flatten spatial dimensions
image_data_flat = image_data.reshape((-1, 3))
# If we shift each pixel in the flat spatial dimensions to the right, the
# pixels will automatically move 1 position to the right in each row and "jump"
# to the next row when reaching the end of a row. See unit 08, linear and 2D
# array for more information.
# `image_data_flat` is only a view on the data in image_data, so if we modify
# the values in `image_data_flat`, we also modify the values in `image_data`.

for i in range(int(image_data_flat.shape[0] / 4)):
    # Plot the image data
    imshow_handle.set_data(image_data)
    plt.pause(0.00001)
    # Shift all pixels to the right in the flattened spatial dimensions
    image_data_flat[:] = np.concatenate([image_data_flat[-1:],
                                         image_data_flat[:-1]], axis=0)


#
# Task 3
#

# Extend the example for the animation from the file 09_matplotlib.py to rgb
# instead of grayscale.

# Your code here #
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

frames = np.zeros((100, 100, 3, 4))

# Now we will set different image data for each frame
frames[:50, :50, 0, 0] = 1
frames[:50, 50:, 1, 1] = 1
frames[50:, 50:, 2, 2] = 1
frames[50:, :50, 0:2, 3] = 1

# Let's plot our frames
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(frames[:, :, :, 0])
ax[0, 1].imshow(frames[:, :, :, 1])
ax[1, 1].imshow(frames[:, :, :, 2])
ax[1, 0].imshow(frames[:, :, :, 3])


# We need to define a function that sets the content of a frame based on the
# current frame index:
def update_frame(frame_number, data, imshow_handle):
    """This function will update the frame content for each frame of the video.

    Parameters
    ----------
    frame_number: int
        Index of current frame in video
    data
        The numpy array we get our frames from
    imshow_handle
        The handle to our imshow object
    """
    # We update the current content of the imshow object with the data we
    # want to have at this frame:
    data_for_frame = data[..., frame_number % 4]
    imshow_handle.set_data(data_for_frame)
    # Then we return a tuple with the imshow handle
    return (imshow_handle,)


# Create a figure
fig, ax = plt.subplots()
# Plot the initial frame
imshow_handle = ax.imshow(frames[:, :, :, 0])
# Create an animation object using our update function. Create an animation
# with 100 frames:
video = animation.FuncAnimation(fig=fig, func=update_frame, frames=100,
                                fargs=(frames, imshow_handle))
video.save(filename='09_solutions_video.avi', fps=5)
plt.close(fig)
del fig
