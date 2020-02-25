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

# Extend the example for the animation from the file 09_matplotlib.py to rgb
# instead of grayscale.

# Your code here #
import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt

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
video = animation.FuncAnimation(fig=fig, func=update_frame,
                                           frames=100,
                                           fargs=(frames, imshow_handle))
video.save(filename='09_solutions_video.avi', fps=5)
plt.close(fig)
del fig
