# -*- coding: utf-8 -*-
"""09_matplotlib.py

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

In this file we will learn how to create plots in Python using matplotlib.

"""

###############################################################################
# matplotlib - plotting all kinds of stuff in Python
###############################################################################
# Matplotlib includes plotting functions for all kinds of plotting. The large
# functionality and flexibility of this module unfortunately comes at the price
# of partly over-complicated function calls and parameters.
# Plotting with matplotlib without internet connection is either brave or
# foolish.
# Homepage: https://matplotlib.org/
# Tutorials: https://matplotlib.org/users
# Main tutorial: https://matplotlib.org/tutorials/introductory/usage.html

# Important: matplotlib uses system backends to do the plotting. Different OS
# offer different backends, some of which are specialized for certain tasks.
# If you run into performance issues (e.g. 3D plots or creating videos), you
# may want to switch to a more suitable backend.

# We will now work through parts of the tutorial together. Depending on your
# needs or interests consult the matplotlib gallery
# https://matplotlib.org/gallery
# for examples and code snippets that you can copy/paste.

from matplotlib import pyplot as plt
import numpy as np

# Choose interactive mode on or off
# plt.ioff()  # -> show figures when explicitly stated
plt.ion()  # -> show figures immediately (also depends on backend)

# Data for plotting
t = np.arange(0, 100)

#
# Basic line plot
#
fig, ax = plt.subplots()  # this creates a figure and axis/axes handle
# The axis is the canvas where we can plot something
# The figure is the window that contains axes
ax.plot(t)  # add a plotted line to the axis
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets')  # it allows for many modifications
ax.grid()  # add a grid to our axis
fig.suptitle('This is a figure title',
             fontsize=16)  # set a super-title for the figure
fig.savefig('09_test.pdf')  # we can save the figure to a file
fig.savefig('09_test.png')  # the file extension will be interpreted by matplotlib
plt.close(fig)  # we can close the figure if we don't need it anymore
del fig  # then we can delete the handle

#
# Multiple subplots
#
fig, ax = plt.subplots(2, 3)  # now ax is a 2x3 array containing the axes
ax[1, 1].plot(t)  # select an axis and plot data t
ax[0, 1].plot(t)  # select another axis and plot data t again
ax[0, 1].plot(-t)  # select another axis and plot data t * (-1)
ax[0, 0].plot(t, label='data t')  # plot t again and add a label
ax[0, 0].plot(-t, label='data -t')  # plot -t again and add a label
ax[0, 0].legend()  # Create a legend for the labeled plots
fig.tight_layout()  # Tweak spacing to prevent clipping of ylabel
fig.savefig('09_subplots.png')
plt.close(fig)
del fig

#
# Basic histograms
#
fig, ax = plt.subplots()
# some random data
a = np.random.normal(size=(500,))
# the histogram of the data
n, bins, patches = ax.hist(a, 5, density=1)
# Alternative way of setting labels
ax.set_xlabel('Smarts')
ax.set_ylabel('Probability density')
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')
fig.tight_layout()  # Tweak spacing to prevent clipping of ylabel
fig.savefig('09_histogram.png')
plt.close(fig)
del fig

#
# Plotting images
#
pseudo_image = np.random.uniform(size=(150, 150, 3))  # create random data
fig, ax = plt.subplots()
ax.imshow(pseudo_image)
ax.set_xticks([], [])  # Remove xaxis ticks
ax.set_yticks([], [])  # Remove yaxis ticks
ax.set_title('Some pseudo image data')
fig.tight_layout()
fig.savefig('09_pseudo_image.png')
plt.close(fig)
del fig


#
# Reading image data
#
read_image_data = plt.imread("09_histogram.png")
fig, ax = plt.subplots()
ax.imshow(read_image_data)
ax.set_xticks([], [])  # Remove xaxis ticks
ax.set_yticks([], [])  # Remove yaxis ticks
ax.set_title('The image data we just read')
fig.tight_layout()

# Keeping plots open after program ends:
plt.show(block=True)


#
# Simple animations
#
# Matplotlib provides a submodule for animations but its documentation is
# suboptimal:
# https://matplotlib.org/api/animation_api.html
# https://matplotlib.org/gallery/animation/basic_example.html

# We will now create a  video as shown in file "09_video.avi".

# First we set up a 4 frames. We will use an array "frames" to store 4 100x100
# images:
frames = np.zeros((100, 100, 4))

# Now we will set different image data for each frame
frames[:50, :50, 0] = 1
frames[:50, 50:, 1] = 1
frames[50:, 50:, 2] = 1
frames[50:, :50, 3] = 1

# Let's plot our frames
fig, ax = plt.subplots(2, 2)
ax[0, 0].imshow(frames[:, :, 0])
ax[0, 1].imshow(frames[:, :, 1])
ax[1, 1].imshow(frames[:, :, 2])
ax[1, 0].imshow(frames[:, :, 3])

# For matplotlib animations, we need the animation submodule
import matplotlib.animation as animation

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
imshow_handle = ax.imshow(frames[:, :, 0])
# Create an animation object using our update function. Create an animation
# with 100 frames:
video = animation.FuncAnimation(fig=fig, func=update_frame,
                                           frames=100,
                                           fargs=(frames, imshow_handle))
video.save(filename='09_video.avi', fps=5)
plt.close(fig)
del fig


###############################################################################
# Other modules for vision tasks
###############################################################################

#
# OpenCV: Fast image processing
#

# Advanced module specialized on fast image/videoframe processing. Setup
# might not be trivial but it's performance is great.
# https://docs.opencv.org/master/d6/d00/tutorial_py_root.html

#
# Datashader: Optimized large-scale plotting in Python
#

# Allows for fast large-scale plotting, such as scatter plots with millions
# of points or plotting pipelines.
# https://datashader.org/

#
# Videos/Animations
#

# Depending on the task it might be faster/more stable to use the ffmpeg
# package (not a Python package!), which supports fast video- and audio
# editing on many different OS:
# https://www.ffmpeg.org/

#
# Web-interfaces for visualization (like Shiny in R):
#

# https://bokeh.pydata.org/en/latest/
# https://plot.ly/products/dash/
