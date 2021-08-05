# -*- coding: utf-8 -*-
"""04_data_analysis.py

Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.02.2020

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

In this file we will look into how to normalize and analyse our dataset. You
will need to download our dataset (or have access to a folder with similar
files and structure). Start by working on a small version of your dataset.
"""

###############################################################################
# Our dataset
###############################################################################
# Our dataset consists of grayscale images stored as .jpg files. The files are
# organized in folders, each folder representing a student and their submitted
# files.
# The files were already converted to gray-scale. Would you be able to
# convert an image stored as numpy array into a gray-scale image? -> Task 01

# Here you have to set the path to your dataset
# ("04_dataset_preview.zip" contains a subset of the data)
input_path = "04_dataset_preview"

#
# Let's start by taking a look at our data by reading in one file
#

# Some modules we will need. We will use numpy and pillow (PIL) for the images.
import os
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

# Get list of all .jpg files
image_files = sorted(glob.glob(os.path.join(input_path, '**', '*.jpg'),
                               recursive=True))

# Check number of found files
print(f"Found {len(image_files)} image files")
# image_files = image_files[:100]  # Working on a slow machine? Use a subset :)

# Read first image file
image = Image.open(image_files[0])  # This returns a PIL image
image = np.array(image)  # We can convert it to a numpy array

print(f"image data:\n"
      f"shape: {image.shape}; min: {image.min()}; "
      f"max: {image.max()}; dtype: {image.dtype}")

# We are dealing with image data, so each sample is high-dimensional and
# contains as many features as it has pixels.
# Pixel values range from 0 to 255 in 3 color channels for RGB and in 1 channel
# for gray-scale images.

#
# Check means and standard deviations of images
#

# We know how many images to expect -> we can already allocate numpy arrays to
# store mean and std values and the folder_names of the folders as float values
means = np.zeros(shape=(len(image_files), ))
stds = np.zeros(shape=(len(image_files), ))
folder_names = []

# Loop through files, read them, and store mean, std, and folder folder_name
for i, image_file in tqdm(enumerate(image_files), desc="Processing files",
                          total=len(image_files)):
    image = np.array(Image.open(image_file))
    means[i] = image.mean()
    stds[i] = image.std()
    folder_names.append(os.path.basename(os.path.dirname(image_file)))
folder_names = np.array(folder_names)

# It's a good idea to create save-points if computation takes a while. Here
# we save the means, stds, and folder_names that we computed. You may also
# load the pre-computed data in file "04_means_stds_precomputed.pklz" instead.
import dill as pkl
import gzip

# Save our data in a compressed pickle file
with gzip.open(f"04_means_stds.pklz", 'wb') as fh:
    pkl.dump(dict(means=means, stds=stds, folder_names=folder_names), file=fh)
    
# Load pre-computed data from the compressed pickle file
with gzip.open(f"04_means_stds_precomputed.pklz", 'rb') as fh:
    load_dict = pkl.load(fh)
means = load_dict['means']
stds = load_dict['stds']
folder_names = load_dict['folder_names']

# Now we want to visualize our data. We will use a pyplot 2D scatter-plot.
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
_, point_colors = np.unique(folder_names, return_inverse=True)
point_colors = np.asarray(point_colors, dtype=np.float)
point_colors = point_colors / point_colors.max()  # Use folder names as colors
ax.scatter(x=means, y=stds, c=point_colors, s=0.1, cmap='nipy_spectral')
ax.set_xlabel('mean')
ax.set_ylabel('standard deviation')
ax.grid(True)
fig.savefig('mean_std.png', dpi=250)
fig.savefig('mean_std_large.png', dpi=500)
fig.savefig('mean_std.svg')

# What's up with the cluster at (mean=164, std=78)? -> Task 02


###############################################################################
# Clustering our data
###############################################################################
# We want to check for clusters in our data. We want to use t-SNE or UMAP but
# our raw image data is not suited for this method. Therefore, we will use a
# pre-trained CNN to downproject the images into a better suited feature-space
# before applying our clustering method.

#
# Projecting our images to a CNN feature space
#

# PyTorch gives us convenient access to datasets and pretrained models via
# torchvision. The original models and data might not be hosted by PyTorch
# itself -> they might change, so always keep a local copy if you want
# reproducibility.

import torch
import torchvision
device = 'cuda:0'  # Set this to 'cpu' if you are using CPU only

# Possible issues: incompatible versions (e.g. pillow version)
# https://github.com/pytorch/vision/issues/1712

# We can choose from a variety of models:
# https://pytorch.org/docs/stable/torchvision/models.html
# We want some model that has few features before its output layer and was
# trained on targets that are related to our task.

# Get SqueezeNet 1.1 model, pretrained on ImageNet
pretrained_model = torchvision.models.squeezenet1_1(pretrained=True,
                                                    progress=True)
# Send pretrained_model to GPU (or keep on CPU if device='cpu')
pretrained_model = pretrained_model.to(device=device)

# What does the model expect our data to look like?
# PyTorch docs: All pre-trained models expect input images normalized in the
# same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
# where H and W are expected to be at least 224. The images have to be loaded
# in to a range of [0, 1] and then normalized using
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
# -> We need to normalize our data mean and std
norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32,
                         device=device)
norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32,
                        device=device)

# Again, we can allocate the numpy array before-hand: We know how many images
# to expect and we can look up that the layer before the output layer has
# 512 features in SqueezeNet 1.1.
n_model_features = 512
images_projected_cnn = np.zeros(shape=(len(image_files), n_model_features),
                                dtype=np.float)

# Loop through image files and extract CNN features (we will later learn how
# to do this using optimized PyTorch tools)
with torch.no_grad():  # We do not need to store gradients for this
    for i, image_file in tqdm(enumerate(image_files), desc="Processing files",
                              total=len(image_files)):
        # Open image file, convert to numpy array with values in range [0, 1]
        image = np.array(Image.open(image_file), dtype=np.float32)
        image[:] /= 255.
        # Convert numpy array to tensor
        image = torch.tensor(image, dtype=torch.float32, device=device)
        
        # Our pretrained_model expects color images of shape [3, H, W] -> we
        # can stack the gray-scale image 3 times to simulate 3 color-channels
        image = torch.stack([image] * 3, dim=0)
        
        # Perform normalization for each channel
        image = (image - norm_mean[:, None, None]) / norm_std[:, None, None]
        
        # Now we can apply the pretrained model
        image_features = pretrained_model.features.forward(image[None])
        # ... if the image is large, we might end up with shape [512, H, W], so
        # we will take the maximum over the H and W dimensions
        image_features = torch.mean(torch.mean(image_features, dim=-1), dim=-1)
        # Finally, we can store the computed CNN features
        images_projected_cnn[i, :] = image_features.cpu()

#
# Applying t-SNE
#

# sklearn provides us with a nice t-SNE implementation and good documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
# https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html#sphx-glr-auto-examples-manifold-plot-t-sne-perplexity-py
from sklearn import manifold

# Get t-SNE model
tsne = manifold.TSNE(random_state=1)

# Fit (=train) model on our data
images_projected_tsne = tsne.fit_transform(images_projected_cnn)
print(f"t-SNE projected our data to shape {images_projected_tsne.shape}")

# Save our results in case we want to use them later
np.savez_compressed('04_clustering', images_projected_tsne=images_projected_tsne,
                    images_projected_cnn=images_projected_cnn,
                    folder_names=folder_names, file_names=image_files)

# Load pre-computed data from the compressed numpy file
loaded = np.load("04_clustering_precomputed.npz")
images_projected_tsne = loaded['images_projected_tsne']
images_projected_cnn = loaded['images_projected_cnn']
folder_names = loaded['folder_names']
file_names = loaded['file_names']

# Plot the result
fig, ax = plt.subplots()
ax.scatter(x=images_projected_tsne[:, 0], y=images_projected_tsne[:, 1],
           c=point_colors, s=0.1, cmap='nipy_spectral')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
fig.savefig('tsne.png', dpi=250)
fig.savefig('tsne_large.png', dpi=500)
fig.savefig('tsne.svg')

#
# Inspecting the clustered images
#

# t-SNE and UMAP are good for visualization but we cannot blindly trust them.
# To inspect our clusters and check what they mean/if they make sense, we
# have to take a look at the clustered images. This can give you important
# information about the properties of your dataset.

# Task 03: Use UMAP to cluster the data
# Task 04: Visually inspect the clustered images
