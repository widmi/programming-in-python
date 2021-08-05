# -*- coding: utf-8 -*-
"""08_data_augmentation.py

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

In this file we will see how to apply some basic data augmentation methods
using PyTorch. We will then use torchvision "transforms" to chain multiple
image transformation methods in one pipeline.
"""

import os
output_folder = "08_output_folder"
os.makedirs(output_folder, exist_ok=True)

#
# Here we prepare an example input image as PyTorch tensor
#
import numpy as np
np.random.seed(0)  # Set a known random seed for reproducibility
from PIL import Image
import torch
torch.random.manual_seed(0)  # Set a known random seed for reproducibility
import torchvision
from matplotlib import pyplot as plt

# Converter: PIL -> tensor
pil_to_tensor = torchvision.transforms.functional.to_tensor
# Converter: tensor -> PIL
tensor_to_pil = torchvision.transforms.functional.to_pil_image

image = Image.open("08_example_image.jpg")  # Read a PIL image
image = pil_to_tensor(image)  # Shape (C, H, W)
print(f"Image shape: {image.shape}")


###############################################################################
# Additive input noise
###############################################################################
# If we add noise to our input data, we have to consider possible changes in
# the input distribution. By using mean=0 for our noise, we will at least not
# change the mean of the feature values. The standard deviation will be a
# hyperparameter we will have to figure out through a hyperparameter search.
# Let's create a function that adds noise from a normal distribution to our
# tensor (not restricted to image data):


def add_normal_noise(input_tensor, mean: int = 0, std: float = 0.5):
    """Simple function that adds noise from a normal distribution to
    `input_tensor`"""
    # Create the tensor containing the noise
    noise_tensor = torch.empty_like(input_tensor)
    noise_tensor.normal_(mean=mean, std=std)
    # Add noise to input tensor and return results
    return input_tensor + noise_tensor


# Let's apply our function and check the result for different noise stds
fig, axes = plt.subplots(1, 4)
axes[0].imshow(tensor_to_pil(image))
axes[0].set_xticks([], [])  # Remove xaxis ticks
axes[0].set_yticks([], [])  # Remove yaxis ticks
axes[0].set_title('Original image')
for i, std in enumerate([0.1, 0.5, 1.0]):
    image_noisy = add_normal_noise(image, std=std)
    # Make sure image values are in valid range
    image_noisy = image_noisy.clamp(min=0, max=1)
    axes[i+1].imshow(tensor_to_pil(image_noisy))
    axes[i+1].set_xticks([], [])  # Remove xaxis ticks
    axes[i+1].set_yticks([], [])  # Remove yaxis ticks
    axes[i+1].set_title(f'Noise (std: {std})')
fig.tight_layout()
fig.savefig(os.path.join(output_folder, "image_noise.png"), dpi=500)


###############################################################################
# Input Dropout
###############################################################################
# We can use the the different versions of dropout modules in PyTorch.

# The simples dropout version torch.nn.Dropout is dropping out random values.
# Note that this will dropout random values of a tensor (not restricted to
# image data), independent of the channels or spatial dimensions. The remaining
# values will be re-scaled, to keep the distribution of pixel values closer to
# the original distribution.
fig, axes = plt.subplots(1, 4)
axes[0].imshow(tensor_to_pil(image))
axes[0].set_xticks([], [])  # Remove xaxis ticks
axes[0].set_yticks([], [])  # Remove yaxis ticks
axes[0].set_title('Original image')
for i, p in enumerate([0.1, 0.5, 0.7]):
    simple_dropout = torch.nn.Dropout(p=p)
    image_dropout = simple_dropout(image)
    # Make sure image values are in valid range
    image_dropout = image_dropout.clamp(min=0, max=1)
    axes[i+1].imshow(tensor_to_pil(image_dropout))
    axes[i+1].set_xticks([], [])  # Remove xaxis ticks
    axes[i+1].set_yticks([], [])  # Remove yaxis ticks
    axes[i+1].set_title(f'Simple dropout\n(p: {p})')
fig.tight_layout()
fig.savefig(os.path.join(output_folder, "simple_dropout.png"), dpi=500)


###############################################################################
# Chaining image transformations with torchvision
###############################################################################
# When dealing with image data, torchvision can be used to combine different
# image transformation methods ("transforms") using
# `torchvision.transforms.Compose`. This class allows us to chain multiple
# image transformations, some of them working on PIL images, others working on
# PyTorch tensors.
#
# Concept:
# 1. Create list of transforms instances
# 2. Use list to create `torchvision.transforms.Compose` instance
# 3. Apply `torchvision.transforms.Compose` instance to image (e.g. in
# `torch.utils.data.Dataset`)
#
# Below we will see examples on how to combine transforms.
# Documentation: https://pytorch.org/docs/stable/torchvision/transforms.html

#
# Example: PIL -> tensor
#
from torchvision import transforms
# Create chain of transforms (only 1 transform in this case)
transform_chain = transforms.Compose([
    transforms.ToTensor()  # Transform a PIL or numpy array to a tensor
])
# Apply image transformations
image = Image.open("08_example_image.jpg")  # Read a PIL image
transformed_image = transform_chain(image)  # Apply transforms chain
print(f"Transformed image dtype: {transformed_image.dtype}")
print(f"Transformed image min/max: "
      f"{transformed_image.min()}/{transformed_image.max()}")
print(f"Transformed image shape: {transformed_image.shape}")

# Note: ToTensor() automatically converts the uint8 pixel values of a numpy
# array or PIL image with values in range [0, 255] to a tensor of torch.float,
# range [0.0, 1.0], and shape (C x H x W).

#
# Example: PIL -> RandomRotation -> tensor
#
# Create chain of transforms
transform_chain = transforms.Compose([
    transforms.RandomRotation(degrees=180),  # Rotate in range (-180, 180)
    transforms.ToTensor()  # Transform a PIL or numpy array to a tensor
])
# Apply image transformations
image = Image.open("08_example_image.jpg")  # Read a PIL image
transformed_image = transform_chain(image)  # Apply transforms chain

fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[0].set_xticks([], [])  # Remove xaxis ticks
axes[0].set_yticks([], [])  # Remove yaxis ticks
axes[0].set_title('Original image')
axes[1].imshow(tensor_to_pil(transformed_image))
axes[1].set_xticks([], [])  # Remove xaxis ticks
axes[1].set_yticks([], [])  # Remove yaxis ticks
axes[1].set_title(f'Random rotation')
fig.tight_layout()
fig.savefig(os.path.join(output_folder, "random_rotation.png"), dpi=500)


#
# Custom transformations
# We can define custom transformations using `torchvision.transforms.Lambda`
#
def wrap_add_normal_noise(mean: int = 0, std: float = 0.5):
    """Return function that calls add_normal_noise() and clamps values to
    range [0, 1]"""
    def noisy_image(input_tensor):
        input_tensor = add_normal_noise(input_tensor, mean, std)
        return torch.clamp(input_tensor, min=0, max=1)
    
    return noisy_image


noise_transform = transforms.Lambda(lambd=wrap_add_normal_noise(std=0.1))
# -> we can now use noise_transform in our transforms chain

#
# Example: PIL -> Resize -> ColorJitter -> RandomRotation -> RandomVerticalFlip
#          -> RandomHorizontalFlip -> tensor -> noise -> RandomErasing
#
# Create chain of transforms
transform_chain = transforms.Compose([
    transforms.Resize(size=100),  # Resize image to minimum edge=100 px
    transforms.ColorJitter(),
    transforms.RandomRotation(degrees=180),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    noise_transform,
    transforms.RandomErasing()
])
# Apply image transformations
image = Image.open("08_example_image.jpg")  # Read a PIL image
transformed_image = transform_chain(image)  # Apply transforms chain

fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[0].set_xticks([], [])  # Remove xaxis ticks
axes[0].set_yticks([], [])  # Remove yaxis ticks
axes[0].set_title('Original image')
axes[1].imshow(tensor_to_pil(transformed_image))
axes[1].set_xticks([], [])  # Remove xaxis ticks
axes[1].set_yticks([], [])  # Remove yaxis ticks
axes[1].set_title(f'Random rotation')
fig.tight_layout()
fig.savefig(os.path.join(output_folder, "chained_transforms.png"), dpi=500)

#
# Notes
#
# - Some transforms work on PIL images, others on tensors -> keep order and
#   position of transformation in mind
# - You might need 2 transform chains, one for training, and one for evaluation
#


###############################################################################
# torchvision: Functional transforms
###############################################################################
# We can access the transformation functions of the torchvision transform
# classes directly (without randomness). To do this, we use the "functionals"
# of the transforms in `torchvision.transforms.functional`.
# Examples:

import torchvision.transforms.functional as TF
image = Image.open("08_example_image.jpg")  # Read a PIL image
hflipped = TF.hflip(image)  # Horizontal flip
vflipped = TF.vflip(image)  # Vertical flip
cropped_resized = TF.resized_crop(image, i=10, j=200, h=3000, w=2000,
                                  size=1000)  # Crop and resize
rotated = TF.rotate(image, angle=70)  # Crop and resize

fig, axes = plt.subplots(2, 3)
axes[0, 0].imshow(image)
axes[0, 0].set_xticks([], [])  # Remove xaxis ticks
axes[0, 0].set_yticks([], [])  # Remove yaxis ticks
axes[0, 0].set_title('Original image')
axes[1, 0].imshow(hflipped)
axes[1, 0].set_xticks([], [])  # Remove xaxis ticks
axes[1, 0].set_yticks([], [])  # Remove yaxis ticks
axes[1, 0].set_title(f'h. flip')
axes[0, 1].imshow(vflipped)
axes[0, 1].set_xticks([], [])  # Remove xaxis ticks
axes[0, 1].set_yticks([], [])  # Remove yaxis ticks
axes[0, 1].set_title(f'v. flip')
axes[0, 2].remove()
axes[1, 1].imshow(cropped_resized)
axes[1, 1].set_xticks([], [])  # Remove xaxis ticks
axes[1, 1].set_yticks([], [])  # Remove yaxis ticks
axes[1, 1].set_title(f'cropped and resized')
axes[1, 2].imshow(rotated)
axes[1, 2].set_xticks([], [])  # Remove xaxis ticks
axes[1, 2].set_yticks([], [])  # Remove yaxis ticks
axes[1, 2].set_title(f'rotated')
fig.tight_layout()
fig.savefig(os.path.join(output_folder, "transformations.png"), dpi=500)


###############################################################################
# Combining PyTorch Dataset and transforms
###############################################################################
# PyTorch Dataset and torchvision transforms can be freely combined. I
# recommend 2 options, where both options perform the transformations in the
# __get_item__() method of the PyTorch Dataset and can thereby be performed via
# multiprocessing in combination with the PyTorch DataLoader.
from torch.utils.data import Dataset, Subset, DataLoader

#
# Option 1: Adding the transforms to the __get_item__() method of the Dataset
# This option is for the case where we want to apply transforms always, for
# training and evaluation. E.g. converting images to grayscale or resizing
# them.
#


class SimpleRandomImageDataset(Dataset):
    def __init__(self, transforms: transforms.Compose = None):
        """Create random PIL images and optionally process them with
        torchvision transforms.

        Parameters
        -------------
        transforms : torchvision.transforms.Compose
            Optional: Chain of torchvision transforms to process image data
        """
        self.transforms = transforms
        self.np_image_shape = (5, 7, 3)  # H, W, C
        self.n_samples = 1000
    
    def __make_image__(self, rnd_seed: int):
        """Create random PIL image"""
        rnd_gen = np.random.RandomState(rnd_seed)
        rnd_image = rnd_gen.randint(size=self.np_image_shape, low=0, high=256,
                                    dtype=np.uint8)
        rnd_image = Image.fromarray(rnd_image)
        return rnd_image
    
    def __getitem__(self, index: int):
        """Creates random image-like array and processes it with torchvision
        transforms.

        Parameters
        -------------
        index : int
            Sample index
    
        Returns
        -------------
        image
            Returns random image after transforms
        """
        # Create some random image
        rnd_image = self.__make_image__(rnd_seed=index)
        # Apply transforms
        image = self.transforms(rnd_image)
        # Return transformed image
        return image

    def __len__(self):
        """Return number of samples"""
        return self.n_samples


# Create the dataset and specify some transforms to apply:
sri_dataset = SimpleRandomImageDataset(
        transforms=transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ]))
# Split into training, validation, and test set
sri_trainingset = Subset(sri_dataset,
                         indices=range(int(len(sri_dataset) * 3/5)))
sri_validationset = Subset(sri_dataset,
                           indices=range(int(len(sri_dataset) * 3/5),
                                         int(len(sri_dataset) * 4/5)))
sri_testset = Subset(sri_dataset,
                     indices=range(int(len(sri_dataset) * 4/5),
                                   len(sri_dataset)))

# Create DataLoaders for the 3 dataset splits
sri_trainingloader = DataLoader(sri_trainingset, batch_size=1, num_workers=0)
sri_validationset = DataLoader(sri_validationset, batch_size=1, num_workers=0)
sri_testset = DataLoader(sri_testset, batch_size=1, num_workers=0)

# We can now use the DataLoader to load the samples. The transforms are applied
# to all 3 dataset splits:

for training_mb, validation_mb, test_mb in zip(sri_trainingloader,
                                               sri_validationset, sri_testset):
    # Plot the first minibatch of the dataset splits as example, then exit loop
    print(f"Training mb: {training_mb}")
    print(f"Validation mb: {validation_mb}")
    print(f"Test mb: {test_mb}")
    break


#
# Option 2: Adding the tranforms to the __get_item__() method of a new Dataset
# Often we want to apply different transforms to training and evaluation data.
# We can do this by creating a dedicated Dataset class that wraps a Dataset and
# only applies the transforms.
#


# First we create a Dataset that returns samples without transforms:
class SimpleRandomImageDataset(Dataset):
    def __init__(self):
        """Create random PIL images and return them as numpy array"""
        self.np_image_shape = (5, 7, 3)  # H, W, C
        self.n_samples = 1000
    
    def __make_image__(self, rnd_seed: int):
        """Create random image data"""
        rnd_gen = np.random.RandomState(rnd_seed)
        rnd_image = rnd_gen.randint(size=self.np_image_shape, low=0, high=256,
                                    dtype=np.uint8)
        rnd_image = Image.fromarray(rnd_image)
        return rnd_image
    
    def __getitem__(self, index: int):
        """Creates random PIL image

        Parameters
        -------------
        index : int
            Sample index
    
        Returns
        -------------
        image
            Returns random PIL image
        """
        # Create some random image
        rnd_image = self.__make_image__(rnd_seed=index)
        # Return image and sample index
        return rnd_image, index
    
    def __len__(self):
        """Return number of samples"""
        return self.n_samples


# The we create a Dataset that applies transforms to the output of the other
# Dataset:
class TransformsDataset(Dataset):
    def __init__(self, dataset: Dataset,
                 transforms: transforms.Compose = None):
        """Apply transforms to first object in sample tuple of PyTorch Dataset

        Parameters
        -------------
        transforms : torchvision.transforms.Compose
            Optional: Chain of torchvision transforms to process image data
        """
        self.dataset = dataset
        self.transforms = transforms
    
    def __getitem__(self, index: int):
        """Get sample tuple from specified PyTorch Dataset and apply transfomrs
         to first object in minibatch

        Parameters
        -------------
        index : int
            Sample index
    
        Returns
        -------------
        image
            Returns random image after transforms
        """
        # Get sample as tuple and convert to list
        sample = list(self.dataset.__getitem__(index))
        # Apply transforms to first object in sample
        sample[0] = self.transforms(sample[0])
        # Return sample tuple
        return tuple(sample)
    
    def __len__(self):
        """Return number of samples"""
        return len(self.dataset)


# We can now create our Dataset and split it into our 3 sets:
sri_dataset = SimpleRandomImageDataset()
# Split into training, validation, and test set
sri_trainingset = Subset(sri_dataset,
                         indices=range(int(len(sri_dataset) * 3/5)))
sri_validationset = Subset(sri_dataset,
                           indices=range(int(len(sri_dataset) * 3/5),
                                         int(len(sri_dataset) * 4/5)))
sri_testset = Subset(sri_dataset,
                     indices=range(int(len(sri_dataset) * 4/5),
                                   len(sri_dataset)))

# Now we can apply transforms specific to the Dataset splits:
training_transforms = transforms.Compose([
    transforms.Resize(4),
    transforms.Grayscale(),
    transforms.RandomRotation(degrees=180),
    transforms.ToTensor()
])
evaluation_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])
# Wrap trainingset in TransformsDataset to apply training_transforms
sri_trainingset = TransformsDataset(dataset=sri_trainingset,
                                    transforms=training_transforms)
# Wrap evaluation sets in TransformsDataset to apply evaluation_transforms
sri_validationset = TransformsDataset(dataset=sri_validationset,
                                      transforms=evaluation_transforms)
sri_testset = TransformsDataset(dataset=sri_testset,
                                transforms=evaluation_transforms)

# Create DataLoaders for the 3 dataset splits
sri_trainingloader = DataLoader(sri_trainingset, batch_size=1, num_workers=0)
sri_validationset = DataLoader(sri_validationset, batch_size=1, num_workers=0)
sri_testset = DataLoader(sri_testset, batch_size=1, num_workers=0)

# We can now use the DataLoader to load the samples. The transforms are applied
# to all 3 dataset splits:
for training_mb, validation_mb, test_mb in zip(sri_trainingloader,
                                               sri_validationset, sri_testset):
    # Plot the first minibatch of the dataset splits as example, then exit loop
    print(f"Training mb: {training_mb}")
    print(f"Validation mb: {validation_mb}")
    print(f"Test mb: {test_mb}")
    break

# Notice how the image tensor in the training_mb has a different shape since it
# went through different transforms than the validation_mb and test_mb.
