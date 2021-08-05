# -*- coding: utf-8 -*-
"""08_tasks.py

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

Example solutions for tasks in file 07_tasks.py.

"""

#
# Task 01
#

# To make dropout on image data more effective, we can also drop out blocks of
# pixels that are adjacent in the spatial and channel dimensions (see DropBlock
# https://arxiv.org/abs/1810.12890) or replace them with random values (see
# Random Erasing https://arxiv.org/abs/1708.04896).
# Create a DropBlock PyTorch Module that processes an input tensor


class DropBlock(torch.nn.Module):
    def __init__(self, p: float, n: int):
        """DropBlock module that will drop out a block of `n` by `n` pixels
        with probability `p`"""
        super(DropBlock, self).__init__()
        self.n = n
        self.dropout = torch.nn.Dropout(p=p)
    
    def forward(self, input_tensor):
        image_shape = list(input_tensor.shape)
        image_shape[-3] = 1
        image_shape[-2] = int(image_shape[-2] / self.n)
        image_shape[-1] = int(image_shape[-1] / self.n)
        
        dropout_mask = torch.ones(size=image_shape, dtype=input_tensor.dtype,
                                  device=input_tensor.device)
        dropout_mask = self.dropout(dropout_mask)
        dropout_mask = dropout_mask.repeat_interleave(dim=-2, repeats=self.n)
        dropout_mask = dropout_mask.repeat_interleave(dim=-1, repeats=self.n)
        input_dropped = (input_tensor[..., :dropout_mask.shape[-2],
                         :dropout_mask.shape[-1]] * dropout_mask)
        return input_dropped

fig, axes = plt.subplots(1, 4)
axes[0].imshow(image.cpu().numpy()[0].transpose(1, 2, 0))
axes[0].set_xticks([], [])  # Remove xaxis ticks
axes[0].set_yticks([], [])  # Remove yaxis ticks
axes[0].set_title('Original image')
for i, n in enumerate([2, 8, 32]):
    simple_dropout = DropBlock(p=0.1, n=n)
    image_dropout = simple_dropout(image)
    axes[i+1].imshow(image_dropout.cpu().numpy()[0].transpose(1, 2, 0))
    axes[i+1].set_xticks([], [])  # Remove xaxis ticks
    axes[i+1].set_yticks([], [])  # Remove yaxis ticks
    axes[i+1].set_title(f'DropBlock dropout\n(p: {p}, n: {n})')
fig.tight_layout()
fig.savefig("08_dropout_dropblock.png", dpi=500)