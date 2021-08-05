# -*- coding: utf-8 -*-
"""10_tasks.py
Author -- Van Phuong Huynh and Michael Widrich
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

Tasks for self-study. Try to solve these tasks on your own and
compare your solution to the solution given in the file 10_tasks.py.
See 10_torchscript.py for more information on the tasks.

"""

# Know that the below data (x, y) can be fitted with a second order polynomial
# Complete the below requirements (Task 1, 2, 3, 4)  to find a fitting model for the data

import torch
from torch import Tensor

# Training Data
x = torch.linspace(-3, 1, 4000)
y = 1 + 2 * x + x ** 2


class Quadratic(torch.nn.Module):
    def __init__(self):
        super(Quadratic, self).__init__()
        # Task 01: Define trainable parameters of the model (coefficients of the polynomial)
        # add your code here

    def forward(self, x: Tensor):
        # Task 02: Define calculation for the forward pass
        # add your code here
        pass

    def print(self):
        return f'y = {self.a.item()} + {self.b.item()}x + {self.c.item()}x^2'


def quadratic_fit(quadratic: Quadratic, x: Tensor, y: Tensor, epoch: int):
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(quadratic.parameters(), lr=1e-6)
    for t in range(epoch):
        # Task 03: Define training method
        # add your code here
        pass

# Task 04:
# 1.) Train a model to fit the data
# 2.) Create a TorchScript model from the trained model
# 3.) Save the TorchScript model to a file, then load it
# 4.) Print and inspect the loaded model and it's TorchScript code
# Add your code here
