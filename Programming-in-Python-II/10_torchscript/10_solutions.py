# -*- coding: utf-8 -*-
"""10_solutions.py
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

Example solutions for tasks in file 10_tasks.py.

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
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))

    def forward(self, x: Tensor):
        # Task 02: Define calculation for the forward pass
        # add your code here
        return self.a + self.b * x + self.c * x ** 2

    def print(self):
        return f'y = {self.a.item()} + {self.b.item()}x + {self.c.item()}x^2'


def quadratic_fit(quadratic: Quadratic, x: Tensor, y: Tensor, epoch: int):
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(quadratic.parameters(), lr=1e-6)
    for t in range(epoch):
        # Task 03: Define training method
        # add your code here
        # Forward pass: Compute predicted y by passing x to the model
        pred_y = model(x)

        # calculate and print loss
        loss = criterion(pred_y, y)
        if t % 100 == 99:
            print(t, loss.item())

        # reset old gradients in the previous backward pass
        optimizer.zero_grad()
        # do a backward pass for gradients of the loss w.r.t a, b, c.
        loss.backward()
        # update parameters a, b, c with the corresponding gradients.
        optimizer.step()


# Task 04:
# 1.) Train a model to fit the data
# 2.) Create a TorchScript model from the trained model
# 3.) Save the TorchScript model to a file, then load it
# 4.) Print and inspect the loaded model and it's TorchScript code
# Add your code here

model = Quadratic()
epoch = 5000
quadratic_fit(model, x, y, epoch)

scripted_model = torch.jit.script(model)
scripted_model.save('quadratic.pt')
loaded_model = torch.jit.load('quadratic.pt')

print(f'\nLoaded Model: {model.print()}\n')
print(f'loaded_model.code:\n {loaded_model.code}\n')
