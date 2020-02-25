# -*- coding: utf-8 -*-
"""12_solutions.py

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

Example solutions for tasks in file 12_tasks.py.

"""

###############################################################################
# 12 PyTorch
###############################################################################

#
# Task 1
#

# Use PyTorch to compute the derivative of "e" in
# e = a * d + b * d ** 2. + c * d ** 3.
# w.r.t. input variable "a" and "d" for values of
a = 2.
b = -3.
c = 4.
d = -5.

# Your code here #
import torch

a = torch.tensor([a], requires_grad=True)
b = torch.tensor([b], requires_grad=False)
c = torch.tensor([c], requires_grad=False)
d = torch.tensor([d], requires_grad=True)

e = a * d + b * d ** 2. + c * d ** 3.

e.backward()
print(f"a.grad = {a.grad}")
print(f"d.grad = {d.grad}")


#
# Task 2
#

# Use PyTorch to compute the derivative of "e" in
# e = sum(a * d + b * d ** 2. + c * d ** 3.)
# w.r.t. input variable "a" and "d". The input variables are arrays and the
# computation of the formula should be done element-wise, resulting in an
# output array of shape 100, which should then be summed up, resulting in a
# scalar.
import numpy as np
a = np.linspace(-5., 5., num=100)
b = np.linspace(0., 1., num=100) ** 2
c = np.ones(shape=(100,), dtype=np.float)
d = np.linspace(5., -5., num=100)

# Your code here #
import torch

a = torch.tensor([a], requires_grad=True)
b = torch.tensor([b], requires_grad=False)
c = torch.tensor([c], requires_grad=False)
d = torch.tensor([d], requires_grad=True)

e = torch.sum(a * d + b * d ** 2. + c * d ** 3.)

e.backward()
print(f"a.grad = {a.grad}")
print(f"d.grad = {d.grad}")


#
# Task 3
#

# Perform Task 2 on the GPU. Keep in mind that you have to convert the values
# to torch.float32 .
import numpy as np
a = np.linspace(-5., 5., num=100)
b = np.linspace(0., 1., num=100) ** 2
c = np.ones(shape=(100,), dtype=np.float)
d = np.linspace(5., -5., num=100)

# Your code here #
import torch

a = torch.tensor([a], requires_grad=True, dtype=torch.float32, device="cuda:0")
b = torch.tensor([b], requires_grad=False, dtype=torch.float32, device="cuda:0")
c = torch.tensor([c], requires_grad=False, dtype=torch.float32, device="cuda:0")
d = torch.tensor([d], requires_grad=True, dtype=torch.float32, device="cuda:0")

e = torch.sum(a * d + b * d ** 2. + c * d ** 3.)

e.backward()
print(f"a.grad = {a.grad}")
print(f"d.grad = {d.grad}")
