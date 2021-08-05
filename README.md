# Programming in Python for Machine Learning
Hi there!
This is an interactive tutorial on how to program in [Python](www.python.org) with an additional focus on Machine Learning (ML).
I use it to teach Python for ML, i.a. at the [Artificial Intelligence study at the Kepler University Linz](https://www.jku.at/en/institute-for-machine-learning).

This course is suitable for complete beginners as well as veteran programmers.

## Why Python?
Python is one of the most commonly used Programming languages in machine learning and AI.
It is a powerful and versatile programming language that allows for fast prototyping in simple scripts up to complex software development.
These materials shall provide an introduction to programming and Python with a focus on Machine Learning.

## What awaits you here?
- A full tutorial on how to program in Python from the beginning (bits/bytes) to programming neural networks in PyTorch.
- Interactive materials for self-study.
- Small tasks for you to check your knowledge.
- Lots of examples and useful code snippets.

## Which areas are covered?
1. **Basics and setup**
   - Basics on hardware, software, programming, and datatypes
   - Python setup and installation
   - PyCharm editor and debugger
2. **Programming-in-Python-I**: General Python programming
   - Python basics
   - Advanced Python
   - Important Python modules
   - PyTorch preview
3. **Programming-in-Python-II**: Python and Machine Learning
   - Building ML projects in Python using PyTorch
   - Data collection, analysis, preprocessing
   - Neural network implementation, training, and evaluation
   - More details on Python classes
   - ML standards, hints, and good-practice

## Requirements
- No prior knowledge in programming is required. If you're already familiar with similar languages or Python, just skim over the first units.
- Laptop, PC, or access to a server is required. No high-end machines are required. GPU is optional (but faster and more fun).
- 64bit Python 3.6 or higher. (For installation instructions see slides.)
- Recommended operating system: Ubuntu 18.04 or higher. (optional)
- Recommended editor and debugger: PyCharm. (optional)

## Usage
- Complete materials GS1 to GS3 in folder `basics_setup/` if you are new to programming or need to set up your Python programming environment.
- Complete each unit in folder `Programming-in-Python-I/` and `Programming-in-Python-II/`. For each Unit `xx` (if existing):
   1. Go through the slides file `xx_slides.py` (=theoretical part and background).
   2. Step through the code file`xx_code.py` in the debugger (=practical part and main content). Observe the changes of the variables in the variable explorer of PyCharm. Feel free to play around with the code.
   3. Try to solve the tasks in `xx_tasks.py`.
   4. Check the example solutions for the tasks in `xx_solutions.py`.
- Assignment sheets, datasets collected by students, and access to ML challenge server are only available to enrolled students. (But you can use other image data for the ML project or skip the project part.)
   
## Contents
### Basics and setup
In folder [basics_setup/](basics_setup/)
- [GS1_Basics_of_Programming.pdf](basics_setup/GS1_Basics_of_Programming.pdf)
  - Basics on hardware, computer programs, bits and bytes
- [GS2_Installation_OS_Terminal.pdf](basics_setup/GS2_Installation_OS_Terminal.pdf)
  - Basics on operating system, terminal/console, Python installation and usage
- [GS3_Editor_Debugger.pdf](basics_setup/GS3_Editor_Debugger.pdf)
  - Setup and usage of PyCharm editor and debugger
### Programming in Python I
In folder [Programming-in-Python-I/](Programming-in-Python-I/)
- [00_comments_variables](Programming-in-Python-I/00_comments_variables)
  - Syntax, comments, and docstrings in Python, common Python variables, variable operations, and datatype conversions.
- [01_tuples_lists_indices_dictionaries_slices](Programming-in-Python-I/01_tuples_lists_indices_dictionaries_slices)
  - Tuples, lists, how to index them, how to create dictionaries, and how to use slices to retrieve multiple elements.
- [02_conditions_loops](Programming-in-Python-I/02_conditions_loops)
  - If, elif, and else conditions, for loops, while loops, and list comprehensions.
- [03_functions_print_input_modules](Programming-in-Python-I/03_functions_print_input_modules)
  - Functions, passing arguments to and returning values from functions, printing to and reading from the console, and how to import python modules.
- [04_exceptions](Programming-in-Python-I/04_exceptions)
  - How to raise and catch exceptions. (Error-handling in Python.)
- [05_files_glob](Programming-in-Python-I/05_files_glob)
  - How to open, close, and read from files.
  - Finding files in directories using the glob module.
- [06_os_sys_subprocess](Programming-in-Python-I/06_os_sys_subprocess)
  - How to use the os/sys modules to access OS
    operations and to get the arguments passed to our Python script using argparse.
  - How to start external programs in the background using
    the subprocessing module and how to call functions or external programs in a
    parallel fashion using the multiprocessing module.
  - Python as powerful
    alternative to shell-/bash-scripts to call and communicate with other programs.
- [07_regex](Programming-in-Python-I/07_regex)
  - How to use the re module to search for
    more complex patterns in strings via regular expressions ("regex").
- [08_numpy_pickle](Programming-in-Python-I/08_numpy_pickle)
  - How to perform fast (vector/matrix) calculations with NumPy.
  - How to save Python objects to files, e.g. via the pickle module, and how to save large data in hdf5 files via h5py.
- [09_matplotlib](Programming-in-Python-I/09_matplotlib)
  - How to create plots in Python using matplotlib.
- [10_classes](Programming-in-Python-I/10_classes)
  - Introduction to classes in Python.
- [11_decorators_numba](Programming-in-Python-I/11_decorators_numba)
  - How to speed up your Python code by using the numba
    package. We will also briefly learn about decorators in Python.
- [12_tensorflow_pytorch](Programming-in-Python-I/12_tensorflow_pytorch)
  - How to create computational graphs, speed up your Python code, utilize the GPU, and get ready for the basics of ML code with the  PyTorch and TensorFlow modules.
### Programming in Python II
In folder [Programming-in-Python-II/](Programming-in-Python-II/)
- [00_introduction](Programming-in-Python-II/00_introduction)
  - Introduction to the ML part and the example ML project. (no code)
- [01_project_design](Programming-in-Python-II/01_project_design)
  - How to tackle an ML project (quick guide). (no code)
- [02_resources](Programming-in-Python-II/02_resources)
  - Literature resources for ML. (no code)
- [03_git_hashing](Programming-in-Python-II/03_git_hashing)
  - Version control and cooperation using git.
    Hashing in Python.
- [04_data_analysis](Programming-in-Python-II/04_data_analysis)
  - Analysing and inspecting the dataset.
- [05_data_loading](Programming-in-Python-II/05_data_loading)
  - Fast loading of data using PyTorch and typical bottlenecks.
- [06_neural_network_inference](Programming-in-Python-II/06_neural_network_inference)
  - Implementation of neural networks in PyTorch.
  - Inference for neural networks in PyTorch.
- [07_neural_network_training](Programming-in-Python-II/07_neural_network_training)
  - Training neural networks in PyTorch.
- [08_data_augmentation](Programming-in-Python-II/08_data_augmentation)
  - Increasing your dataset size using data augmentation.
- [09_evaluation](Programming-in-Python-II/09_evaluation)
  - Evaluation of model performance.
- [10_torchscript](Programming-in-Python-II/10_torchscript)
  - Speeding up and deploying your PyTorch code using TorchScript.
- [example_project](Programming-in-Python-II/example_project)
  - Example ML project for image data.
    

-----------------

#### Best wishes and have fun!

*-- Michael Widrich (widi)*
