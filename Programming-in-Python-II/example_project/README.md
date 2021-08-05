# Example ML project
This project is an example ML project and contains a simple CNN 
that is trained to rotate an input image.
The input images used are CIFAR10 images, which have been converted to grayscale and
are rotated to obtain the targets.

In this first section you would put general information on the project, references, and
 an author list, if applicable.

### Example usage
It helps, especially for people new to ML, to include various example usage scenarios and
 an installation guide of your project, if applicable.

In our case the simple usage is:
```
python3 main.py working_config.json
```

### Structure
Having a tree with the files and folders is nice to get an overview.
However, this is sometimes tedious to maintain and omitted.
```
example_project
|- architectures.py
|    Classes and functions for network architectures
|- datasets.py
|    Dataset classes and dataset helper functions
|- main.py
|    Main file. In this case also includes training and evaluation routines.
|- README.md
|    A readme file containing info on project, example usage, authors, publication references, and dependencies.
|- utils.py
|    Utility functions and classes. In this case contains a plotting function.
|- working_config.json
|     An example configuration file. Can also be done via command line arguments to main.py.
```

### Dependencies
Dependencies are usually given as a list of packages + version numbers or as conda environments.
