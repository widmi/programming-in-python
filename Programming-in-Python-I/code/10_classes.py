# -*- coding: utf-8 -*-
"""10_classes.py

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

In this file we will learn about object-oriented programming in Python.

"""

###############################################################################
# Classes
###############################################################################
# As we have seen with functions, the ability to effectively re-use code can
# benefit the program design. Object-oriented programming tries to achieve a
# higher level of reusability and modularity based on the idea of "objects".
# Such objects can hold data (referred to as "fields", "attributes", or
# "properties") and procedures/functions (referred to as "methods").

# In Python this concept is applied using "classes" to create a class of an
# object, e.g. the class "Dog", and "instances", which are an object instance
# from the class, e.g. an instance of "Dog" with name "bello" and another
# instance of "Dog" with name "barky".

# Classes are defined via the "class" statement. Class names should use
# CamelCase format. Instances of classes should use lower_case format.
# Tutorial: https://docs.python.org/3/tutorial/classes.html


#
# Creating a class and instances
#

# Let's create a class "Dog":
class Dog:
    def __init__(self, name):
        """The __init__ method will be executed at instantiation of the class.
        As all class-functions, it has to take the 'self' object at first
        argument. 'self' will give us access to the attributes and methods
        within the instance.
        """
        # Objects defined within the functions of the class are only
        # available in the function scope (like a normal variable in a
        # function)
        some_variable = 5  # this variable exists only within this function
        
        # ...but they can be made available by turning them into attributes.
        # The syntax for creating an attribute is self.varname = var:
        self.name = name  # Create an attribute "self.name"
        
        print(f"Created {self.name}")


# We can now create instances of our Dog class:
dog_1 = Dog(name='bello')  # This will create an instance of our class

# We can now access the instance and its attributes:
print(f"dog_1: {dog_1}")
print(f"dog_1.name: {dog_1.name}")

# We can create another instance of our Dog class:
dog_2 = Dog(name='barky')  # This will create an instance of our class

# We can now access the instance and its attributes:
print(f"dog_2: {dog_2}")
print(f"dog_2.name: {dog_2.name}")

# Note that dog_1 and dog_2 are not the same object. They are 2 different
# instances:
print(f"dog_1 is dog_2? answer: {dog_1 is dog_2}")


# We can also add methods to our class. For example, we can add a function
# "communicate()":
class Dog:
    def __init__(self, name):
        """The __init__ method will be executed at instantiation of the class.
        As all class-functions, it has to take the 'self' object at first
        argument. 'self' will give us access to the attributes and methods
        within the instance.
        """
        self.name = name  # Create an attribute "self.name"
    
    def communicate(self):
        """Here we defined another function available through the class
        instances. Note that the 'self' object has to be the first function-
        argument again. Functions of instances are called 'methods'. We can
        call the method later using the instance."""
        # Let's say we want the dog to bark:
        communication = "bark"
        # ... and we also want to include the name of the dog into the
        # communication. For this we can access the self.name attribute
        # that we created:
        communication = f"{self.name}: {communication}"
        return communication


# We can now again create instances of our new Dog class:
dog_1 = Dog(name='bello')  # This will create an instance of our class
dog_2 = Dog('barky')  # This will create another instance of our class

# We can now access the methods the same way we accessed the attributes:
print(dog_1.communicate())
print(dog_2.communicate())

#
# Inheritance
#

# Let's say we also want to add a class "Cat" and "Frog", that also have an
# attribute "name" and a method "communicate". However, we want to change what
# the communicate method does for "Cat" and "Frog". To do this, we can make use
# of "inheritance".


# First we define a base class that we call "Animal":
class Animal:
    def __init__(self, name):
        """The __init__ method will be executed at instantiation of the class.
        As all class-functions, it has to take the 'self' object at first
        argument. 'self' will give us access to the attributes and methods
        within the instance.
        """
        self.name = name  # Create an attribute "self.name"
    
    def communicate(self):
        """Returns the animal communication as string."""
        # We assume that the animal can make some noise:
        communication = "some noise"
        # ... and we again want to include the name of the animal into the
        # communication:
        communication = f"{self.name}: {communication}"
        return communication


# Then we can define a class "Dog" that inherits from the class "Animal". By
# doing this we will have a class "Dog" that contains the same attributes and
# methods as the base class "Animal" does. If we want to add or modify the
# methods/attributes we can simply overwrite them:
class Dog(Animal):
    # We do not need a new __init__ function since the __init__ function
    # from the "Animal" base class already does what we want it to do.
    
    # Here we overwrite the method "communicate" by a new version:
    def communicate(self):
        """Returns the dog communication as string."""
        # We assume that the dog barks:
        communication = "bark"
        communication = f"{self.name}: {communication}"
        return communication


# We can do the same for the classes "Cat" and "Frog":
class Cat(Animal):
    # We do not need a new __init__ function since the __init__ function
    # from the "Animal" base class already does what we want it to do.
    
    # Here we overwrite the method "communicate" by a new version:
    def communicate(self):
        """Returns the cat communication as string."""
        # We assume that the cat meows:
        communication = "meow"
        communication = f"{self.name}: {communication}"
        return communication


class Frog(Animal):
    # We do not need a new __init__ function since the __init__ function
    # from the "Animal" base class already does what we want it to do.
    
    # Here we overwrite the method "communicate" by a new version:
    def communicate(self):
        """Returns the frog communication as string."""
        # We assume that the frog ribbits:
        communication = "ribbit"
        communication = f"{self.name}: {communication}"
        return communication


# We can now again create and use instances of our classes Dog, Cat, and Frog
# that were all derived from the class Animal:
animal_1 = Dog(name='bello')
animal_2 = Cat(name='scratchy')
animal_3 = Frog(name='jumper')

print(animal_1.communicate())
print(animal_2.communicate())
print(animal_3.communicate())


# We could again derive a new class from our classes. Let's say we want to add
# a class SouthernLeopardFrog:
class SouthernLeopardFrog(Frog):
    # We do not need a new __init__ function since the __init__ function
    # from the "Animal" base class already does what we want it to do.
    
    # Here we overwrite the method "communicate" by a new version:
    def communicate(self):
        """Returns the frog communication as string."""
        # We assume that the southern leopard frog screeches:
        communication = "screech"
        communication = f"{self.name}: {communication}"
        return communication


animal_4 = SouthernLeopardFrog(name='froggy')
print(animal_4.communicate())


# We can also add new methods when deriving from a class:
class Bird(Animal):
    # We do not need a new __init__ function since the __init__ function
    # from the "Animal" base class already does what we want it to do.
    
    # Here we overwrite the method "communicate" by a new version:
    def communicate(self):
        """Returns the bird communication as string."""
        # We assume that the bird peeps:
        communication = "peep"
        communication = f"{self.name}: {communication}"
        return communication
    
    # Here we add a new method "fly":
    def fly(self, distance):
        return f"{self.name} flew {distance} km!"
    

animal_5 = Bird(name='peeps')
print(animal_5.communicate())
print(animal_5.fly(500))


#
# Some more details
#

class MyClass:
    # Variables defined here will be shared by all instances and are accessible
    # like attributes
    var = 55
    
    # Mutable objects are dangerous here, as we will see below
    weird_list = []
    
    def __init__(self, a):
        self.a = a
        
        # A method may call other methods
        self.class_function()
        
        # Class-private members can be indicated with underscores:
        self._do_not_change_ = 4  # reading allowed but don't change
        self.__keep_away__ = 3  # this is private, don't use outside
        
    def class_function(self):
        print(f"MyClass says: {self.a * self.var}")


# Create instances and note that __init__() calls the class_function() method:
instance1 = MyClass(a=1)
instance2 = MyClass(a=2)

# Variable var is available for all instances
print(f"instance1.var: {instance1.var}")
print(f"instance2.var: {instance2.var}")

# Overwriting the attribute "var" for instance1 only affects instance1:
instance1.var = 4
print(f"instance1.var: {instance1.var}")
print(f"instance2.var: {instance2.var}")

# Be careful - modifying elements of mutable objects (list, dictionaries)
# that are shared between instances will affect all instances:
print(f"instance1.weird_list: {instance1.weird_list}")
print(f"instance2.weird_list: {instance2.weird_list}")
instance1.weird_list.append('element')
print(f"instance1.weird_list: {instance1.weird_list}")
print(f"instance2.weird_list: {instance2.weird_list}")
instance3 = MyClass(a=3)
print(f"instance3.weird_list: {instance3.weird_list}")


#
# Modifying __init__ of inherited class:
#

class DerivedClass(MyClass):
    def __init__(self):
        a = 4
        # If we modify __init__(), we have to call __init__ of our parent class
        # at some point. The "super()" function helps us here since it will
        # return the class that we have derived the current class from:
        super(DerivedClass, self).__init__(a=a)  # calls MyClass.__init__(a=a)
    
    def class_function(self):
        """We could also change the function arguments BUT we have to make sure
        that it is still compatible with our __init__()!"""
        print(f"DerivedClass says: {self.a * self.var}")


# Create instances and note that __init__() calls the class_function() method:
instance1 = MyClass(a=1)
instance2 = DerivedClass()

print(f"instance1.class_function():")
instance1.class_function()
print(f"instance2.class_function():")
instance2.class_function()

#
# Recap: Directories as classes
#

# Adding a file "__init__.py" to a folder will make Python recognize the folder
# as "class" and you can import from it.
