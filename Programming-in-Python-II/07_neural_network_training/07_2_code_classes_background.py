# -*- coding: utf-8 -*-
"""07_2_classes_background.py

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

In this file we will take a peak behind the curtains of how object attributes
are organized and how we can use descriptors to manipulate how attributes are
accessed.
"""

###############################################################################
# PyTorch Tensors - Recap
###############################################################################

# As we saw in Programming in Python I, Unit 12, PyTorch creates a dynamic
# computational graph from our Python code. The basis for this is the
# PyTorch Tensor class torch.Tensor(). Using operations on such tensors adds
# the operations to a computational graph in the background, which is evaluated
# once we actually access the result. This has many advantages, such as
# automatic optimization for different hardware types and different data types,
# automatic computation of gradients, and so on.

import torch
a = torch.tensor([5.], requires_grad=True)
b = torch.tensor([4.], requires_grad=True)
c = a * b  # Point "c" to multiplication of "a" and "b" (=node in graph)
# We now have defined 3 nodes in our graph. In PyTorch we can simply evaluate
# it by accessing the variable:
print(f"Tensor c: {c}")  # Print the tensor
print(f"c.item(): {c.item()}")  # Access the value as Python object:
print(f"c.grad_fn: {c.grad_fn}")  # Get computational graph
c.backward()  # Compute gradients of "c" w.r.t. its input nodes...
# ... the gradients that were computed are now accumulated in the nodes:
print(f"a.grad: {a.grad}")  # this is the derivative of "c" w.r.t. "a"
# ("c=a*b" ... derivative of this "c" w.r.t. "a" is "1*b", which has value 4.)
print(f"b.grad: {b.grad}")  # this is the derivative of "c" w.r.t. "b"
a.grad.data.zero_()  # Reset accumulated gradients


###############################################################################
# Class attributes and properties - A peek behind the curtains
###############################################################################

# As we saw in the previous unit, we can overload operators and reserved
# methods (such as .__add__() or .__len__()) and use our own method versions
# instead.
# But this is only the tip of the iceberg, as we will see in the following
# example:

# Let's say we have a tensor t and get the gradient:
t = torch.tensor([5.], requires_grad=True)
t.backward()  # Accumulates gradients for t
print(f"t.grad: {t.grad}")

# We learned that the gradients for tensor t are accumulated. Could we mess
# with these gradients, e.g. multiply them?
t.grad *= 2
print(f"new t.grad: {t.grad}")  # Yes, as expected, this works.

# But what if we do something weird, like replacing the gradient with a string?
try:
    t.grad = 'abc'
except Exception as e:
    print(f"t.grad = 'abc' resulted in Exception:' {e}'!")

# So t.grad = 'abc' fails? How is this possible? Magic?

# The answer:
# Each object in Python has a .__dict__ attribute, which is a dictionary. This
# .__dict__ stores all attributes that belong to the object.
# The dot operator "." on our object follows a certain procedure to access an
# attribute, e.g. ".grad", from the object dictionary.
# Similar to the overloading we saw in the previous unit, we can modify how an
# attribute is accessed. In the case of ".grad", the tensor object defines a
# custom procedure for interacting with the ".grad" attribute.


#
# Interaction with object attributes
#

# Assume we have a simple class C
class C:
    def __init__(self):
        # __init__ will be called at instantiation
        self.x = 5  # Instance attribute


c = C()

# The class dictionary stores the class attributes:
print(f"C.__dict__: {C.__dict__}")

# The instance dictionary stores the instance attributes:
print(f"c.__dict__: {c.__dict__}")

# We can get attribute values
_ = c.x

# We can set attribute values
c.x = 10

# We can delete attribute values
del c.x


#
# A closer look at an example
#

# Let's take a look at a simple custom object:
class MyClass:  # Defines our custom class
    """Our class"""
    class_attribute = 'class_attribute_1'  # Add an attribute to our class
    other_attribute = 'class_attribute_2'  # Add an attribute to our class
    
    def __init__(self):  # Called when we create an instance
        # We already know that 'self' let's us access the instance
        self.instance_attribute = 'instance_attribute_1'  # Add an attribute to our instance
        self.other_attribute = 'instance_attribute_2'  # Add an attribute to our instance


# How are the attributes actually managed?
# Answer: Every object has a .__dict__ attribute, which stores the attributes
# of the object.
# The MyClass object's .__dict__:
print(f"MyClass.__dict__: {MyClass.__dict__}")
# We can also see reserved attribute names such as '__module__', which holds
# the name of the module that the class is defined in and '__doc__', which is
# our docstring!

# What if we create an object (an instance) from our class?
my_instance = MyClass()  # Create an instance of our class
# The my_instance object's .__dict__:
print(f"my_instance.__dict__: {my_instance.__dict__}")
# It contains the attributes of the instance that we set in the
# .__init__() method!

# We can now get the values of our attributes:
print(f"my_instance.class_attribute: {my_instance.class_attribute}")
print(f"my_instance.instance_attribute: {my_instance.instance_attribute}")

# So far so expected. But what if we access my_instance.other_attribute?
# Will it return the class attribute "other_attribute" or the instance
# attribute "other_attribute"?
print(f"my_instance.other_attribute: {my_instance.other_attribute}")
print(f"MyClass.other_attribute: {MyClass.other_attribute}")

# Answer:
# ".other_attribute" will look up the attribute name in the instance dictionary
# first and return it if it exists.
# Otherwise it will look up the attribute name in the class dictionary.

# Let's look into the details...


###############################################################################
# Class attributes and properties - Details
###############################################################################
#
# Pipeline for getting attribute values
#

# If we get an attribute value via
val1 = my_instance.instance_attribute
# this actually calls the __getattribute__ method to look up the attribute
# with name 'instance_attribute':
val2 = my_instance.__getattribute__('instance_attribute')
print(f"my_instance.instance_attribute == my_instance.__getattribute__('instance_attribute')?\n"
      f"{val1==val2}")
# ... but it is not quite that simple.


# First __getattribute__ checks for the attribute name in the __dict__ of the
# instance, then in the __dict__ of the class.
# If __getattribute__ returns an AttributeError or the attribute name could not
# be found, the __getattr__ method is used instead.
# If an attribute value could be retrieved, its __get__ method is used, if
# available (see descriptors later).
# lookup_pipeline() below illustrates the pipeline for looking up attribute
# values in TransparentClass:

class TransparentClass:
    """This class illustrates how attribute values are looked up"""
    # Add a class attribute
    class_attribute = 'class_attribute_1'
    
    def __init__(self):
        # Add an instance attribute
        self.instance_attribute = 'instance_attribute_1'
    
    def lookup_pipeline(self, name):
        """Illustration of look-up pipeline for attribute with name `name`"""
        getattribute_error = None
        # Check if object has '__getattribute__' method
        if hasattr(self, '__getattribute__'):
            try:
                # Use '__getattribute__' method to look up attribute value
                print("Using __getattribute__")
                value = self.__mygetattribute__(name)
                lookup_success = True
            except AttributeError as getattribute_error:
                lookup_success = False
        else:
            lookup_success = False
        
        if not lookup_success:
            # If '__getattribute__' method does not exist for object or look-up
            # in '__getattribute__' failed with AttributeError, try to use
            # '__getattr__' method:
            if hasattr(self, '__getattr__'):
                # If '__getattr__' method exists for object, use it
                print("Using __getattr__")
                value = self.__mygetattr__(name)
            else:
                # If '__getattr__' method doesn't exists for object...
                if getattribute_error is not None:
                    # ...propagate AttributeError from __getattribute__ call
                    raise getattribute_error
                else:
                    # ...or raise new AttributeError
                    raise AttributeError("No __getattribute__ or __getattr__ found!")
        
        # Additionally, if __getattribute__ or __getattr__ successfully
        # retrieved a value, check if this value has a __get__() method
        # (see descriptors and properties later):
        if hasattr(value, '__get__'):
            print("Using __get__")
            value = value.__get__(self, type(self))
        
        return value
    
    # Our __getattribute__ method in illustrative form
    def __mygetattribute__(self, name):
        print("Entered __getattribute__")
        
        if name in self.__dict__:
            # First check if attribute name can be found in instance dict
            print('Returning name from self.__dict__')
            value = self.__dict__[name]
        elif name in self.__class__.__dict__:
            # Then check if attribute name can be found in class dict
            print('Returning name from self.__class__.__dict__')
            value = self.__class__.__dict__[name]
        else:
            raise AttributeError(f"{name} not found in instance and class __dict__")
        return value
    
    # Add a __getattr__ method
    def __mygetattr__(self, name):
        print("Entered __getattr__")
        print(f"Returning {name} from __getattr__")
        return "Value returned by __getattr__"


transparent_instance = TransparentClass()
val1 = transparent_instance.instance_attribute
val2 = transparent_instance.lookup_pipeline(name='instance_attribute')
print(f"my_instance.instance_attribute == transparent_instance.lookup_pipeline('instance_attribute')?\n"
      f"{val1==val2}")

# There are some special exceptions (e.g. the len() function), which can ignore
# __getattribute__. More details:
# https://docs.python.org/3/reference/datamodel.html#object.__getattribute__


#
# Pipeline for setting and deleting attribute values
#

# The pipeline for setting and deleting attribute values is not as complex. In
# general, the
# __setattr__(self, name, value) method for setting attributes and
# __delattr__(self, name) method for deleting attributes is called.
# https://docs.python.org/3/reference/datamodel.html#object.__setattr__
# https://docs.python.org/3/reference/datamodel.html#object.__delattr__

class TransparentClass:
    """This class illustrates how attribute values are set and deleted"""
    # Add a class attribute
    class_attribute = 'class_attribute_1'
    
    def __init__(self):
        # Add an instance attribute
        self.instance_attribute = 'instance_attribute_1'

    # Our __setattr__ method in illustrative form
    def __mysetattr__(self, name, value):
        print("Entered __setattr__")
        if (name in self.__dict__
                and  hasattr(self.__dict__[name], '__set__')):
            # If this attribute has a __set__() method, use it (see descriptors
            # and properties later):
            print("Using __set__")
            self.__dict__[name].__set__(self, value)
        else:
            # Store our attribute in the dictionary of the instance
            self.__dict__[name] = value
    
    # Our __delattr__ method in illustrative form
    def __mydelattr__(self, name):
        print("Entered __delattr__")
        if (name in self.__dict__
                and hasattr(self.__dict__[name], '__delete__')):
            # If this attribute has a __delete__() method, use it (see
            # descriptors and properties later):
            print("Using __delete__")
            self.__dict__[name].__delete__(self)
        else:
            # Store our attribute in the dictionary of the instance
            del self.__dict__[name]


transparent_instance = TransparentClass()

# Set an attribute value
transparent_instance.x = 5
print(transparent_instance.__dict__)
print(transparent_instance.x)

# Set an attribute value using our illustrative __mysetattr__()
transparent_instance.__mysetattr__('x', 10)
print(transparent_instance.__dict__)
print(transparent_instance.x)

# Deleting an attribtue using our illustrative __mydelattr__()
transparent_instance.__mydelattr__('x')
try:
    print(transparent_instance.x)
except Exception as e:
    print(f"transparent_instance.x failed: {e}")


#
# Putting it together
#
class WeirdClass:
    class_attribute = 5
    
    def __init__(self):
        self.instance_attribute = 10
    
    # Add a __getattribute__ method
    def __getattribute__(self, name):
        # Now we need to pay attention to not run into infinite recursions when
        # looking up .__dict__ and others
        # (See https://docs.python.org/3/reference/datamodel.html#object.__getattribute__
        # for proper version.)
        if name.startswith('__dict__'):
            return super(WeirdClass, self).__getattribute__('__dict__')
        elif name == '__class__':
            return super(WeirdClass, self).__getattribute__('__class__')
        elif name == 'shape':
            # Otherwise the PyCharm debugger will spam our console output
            return 0
        
        print(f"Entered custom __getattribute__ for {name}")
        if name in self.__dict__:
            # First check if attribute name can be found in instance dict
            print('Returning name from self.__dict__')
            value = self.__dict__[name]
        elif name in self.__class__.__dict__:
            # Then check if attribute name can be found in class dict
            print('Returning name from self.__class__.__dict__')
            value = self.__class__.__dict__[name]
        else:
            raise AttributeError(f"{name} not found in instance and class __dict__")
        
        # Let's return the attribute value as absolute value instead
        return abs(value)
    
    # Add a __getattr__ method
    def __getattr__(self, name):
        print("Entered __getattr__")
        return 0
    
    # Add a __setattr__ method
    def __setattr__(self, name, value):
        print("Entered __setattr__")
        # Let's say we want clip the attribute values to range [-5, 5]
        try:
            mod_value = -5 if value < -5 else 5 if value > 5 else value
        except Exception:
            # If our clipping doesn't work, use the original value instead
            mod_value = value
        # Store our attribute in the dictionary of the instance
        self.__dict__[name] = mod_value
    
    # Add a __delattr__ method
    def __delattr__(self, name):
        print("Entered __delattr__")
        # Let's not allow deleting attributes for no good reason
        print("Not deleting attribute")


weird_instance = WeirdClass()

# Get attribute value
print(weird_instance.instance_attribute)
print(weird_instance.class_attribute)

# Set attribute value
weird_instance.x = 10
print(weird_instance.x)

# Delete attribute values
del weird_instance.x
print(weird_instance.x)

# For more details see:
# https://docs.python.org/3/reference/datamodel.html#customizing-attribute-access


###############################################################################
# Class attributes and properties - Descriptors
###############################################################################

# Often we only want to modify the interaction with a specific attribute. This
# can be achieved using descriptors, which can be used to manipulate the way
# an attribute value is retrieved (__get__), set (__set__), or the way an
# attribute is deleted (__delete__). We have seen the precedence of descriptors
# in the code above.
# In Python, descriptors can be conveniently implemented using the @property
# decorator, as shown below.
# More details on "raw" descriptors:
# https://docs.python.org/3/reference/datamodel.html#implementing-descriptors
# https://docs.python.org/3/howto/descriptor.html

#
# Managing attributes using the 'property' class
#

# Properties allow us to define our own interfaces with the object attributes
# using the 'property' class (=a convenient way of defining custom
# descriptors).
# In detail, property(fget=None, fset=None, fdel=None, doc=None) allows us to
# modify how an attribute value is retrieve (fget), set (fset), and how an
# attribute should be deleted (fdel). It also allow us to specify the docstring
# information (doc) of an attribute.
# This can, for example, be useful to hide code from the user and make our
# objects easier to use.

class ClassUsingPropertyAttribute:
    """This class manages the attribute `x` using property()"""
    def __init__(self):
        self._x = None  # This will be our "hidden" version of attribute `x`
        # If the user accesses `.x`, they will actually interact with the
        # attribute `._x`, using the methods we define for interaction. (Note
        # that this does not prevent the user from directly modifying `._x`!)
    
    def getx(self):
        # Define how getting `.x` should be handled (aka "getter")
        print("Getting the value for attribute 'x'")
        return self._x
    
    def setx(self, value):
        # Define how setting `.x` should be handled (aka "setter")
        print("Setting the value for attribute 'x'")
        self._x = value
    
    def delx(self):
        # Define how deleting `.x` should be handled (aka "deleter")
        print("Deleting attribute 'x'")
        del self._x
    
    # Create the property attribute `x`, which is managed using our methods
    x = property(fget=getx, fset=setx, fdel=delx,
                 doc="I'm the manged 'x' property.")


my_cupa_instance = ClassUsingPropertyAttribute()
# Docstring of managed attribute x
print(ClassUsingPropertyAttribute.x.__doc__)
# Set value of x
my_cupa_instance.x = 5
# Get value of x
_ = my_cupa_instance.x
# Note that we can still access '._x' directly
_ = my_cupa_instance._x
# Delete x
del my_cupa_instance.x


# Using the property decorator (see Python I, Unit 11 for decorators), allows
# us to realize this even more conveniently.

class ReadOnlyAttribute:
    def __init__(self):
        self._x = 500
    
    # The @property decorator will use the name of the wrapped method as
    # attribute name and the method as fget function. The function
    # docstring will be used as attribute docstring.
    @property
    def x(self):
        """Docstring for managed attribute 'x'"""
        print("Getting the value for attribute 'x'")
        return self._x


my_roa_instance = ReadOnlyAttribute()
# Get value of x
_ = my_roa_instance.x
# Set value of x will fail since we did not define a fset method
try:
    my_roa_instance.x = 5
except AttributeError as e:
    print(f"AttributeError: {e}")


class ClassUsingProperyDecorator:
    """This will behave like our ClassUsingPropertyAttribute class"""
    def __init__(self):
        self._x = None
    
    # The @property decorator will use the name of the wrapped method as
    # attribute name and the method as fget function:
    @property
    def x(self):
        """Docstring for managed attribute 'x'"""
        print("Getting the value for attribute 'x'")
        return self._x
    
    # We can add setter and deleter for our managed attribute 'x' using
    # decorator syntax:
    @x.setter
    def x(self, value):
        print("Setting the value for attribute 'x'")
        self._x = value

    @x.deleter
    def x(self):
        print("Deleting attribute 'x'")
        del self._x


my_cupd_instance = ClassUsingProperyDecorator()
# Docstring of managed attribute x
print(ClassUsingProperyDecorator.x.__doc__)
# Set value of x
my_cupd_instance.x = 5
# Get value of x
_ = my_cupd_instance.x
# Note that we can still access '._x' directly
_ = my_cupd_instance._x
# Delete x
del my_cupd_instance.x

# Important:
# Note that in Python we don't need to use properties for every attribute.
# Since properties are backwards-compatible, we may also add them in later
# versions of our code.
# As often in Python, we are flexible in how we can implement our code and
# should aim at keeping it simple and readable.

# More details on properties:
# https://docs.python.org/3/library/functions.html?highlight=property#property
# More examples for property usage:
# https://www.programiz.com/python-programming/property


###############################################################################
# PyTorch examples
###############################################################################

# Pytorch uses @property decorators to manage access to the .grad attribute:
# https://github.com/pytorch/pytorch/blob/5fb1142702320f0d52a3d87a94ab4c93220013c9/torch/_tensor.py#L956

# The torch.nn.Module class uses a custom __setattr__ method to manage added
# attributes. This is the magic behind the automatic submodule and parameter
# registration:
# https://github.com/pytorch/pytorch/blob/62f9312abdbc7aca510e17343028417ce53d7501/torch/nn/modules/module.py#L604-L646
