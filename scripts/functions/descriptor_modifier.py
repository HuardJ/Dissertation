# -*- coding: utf-8 -*-
"""
Created on: Sun, 24 Apr 2022

@author: Joe HC
"""

from .child_argument_handlers import ChildArgumentHandler
from inspect import isabstract
from abc import ABC, abstractmethod

class BaseDescriptorModifier(ABC):
    """
    Abstraction Layer, used so functionality of classes can be extended more easily.
    
    Descriptor Modifiers manage:
        1. A specific __init__ parameter, the same way a standard python descriptor does
        2. Setting a special modifier method, related to the __init__ parameter.
    """
    def __init__(
        self, 
        base_class: type,
        argument_type_handling: ChildArgumentHandler,
        error_message: str, 
        **init_kwargs
    ):
        self.base_class = base_class
        self.error_message = error_message
        self.argument_type_handling = argument_type_handling
        self.init_kwargs = init_kwargs

        self.update = 0

    def __set_name__(self, owner, name):
        self.private_name = '_' + name
        self.modifier = self.private_name + '_modifier'

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def get_children_dict(
        self
    ) -> dict[str, type]:
        subclass_dict = {
            cls._key: cls for cls in self.base_class.__subclasses__() 
            if not isabstract(cls)
        }
        return subclass_dict

    @abstractmethod
    def update_object(
        self,
        obj
    ) -> None:
        pass

    def __set__(self, obj, value) -> None:
        """
        This method returns a specific class from a collection of child classes.
        Allows for multiple options to be passed to a class without breaking SOLID principles.
        """
        children = self.get_children_dict()
        modifier = self.argument_type_handling(
            value, 
            children,
            self.error_message
        )

        if modifier:
            if len(self.init_kwargs):
                modifier = modifier(value, **self.init_kwargs)
            else: 
                modifier = modifier(value)

        setattr(obj, self.modifier, modifier)
        setattr(obj, self.private_name, value)

        self.update_object(obj)