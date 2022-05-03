# -*- coding: utf-8 -*-
"""
Created on: Sun, 24 Apr 2022

@author: Joe HC

Family of classes that:
    1. checks whether an arguement is valid
    2. Extracts and returns a specific class from a dict[str, type] object.

The extracted class extends functionality to the class/function it is an argument for.
"""

from abc import ABC, abstractmethod
from typing import Any, Collection

### Abstract class
class ChildArgumentHandler(ABC):
    @abstractmethod
    def __init__():
        pass

    @abstractmethod
    def argcheck(
        self,
        arg : Any,
        child_lookup: dict[Any, type],
        error_message: str
    ) -> None:
        pass

    @abstractmethod
    def get_child(
        self, 
        arg: Any, 
        child_lookup: dict[Any, type],
    ) -> type: 
        pass

    def __call__(
        self,
        arg: Any,
        child_lookup: dict[Any, type],
        error_message: str
    ):
        self.argcheck(
            arg,
            child_lookup,
            error_message
        )
        return self.get_child(arg, child_lookup)

######################

# for primitive type arguments with basic get protocols
class BasicArgHandler(ChildArgumentHandler):
    def __init__(
        self,
        *types: type
    ):
        self.types = set(types)

    def argcheck(
        self,
        arg : Any,
        child_lookup: dict[Any, type],
        error_message: str
    ) -> None:
        if type(arg) not in self.types:
            raise ValueError(error_message)
        if str(arg) not in child_lookup.keys():
            raise ValueError(error_message)

    def get_child(
        self,
        arg: Any,
        child_lookup: dict[Any, type]
    ):
        return child_lookup.get(str(arg))

# for primitive type arguments with a default protocal when argument passed is None
class OptionalArgHandler(ChildArgumentHandler):
    def __init__(
        self,
        *types: type
    ):
        self.types = set(types)

    def argcheck(
        self,
        arg : Any,
        child_lookup: dict[Any, type],
        error_message: str
    ) -> None:
        if arg is not None and type(arg) not in self.types:
            raise ValueError(error_message)
        if arg is not None and str(arg) not in child_lookup.keys():
            raise ValueError(error_message)

    def get_child(
        self,
        arg: Any,
        child_lookup: dict[Any, type]
    ):
        return child_lookup.get(str(arg), child_lookup.get('default'))


# Argument handler for instances of modifer classes. All modifier classes must
# be the child of an abstract class that designates it as a spacy modifier.
class ModifierInstanceArgHandler(ChildArgumentHandler):
    def __init__(
        self,
        *types: type
    ):
        self.types = tuple(types)

    def argcheck(
        self,
        arg : Any,
        child_lookup: dict[Any, type],
        error_message: str
    ) -> None:
        if arg is not None and not isinstance(arg, self.types):
            print(arg, type(arg))
            raise ValueError(error_message)
        if arg is not None and arg.__class__.__base__.__name__ not in child_lookup.keys():
            raise ValueError(error_message)

    def get_child(
        self,
        arg: Any,
        child_lookup: dict[Any, type]
    ):
        return child_lookup.get(arg.__class__.__base__.__name__)


# Argument handler for parameter that accepts 
# None, a boolean input or some collection of objects
class OptionalBoolCollectionArgHandler(ChildArgumentHandler):
    instances = bool, Collection

    def __init__(
        self
    ):
        pass

    def argcheck(
        self,
        arg : Any,
        child_lookup: dict[Any, type],
        error_message: str
    ) -> None:
        if arg is not None and not isinstance(arg, self.instances):
            raise ValueError(error_message)
        if arg is not None and not isinstance(arg, Collection):
            if str(arg) not in child_lookup.keys():
                raise ValueError(error_message)

    def get_child(
        self,
        arg: Any,
        child_lookup: dict[Any, type]
    ):
        if isinstance(arg, Collection):
            return child_lookup.get(str(Collection))

        return child_lookup.get(str(arg), child_lookup.get('False'))


# Is instance argument handler, used to find the modifier 
# class based on wether the input argument is an instance of 
# the modifier class' _key variable.
class IsInstanceArgHandler(ChildArgumentHandler):

    def __init__(
        self
    ):
        pass

    def argcheck(
        self,
        arg : Any,
        child_lookup: dict[Any, type],
        error_message: str
    ) -> None:
        return None

    def get_child(
        self,
        arg: Any,
        child_lookup: dict[Any, type]
    ):
        modifier_dict = {
            isinstance(arg, _key): modifier for _key, modifier in child_lookup.items()
        }
        return modifier_dict.get(True, child_lookup.get(type(None)))