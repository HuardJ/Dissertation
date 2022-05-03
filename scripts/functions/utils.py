# -*- coding: utf-8 -*-
"""
Created on Wed Sep 1 2021

@author: Joe L/Joe HC/Steven H

All the utility functions used in the project are stored here. Anything that is not directly applied to the project
context can be found here.

"""

from typing import Any, Optional, Union
import pandas as pd
import matplotlib.cm as cm
import datetime
import random

random.seed(42)


def dictfilterNoneFalse(dictionary: dict[Any, Any]) -> dict[Any, Any]:
    """
    Removes key-value pairs from dictionary if the value is False or None.
    """
    output = filter(lambda el: el[1] not in {False, None}, dictionary.items())
    return {key: value for key, value in output}


def multi_input(func, key_input, type_singular):
    """ Allows functions made for a singular non-array-like data type to be
        applied to the elements of lists, tuples, pandas Series or the cells
        of a pandas DataFrame.

        func : callable or lambda fuction containing a callable. The collable will normally
        only work on a particular type of data i.e. type_singualar.

        key_input : value or an array-like of values of type_singular.

        type_singular: a non-array-like type such as int or str. NOTE: list is able to be
        passed here, if list is passed, the function will search for the deepest data type
        that contains lists.
    """

    if type(key_input) is type_singular:

        # check for 'list' type_singular caveat
        if type_singular is list:

            # Looking for the deepest list of non-list items to apply func to
            if any([type(el) is not list for el in key_input]):
                output = func(key_input)

            else:  # if key input is a list of lists then perform recursion
                output = [multi_input(func, el, type_singular) for el in key_input]

        else:
            output = func(key_input)

    # Dataframe case
    elif type(key_input) is pd.core.frame.DataFrame:

        output = key_input.applymap(lambda x: multi_input(func, x, type_singular))

    # list,tuple,Series case
    elif any([type(key_input) is el for el in [list, tuple, pd.core.series.Series]]):

        output = [multi_input(func, el, type_singular) for el in key_input]

    else:

        raise TypeError('key_input datatype not recognised')

    return output


######################

### Chainer Classes

class Chainer:
    """This class stores functions and applies them sequentially"""

    def __init__(
        self, 
        *func_list: callable
        ):

        self._functions = func_list

    def add_function(self, func: callable) -> None:
        """Append a function to the list of currently stored functions"""
        self._functions.append(func)
        return None

    def extend_chain(self, *additional_functions: callable) -> None:
        """Append multiple functions to the list of currently stored functions"""
        self._functions.extend(additional_functions)
        return None

    def __call__(
        self, 
        arg: Any,
        partial_chain: int = 0
        ) -> Any:
        """
        Applies functions from _functions to arg in a chain. Partial chain
        allows the chain to start from a specified function position.
        """
        if len(self._functions) == 0:
            return arg

        if len(self._functions) - 1 < partial_chain:
            raise ValueError("partial_chain value beyond valid index number for _functions.")

        # apply first function manually on args
        output = self._functions[partial_chain](arg)

        if len(self._functions) - 1 > partial_chain:

            # apply all following functions on output
            for f in self._functions[partial_chain + 1:]:
                output = f(output)

        return output


class ChainerCleanList(Chainer):
    """
        This class stores functions and applies them sequentially.
        Unlike base Chainer class this class can accept None as a function, which it will then filter out.
    """
    def __init__(
        self, 
        *func_list: Union[callable, None]
        ):
        self._functions = self.clean_list(func_list)

    @classmethod
    def clean_list(
        cls,
        func_list: list[callable]
    ):
        return list(filter(None, func_list))

    def add_function(self, func: callable) -> None:
        """Append a function to the list of currently stored functions"""
        if func is not None:
            super().add_function.append(func)

    def extend_chain(self, *additional_functions: callable) -> None:
        """Append multiple functions to the list of currently stored functions"""
        func_list = self.clean_list(additional_functions)
        if len(func_list):
            super().extend_chain(additional_functions)


class ChainerDictionary:
    """
        This class stores functions and applies them sequentially.
        This class holds functions as a dictionary
    """
    def __init__(
        self, 
        **func_kwargs
        ):
        self._func_dict = func_kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        func_list = list(self._func_dict.values())

        output = func_list[0](*args, **kwargs)

        if len(func_list) > 1:
            for func in func_list[1:]:
                output = func(output)

        return output


class PlotlyColorMap:
    """
        This class converts a specified matplotlib's colour
        map output to an output compatible with plotly:

        Attributes:

            matplotlib_cmap: callable. Matplotlib colourmap to use as a
            template. The map retrievied by passing to __init__ the name of
            the colourmap as a string, and the definition - number of shades
            to run through along the colour scale - as an int.

            transparency: Bool. Whether to vary colour opacity.

        Methods:

            rgba: Input must be a value between 0 and 1. Output is an rgba
            string to be parsed by plotly

    """

    def __init__(self, matplotlib_name, definition=255, transparency=False):
        self.matplotlib_cmap = cm.get_cmap(matplotlib_name, definition)
        self.transparency = transparency

    def rgba(self, zero_to_one):
        # retrieve matplotlib colour format in mutable form
        colour = list(self.matplotlib_cmap(zero_to_one))

        # Check and apply transparency
        if self.transparency:
            colour[-1] = colour[-1] * zero_to_one

        # parse into plotly form - convert to formatted strings and list
        colour = ['{0:.6f}'.format(el) for el in colour]

        colour = 'rgba(' + ','.join(colour) + ')'

        return colour


def dummy_date_generator():
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2021, 9, 1)

    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)

    return random_date

def date_returner(date):
    year = date.year
    month = date.month
    if date.month < 10:
        month = f"{month:02d}"
    else:
        month = month

    date = str(year)+ "-" + str(month)
    return date