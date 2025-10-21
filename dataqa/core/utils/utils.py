import importlib
import inspect
import json
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, Text, Type, TypeVar, Union

import yaml

T = TypeVar("T")


def class_from_module_path(
    module_path: Text, lookup_path: Optional[Text] = None
) -> Type:
    """Given the module name and path of a class, tries to retrieve the class.

    The loaded class can be used to instantiate new objects.

    Args:
        module_path: either an absolute path to a Python class,
                     or the name of the class in the local / global scope.
        lookup_path: a path where to load the class from, if it cannot
                     be found in the local / global scope.

    Returns:
        a Python class

    Raises:
        ImportError, in case the Python class cannot be found.
        RasaException, in case the imported result is something other than a class
    """
    klass = None
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition(".")
        m = importlib.import_module(module_name)
        klass = getattr(m, class_name, None)
    elif lookup_path:
        # try to import the class from the lookup path
        m = importlib.import_module(lookup_path)
        klass = getattr(m, module_path, None)

    if klass is None:
        raise ImportError(f"Cannot retrieve class from path {module_path}.")

    if not inspect.isclass(klass):
        raise TypeError(
            f"`class_from_module_path()` is expected to return a class, "
            f"but for {module_path} we got a {type(klass)}."
        )
    return klass


def cls_from_str(name: str) -> Type[Union[Any, T]]:
    """
    Returns a class object with the name given as a string.
    :param name: The name of the class as a string.
    :return: The class object.
    :raises ImportError: If the class cannot be retrieved from the path.
    """
    try:
        return class_from_module_path(name)
    except (AttributeError, ImportError, TypeError, ValueError):
        raise ImportError(f"Cannot retrieve class from path {name}.")


def load_file(file_path: Union[str, Path]):
    str_file_path = deepcopy(file_path)
    if isinstance(file_path, Path):
        str_file_path = str(file_path)

    if str_file_path.endswith("json"):
        return json.load(open(str_file_path))
    if str_file_path.endswith("yml"):
        return yaml.safe_load(open(str_file_path))
    if str_file_path.endswith(".pkl"):
        return pickle.load(open(str_file_path, "rb"))
    return open(str_file_path).read()


def generate_alphabetic_bullets(n: int):
    """
    Generate a list of alphabetic bullets of length `n`.

    :param n: The length of the list.
    :type n: int

    :return: A list of alphabetic bullets.
    :rtype: List[str]
    """
    bullets = []
    i = 0
    while len(bullets) < n:
        bullet = ""
        temp = i
        while temp >= 0:
            bullet = chr(65 + temp % 26) + bullet
            temp = temp // 26 - 1
        bullets.append(bullet)
        i += 1
    return bullets


def string_list_to_prompt(
    string_list: List[str], prefix: Union[str, List[str]]
) -> str:
    if not isinstance(prefix, list):
        new_list = [prefix + s for s in string_list]
    else:
        new_list = [prefix[i] + s for i, s in enumerate(string_list)]
    return "\n".join(new_list)
