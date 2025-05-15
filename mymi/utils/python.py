import inspect
from typing import *

def delegates(to: Callable) -> Callable:
    def change_outer_fn_sig(outer_fn: Callable) -> Any:
        inner_params = dict(inspect.signature(to).parameters)
        outer_sig = inspect.signature(outer_fn)
        outer_params = dict(outer_sig.parameters)
        new_params = outer_params.copy()
        if 'args' in new_params:
            new_params.pop('args')
        if 'kwargs' in new_params:
            new_params.pop('kwargs')
        for k, v in inner_params.items():
            if k not in new_params:
                new_params[k] = v
        outer_fn.__signature__ = outer_sig.replace(parameters=new_params.values())
        return outer_fn
    return change_outer_fn_sig

def has_private_attr(obj, attr_name):
    attr_name = f"_{obj.__class__.__name__}{attr_name}"
    return hasattr(obj, attr_name)
