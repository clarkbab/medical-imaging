from typing import *

from mymi.typing import *

from .utils import isinstance_generic

def arg_assert_lengths(args: List[List[Any]]) -> None:
    arg_0 = args[0]
    for arg in args[1:]:
        if len(arg) != len(arg_0):
            raise ValueError(f"Expected arg lengths to match. Length of arg '{arg}' didn't match '{arg_0}'.")

def arg_assert_literal(
    arg: Any,
    literal: Union[Any, List[Any]]) -> None:
    literals = arg_to_list(literal, type(arg))
    if arg not in literals:
        raise ValueError(f"Expected argument to be one of '{literals}', got '{arg}'.")

def arg_assert_literal_list(
    arg: Union[Any, List[Any]],
    arg_type: Any,
    literal: Union[Any, List[Any]]) -> None:
    args = arg_to_list(arg, arg_type)
    literals = arg_to_list(literal, arg_type)
    for arg in args:
        if arg not in literals:
            raise ValueError(f"Expected argument to be one of '{literals}', got '{arg}'.")

def arg_assert_present(
    arg: Any,
    name: str) -> None:
    if arg is None:
        raise ValueError(f"Argument '{name}' expected not to be None.")

def arg_broadcast(
    arg: Any,
    b_arg: Any,
    arg_type: Optional[Any] = None,
    out_type: Optional[Any] = None):
    # Convert arg to list.
    if arg_type is not None:
        arg = arg_to_list(arg, arg_type, out_type=out_type)

    # Get broadcast length.
    b_len = b_arg if type(b_arg) is int else len(b_arg)

    # Broadcast arg.
    if isinstance(arg, Iterable) and not isinstance(arg, str) and len(arg) == 1 and b_len != 1:
        arg = b_len * arg
    elif not isinstance(arg, Iterable) or (isinstance(arg, Iterable) and isinstance(arg, str)):
        arg = b_len * [arg]

    return arg

def arg_to_list(
    arg: Optional[Any],
    arg_types: Union[Any, List[Any]],
    broadcast: int = 1,      # Expand a matching type to multiple elements, e.g. None -> [None, None, None].
    literals: Dict[str, Tuple[Any]] = {},
    out_type: Optional[Any] = None) -> List[Any]:
    # Convert arg types to list.
    if not isinstance(arg_types, list) and not isinstance(arg_types, tuple):
        arg_types = [arg_types]
    
    # Check literal matches.
    literal_types = (int, str) 
    for t in literal_types:
        if isinstance(arg, t) and arg in literals:
            arg = literals[arg]
            # If arg is a function, run it now. This means the function
            # is not evaluated every time 'arg_to_list' is called, only when
            # the arg matches the appropriate literal (e.g. 'all').
            if isinstance(arg, Callable):
                arg = arg()

            return arg

    # Check types.
    matched = False
    for t in arg_types:
        if isinstance_generic(arg, t):
            matched = True
            arg = [arg] * broadcast
            break
        
    # Convert to output type.
    if matched and out_type is not None:
        arg = [out_type(a) for a in arg]

    return arg
