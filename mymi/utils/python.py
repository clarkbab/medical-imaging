import ast
import inspect
import sys
from typing import *

from .utils import arg_to_list, is_windows

def create_optional_properties(
    locals: Dict[str, Any],
    parents: Union[str, List[str]],
    children: Union[str, List[str]],
    child_prefix: str = '',     # Prefixs/suffixes are added to the called attributes.
    child_suffix: str = '',
    parent_prefix: str = '',
    parent_suffix: str = '') -> Callable:
    # Creates a property that first checks if the parent attribute is not None,
    # before returning the child attribute.
    parents = arg_to_list(parents, str)
    children = arg_to_list(children, str)
    for p in parents:
        for c in children:
            def prop_fn(self, p=p, c=c):    # Captures p and c in the closure.
                def_mod = getattr(self, f'{parent_prefix}{p}{parent_suffix}')
                if def_mod is not None:
                    return getattr(def_mod, f'{child_prefix}{c}{child_suffix}')
                return None
            locals[f'{p}_{c}'] = property(prop_fn)

def filter_lists(
    lists: List[List[Any]],
    filt_fn: Callable) -> List[List[Any]]:
    n_elements = len(lists[0])
    for l in lists:
        if len(l) != n_elements:
            raise ValueError('All lists must have the same length.')
    lists = map(list, zip(*[i for i in list(zip(*lists)) if filt_fn(i)]))
    return lists

def python_version(gte: Optional[str] = None) -> Union[Tuple[int, int, int], bool]:
    info = sys.version_info
    if gte is not None:
        res = [int(x) for x in gte.split('.')]
        if len(res) == 1:
            gte_major, gte_minor, gte_micro = res[0], -1, -1
        elif len(res) == 2:
            gte_major, gte_minor, gte_micro = res[0], res[1], -1
        elif len(res) == 3:
            gte_major, gte_minor, gte_micro = res[0], res[1], res[2]
        if info.major > gte_major or (info.major == gte_major and info.minor > gte_minor) or (info.major == gte_major and info.minor == gte_minor and info.micro >= gte_micro):
            return True
        else:
            return False
    version = (info.major, info.minor, info.micro)
    return version

class CallVisitor(ast.NodeVisitor):
    def __init__(
        self,
        inner_fn: Callable):
        self.__inner_fn = inner_fn
        self.__args = []
        self.__kwargs = []
    
    @property
    def args(self) -> List[str]:
        return self.__args
        
    @property
    def kwargs(self) -> List[str]:
        return self.__kwargs

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.__inner_fn.__name__:
            for a in node.args:
                if isinstance(a, ast.Starred):
                    self.__args.append('args')
                else:
                    self.__args.append(ast.unparse(a))
            for k in node.keywords:
                if k.arg is None:
                    self.__kwargs.append('kwargs')
                else:
                    self.__kwargs.append(k.arg)

def get_inner_args(
    outer_fn: Callable,
    inner_fn: Callable) -> List[str]:
    source = inspect.getsource(outer_fn)
    tree = ast.parse(source)
    visitor = CallVisitor(inner_fn)
    visitor.visit(tree)
    return visitor.args, visitor.kwargs

def delegates(*inner_fns: Callable) -> Callable:
    if is_windows() or not python_version(gte='3.9'):
        # Ast unparse is not available in Python < 3.9.
        return lambda f: f

    def change_outer_fn_sig(outer_fn: Callable) -> Any:
        # Load params.
        outer_sig = inspect.signature(outer_fn)
        outer_params = dict(outer_sig.parameters)
        outer_params_args = dict((k, v) for k, v in outer_params.items() if v.default is inspect.Parameter.empty)
        outer_params_kwargs = dict((k, v) for k, v in outer_params.items() if v.default is not inspect.Parameter.empty)

        # Bubble some args from the inner function up to the outer function signature.
        bubbled_args = {}
        bubbled_kwargs = {}
        for f in inner_fns:
            inner_params = dict(inspect.signature(f).parameters)
            inner_args, inner_kwargs = get_inner_args(outer_fn, f)
            inner_params_args = dict((k, v) for k, v in inner_params.items() if v.default is inspect.Parameter.empty)
            inner_params_kwargs = dict((k, v) for k, v in inner_params.items() if v.default is not inspect.Parameter.empty)
            if 'args' in outer_params and 'args' in inner_args:
                for k, v in inner_params_args.items():
                    if k not in inner_args:  # I.e. not already passed by inner call.
                        bubbled_args[k] = v
            if 'kwargs' in outer_params and 'kwargs' in inner_kwargs:
                for k, v in inner_params_kwargs.items():
                    if k not in inner_kwargs:  # I.e. not already passed by inner call.
                        bubbled_kwargs[k] = v
                    
        # Create final signature.
        outer_params_args = dict((k, v) for k, v in outer_params_args.items() if k not in ['args', 'kwargs'])
        args = {}
        args = args | outer_params_args
        args = args | bubbled_args
        kwargs = {}
        kwargs = kwargs | outer_params_kwargs
        kwargs = kwargs | bubbled_kwargs
        # Sort alphabetically, but keyword-only params must be last.
        kw_only_kwargs = dict(sorted((k, v) for k, v in kwargs.items() if v.kind is inspect.Parameter.KEYWORD_ONLY))
        other_kwargs = dict(sorted((k, v) for k, v in kwargs.items() if v.kind is not inspect.Parameter.KEYWORD_ONLY))
        kwargs = other_kwargs | kw_only_kwargs
        params = args | kwargs
        
        outer_fn.__signature__ = outer_sig.replace(parameters=params.values())
        return outer_fn
    return change_outer_fn_sig

def has_private_attr(obj, attr_name):
    attr_name = f"_{obj.__class__.__name__}{attr_name}"
    return hasattr(obj, attr_name)

def sort_lists(
    lists: List[List[Any]],
    sort_fn: Callable) -> List[List[Any]]:
    n_elements = len(lists[0])
    for l in lists:
        if len(l) != n_elements:
            raise ValueError('All lists must have the same length.')
    lists = map(list, zip(*sorted(zip(*lists), key=sort_fn)))
    return lists
