from typing import Optional, Union, List, Any, Tuple

def to_list(x: Optional[Union[Tuple[Any, ...], List[Any], Any]]) -> List[Any]:
    """
    Converts an element or a None to a list.
    If x is already a list, returns x.
    """
    if x is None:
        return []
    if isinstance(x, tuple):
        return list(x)
    return x if isinstance(x, list) else [x]

def to_tuple(x: Optional[Union[Tuple[Any, ...], List[Any], Any]]) -> Tuple[Any, ...]:
    """
    Converts an element or a None to a tuple.
    If x is already a tuple, returns x.
    """
    if x is None:
        return ()
    if isinstance(x, list):
        return tuple(x)
    return x if isinstance(x, tuple) else (x,)


