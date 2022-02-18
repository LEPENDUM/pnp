import torch


def to_tensor(numeric, device=None):
    """
    Convert arbitrary iterable type (e.g. list, tuple, numpy array) or scalar value to a tensor (keep input dimensions).
    """
    if not isinstance(numeric, torch.Tensor):
        return torch.tensor(numeric if is_iterable(numeric) else [numeric], device=device)
    else:
        return numeric.to(device)


def is_iterable(obj) -> bool:
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False
