from functools import partial

import torch


def to_device(x, device):
  """
  Moves a tensor or a collection of tensors (list, tuple, dict) to the specified device.

  Args:
    x: A torch.Tensor, or a (possibly nested) list, tuple, or dict containing tensors.
    device: The target device (e.g., 'cpu', 'cuda').

  Returns:
    The input x with all tensors moved to the specified device, preserving the original structure.

  Raises:
    Exception: If x is not a tensor, list, tuple, or dict.
  """
  if torch.is_tensor(x):
    x = x.to(device, non_blocking=True)
    return x
  elif isinstance(x, list):
    x = list(map(partial(to_device, device=device), x))
    return x
  elif isinstance(x, tuple):
    x = tuple(map(partial(to_device, device=device), x))
    return x
  elif isinstance(x, dict):
    x = {
      k: to_device(v, device=device)
      for k, v in x.items()
    }
    return x
  elif isinstance(x, str):
    return x
  else:
    raise Exception(f"Can not handle x with {type(x)}")

