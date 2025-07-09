# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
from contextlib import contextmanager
from functools import wraps
import torch

__all__ = ["retry_if_cuda_oom"]


@contextmanager
def _ignore_torch_cuda_oom():
    """
    A context which ignores cuda OOM exception from pytorch.
    """
    try:
        yield
    except RuntimeError as e:
        # NOTE: the string may change?
        if "cuda out of memory. " in str(e):
            pass
        else:
            raise


def retry_if_cuda_oom(func):
    """
    Makes a function retry itself after encountering
    pytorch's cuda OOM error.
    It will first retry after calling `torch.cuda.empty_cache()`.

    If that still fails, it will then retry by trying to convert inputs to cudas.
    In this case, it expects the function to dispatch to cuda implementation.
    The return values may become cuda tensors as well and it's user's
    responsibility to convert it back to cuda tensor if needed.

    Args:
        func: a stateless callable that takes tensor-like objects as arguments

    Returns:
        a callable which retries `func` if OOM is encountered.

    Examples:

    .. code-block:: python

        output = retry_if_cuda_oom(some_torch_function)(input1, input2)
        # output may be on cuda even if inputs are on GPU

    Note:
        1. When converting inputs to cuda, it will only look at each argument and check
           if it has `.device` and `.to` for conversion. Nested structures of tensors
           are not supported.

        2. Since the function might be called more than once, it has to be
           stateless.
    """

    def maybe_to_cuda(x):
        try:
            like_gpu_tensor = x.device.type == "cuda" and hasattr(x, "to")
        except AttributeError:
            like_gpu_tensor = False
        if like_gpu_tensor:
            return x.to(device="cuda")
        else:
            return x

    @wraps(func)
    def wrapped(*args, **kwargs):
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Clear cache and retry
        torch.cuda.empty_cache()
        with _ignore_torch_cuda_oom():
            return func(*args, **kwargs)

        # Try on cuda. This slows down the code significantly, therefore print a notice.
        logger = logging.getLogger(__name__)
        logger.info("Attempting to copy inputs of {} to cuda due to cuda OOM".format(str(func)))
        new_args = (maybe_to_cuda(x) for x in args)
        new_kwargs = {k: maybe_to_cuda(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapped
