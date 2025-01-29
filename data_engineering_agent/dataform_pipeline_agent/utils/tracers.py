"""
This module provides a decorator for tracing function calls.

The `trace_calls` decorator can be applied to functions to print information
about their arguments and return values, aiding in debugging and understanding
the flow of execution.
"""
from functools import wraps

def trace_calls(func):
    """
    A decorator to trace function calls, printing arguments and return values.

    This decorator wraps the given function, printing the function's name,
    arguments, and return value to the console. It uses `functools.wraps` to
    preserve the original function's metadata (name, docstring, etc.).

    Args:
        func: The function to be traced.

    Returns:
        The wrapped function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with args={args} kwargs={kwargs}")
        print(f"Calling {func.__name__} ")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result

    return wrapper