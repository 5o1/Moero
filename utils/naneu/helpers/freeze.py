from contextlib import contextmanager
from torch.nn import Module

@contextmanager
def freeze(model: Module):
    """
    A context manager to temporarily freeze the parameters of a model by setting
    their `requires_grad` attribute to `False`. This is useful for scenarios where
    you want to perform operations on a model without updating its parameters.
    Attributes:
        model (torch.nn.Module): The model whose parameters will be frozen.
        original_states (dict): A dictionary to store the original `requires_grad`
            states of the model's parameters.
    Methods:
        __enter__(): Freezes the parameters of the model by setting their
            `requires_grad` attribute to `False`.
        __exit__(exc_type, exc_val, exc_tb): Restores the original `requires_grad`
            states of the model's parameters.    
    """
    original_states = {p: p.requires_grad for p in model.parameters()}
    try: # __enter__ method
        for param in model.parameters():
            param.requires_grad = False
        yield
    finally: # __exit__ method
        for param, requires_grad in original_states.items():
            param.requires_grad = requires_grad