# base_utils.py

class BaseClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello from {self.name}!"


def add_numbers(x, y):
    """Adds two numbers together."""
    return x + y


def multiply_by_two(x):
    """Multiplies a number by two."""
    return x * 2
