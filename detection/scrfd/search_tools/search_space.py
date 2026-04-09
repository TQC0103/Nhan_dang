"""Minimal search-space helpers used by SCRFD config generators.

This replaces the small subset of autotorch APIs used in this repo:
Choice, Int, Real, List, and the @obj decorator exposing `.rand`.
"""

import random


class _Space:
    def sample(self):
        raise NotImplementedError


class Choice(_Space):
    def __init__(self, *values):
        if not values:
            raise ValueError('Choice requires at least one value')
        self.values = values

    def sample(self):
        return random.choice(self.values)


class Int(_Space):
    def __init__(self, lower, upper):
        if upper < lower:
            raise ValueError('Int upper bound must be >= lower bound')
        self.lower = int(lower)
        self.upper = int(upper)

    def sample(self):
        return random.randint(self.lower, self.upper)


class Real(_Space):
    def __init__(self, lower, upper):
        if upper < lower:
            raise ValueError('Real upper bound must be >= lower bound')
        self.lower = float(lower)
        self.upper = float(upper)

    def sample(self):
        return random.uniform(self.lower, self.upper)


class List(_Space):
    def __init__(self, *items):
        if not items:
            raise ValueError('List requires at least one item')
        self.items = items

    def sample(self):
        return [_sample_value(item) for item in self.items]


def _sample_value(value):
    if isinstance(value, _Space):
        return value.sample()
    return value


def obj(**spaces):
    def decorator(cls):
        class Wrapped(cls):
            _search_spaces = spaces

            @property
            def rand(self):
                sampled = {
                    key: _sample_value(space)
                    for key, space in self._search_spaces.items()
                }
                return cls(**sampled)

        Wrapped.__name__ = cls.__name__
        Wrapped.__qualname__ = cls.__qualname__
        Wrapped.__module__ = cls.__module__
        Wrapped.__doc__ = cls.__doc__
        return Wrapped

    return decorator
