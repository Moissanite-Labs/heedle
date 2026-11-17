"""Heedle: the layer where a person steers a model's change by hand, inside checked bounds.

    heedle.core    the mathematics, for any host
    heedle.hosts   the cloth host and the text host
    heedle.draft   replaying a steering session without the instrument
"""
from . import core, hosts  # noqa: F401

__version__ = "0.1.0"
