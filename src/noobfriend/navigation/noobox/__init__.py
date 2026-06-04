"""The :class:`NooBox` subpackage: a managed collection of NooBooks.

This subpackage is the "fat module" for :class:`NooBox`: it owns the
collection model and its :attr:`~NooBox.viz` accessor. Only the eponymous
:class:`NooBox` is public; its accessors are internal.
"""

from noobfriend.navigation.noobox._core import NooBox

__all__ = ["NooBox"]
