"""The :class:`NooBook` subpackage: one JWST product file as a rich object.

This subpackage is the "fat module" for :class:`NooBook`: it owns the model,
its name parser, and its :attr:`~NooBook.viz` / :attr:`~NooBook.extract`
accessors. Only the eponymous :class:`NooBook` is public; everything else
(the accessors, the name helpers) is internal.
"""

from noobfriend.navigation.noobook._core import NooBook

__all__ = ["NooBook"]
