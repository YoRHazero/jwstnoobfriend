"""The :class:`BookViz` accessor: plotting sugar bound to one NooBook.

Reached as :attr:`NooBook.viz <noobfriend.navigation.noobook._core.NooBook.viz>`.
Each method is a thin delegate to :mod:`noobfriend.core.display.plot` that
supplies the book's own arrays; the (heavy) Bokeh import stays inside the
methods so importing the package never pulls it in.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from noobfriend.navigation.noobook._core import NooBook


class BookViz:
    """Plotting sugar bound to one :class:`NooBook`.

    Parameters
    ----------
    book : NooBook
        The book whose data these plots draw.
    """

    def __init__(self, book: "NooBook") -> None:
        """See the class docstring for parameters."""
        self._book = book

    def imshow(self, **kwargs: Any) -> Any:
        """Show the book's ``SCI`` array as an interactive Bokeh image.

        Parameters
        ----------
        **kwargs
            Forwarded to :func:`noobfriend.core.display.plot.imshow`.

        Returns
        -------
        Any
            The Bokeh model returned by ``imshow``.
        """
        from noobfriend.core.display.plot import imshow

        return imshow(self._book.data, **kwargs)
