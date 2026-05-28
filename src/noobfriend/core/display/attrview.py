"""Notebook-friendly attribute-style viewer for nested mappings and sequences."""

import uuid
from collections.abc import Mapping, Sequence
from html import escape
from typing import Any

from rich.text import Text
from rich.tree import Tree

_NOT_REAL_SEQUENCE = (str, bytes, bytearray)
_LEAF_REPR_LIMIT = 200


_CSS = """\
.attrview {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 13px;
    line-height: 1.5;
    max-width: 100%;
}
.attrview details,
.attrview .leaf {
    margin-left: 1em;
}
.attrview summary {
    cursor: pointer;
    user-select: none;
}
.attrview .key {
    color: #795e26;
    font-weight: 600;
}
.attrview .meta {
    color: #888;
    margin-left: 0.4em;
}
.attrview code {
    background: transparent;
    color: #0f766e;
}
.attrview .attrview-search {
    width: 100%;
    box-sizing: border-box;
    margin-bottom: 0.5em;
    padding: 0.3em 0.5em;
    font-family: inherit;
    font-size: inherit;
    border: 1px solid #ccc;
    border-radius: 4px;
}
.attrview [data-attrview-hidden] {
    display: none;
}
"""


_JS = """\
(function () {
    function visit(node, query) {
        if (node.classList && node.classList.contains('leaf')) {
            const match = node.textContent.toLowerCase().includes(query);
            if (!match) node.setAttribute('data-attrview-hidden', '');
            return match;
        }
        if (node.tagName === 'DETAILS') {
            const summary = node.querySelector(':scope > summary');
            const summaryMatch = summary && summary.textContent.toLowerCase().includes(query);
            if (summaryMatch) {
                // Whole subtree is matched by inheritance: open this node and
                // skip recursion so descendants stay visible at their defaults.
                node.open = true;
                return true;
            }
            let childMatch = false;
            node.querySelectorAll(':scope > details, :scope > .leaf').forEach(function (child) {
                if (visit(child, query)) childMatch = true;
            });
            if (childMatch) node.open = true;
            else node.setAttribute('data-attrview-hidden', '');
            return childMatch;
        }
        return false;
    }

    function filter(container) {
        const input = container.querySelector('.attrview-search');
        const tree = container.querySelector('.attrview-tree');
        const query = input.value.trim().toLowerCase();

        tree.querySelectorAll('[data-attrview-hidden]').forEach(function (el) {
            el.removeAttribute('data-attrview-hidden');
        });

        if (!query) {
            tree.querySelectorAll('details').forEach(function (d, i) {
                d.open = (i === 0);
            });
            return;
        }

        tree.querySelectorAll(':scope > details, :scope > .leaf').forEach(function (node) {
            visit(node, query);
        });
    }

    document.querySelectorAll('.attrview:not([data-attrview-ready])').forEach(function (container) {
        container.setAttribute('data-attrview-ready', '');
        const input = container.querySelector('.attrview-search');
        if (input) input.addEventListener('input', function () { filter(container); });
    });
})();
"""


def _is_listlike(value: Any) -> bool:
    """Return ``True`` if ``value`` should be traversed as an indexed sequence.

    Strings, bytes, and bytearrays are excluded so they render as leaves
    instead of being expanded character-by-character.
    """
    return isinstance(value, Sequence) and not isinstance(value, _NOT_REAL_SEQUENCE)


def _maybe_wrap(value: Any) -> Any:
    """Return ``AttrView(value)`` for nested containers; otherwise ``value``."""
    if isinstance(value, Mapping) or _is_listlike(value):
        return AttrView(value)
    return value


class AttrView:
    """Attribute-style viewer for nested mappings/sequences with a notebook UI.

    Wraps a JSON-like tree of dicts and lists so that callers can navigate it
    via attribute access in the REPL and explore it interactively in Jupyter
    via a searchable, foldable HTML widget.

    Parameters
    ----------
    node : Mapping or Sequence
        The nested structure to wrap. Inner :class:`~collections.abc.Mapping`
        and non-string :class:`~collections.abc.Sequence` values are wrapped
        on access; all other values are returned as-is.

    Notes
    -----
    Attribute access (``view.key``) is only meaningful when the wrapped node
    is a mapping. For lists or tuples use index access (``view[0]``); the
    returned value is itself wrapped, so ``view.detectors[0].name`` chains
    cleanly.

    The HTML representation is self-contained: each render injects its own
    scoped CSS plus a small JavaScript snippet that powers the search box.
    No external JavaScript dependencies and no Jupyter widgets are required.

    Terminal callers get a coloured :class:`rich.tree.Tree` automatically via
    the ``__rich__`` protocol; print the view with any rich-aware function::

        from noobfriend.core import console, AttrView
        console.print(AttrView(data))

    Examples
    --------
    >>> view = AttrView({"detector": {"name": "NRS1", "gain": 1.2}})
    >>> view.detector.name
    'NRS1'
    """

    __slots__ = ("_node",)

    def __init__(self, node: Mapping[Any, Any] | Sequence[Any]) -> None:
        """Store ``node`` as the wrapped tree; see the class docstring for parameters."""
        self._node = node

    def __getattr__(self, name: str) -> Any:
        """Look up ``name`` as a key in the wrapped mapping and wrap nested containers."""
        node = self._node
        if not isinstance(node, Mapping):
            raise AttributeError(
                f"attribute access requires a Mapping; wrapped {type(node).__name__}"
            )
        try:
            value = node[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return _maybe_wrap(value)

    def __getitem__(self, key: Any) -> Any:
        """Index into the wrapped node and wrap nested containers in the result."""
        return _maybe_wrap(self._node[key])

    def __dir__(self) -> list[str]:
        """Augment default ``dir`` with mapping keys that are valid identifiers (for tab completion)."""
        base = list(super().__dir__())
        if isinstance(self._node, Mapping):
            base.extend(
                k for k in self._node if isinstance(k, str) and k.isidentifier()
            )
        return base

    def __repr__(self) -> str:
        """Return ``AttrView(<repr of wrapped node>)``."""
        return f"AttrView({self._node!r})"

    def __rich__(self) -> Tree:
        """Render the wrapped node as a :class:`rich.tree.Tree` for terminal output.

        Invoked automatically by any rich-aware printer (``console.print``,
        ``rich.print``, ``rich.inspect``...). Plain ``print`` and the
        interactive REPL still use :meth:`__repr__`.
        """
        tree = Tree(self._rich_label("root", self._node))
        self._populate_rich(tree, self._node)
        return tree

    @classmethod
    def _rich_label(cls, key: Any, value: Any) -> Text:
        """Build a coloured ``Text`` label for a ``(key, value)`` tree node."""
        label = Text(str(key), style="yellow")
        if isinstance(value, Mapping):
            label.append(f"  dict[{len(value)}]", style="dim")
        elif _is_listlike(value):
            label.append(f"  {type(value).__name__}[{len(value)}]", style="dim")
        else:
            label.append(": ")
            repr_str = repr(value)
            if len(repr_str) > _LEAF_REPR_LIMIT:
                repr_str = repr_str[:_LEAF_REPR_LIMIT] + "…"
            label.append(repr_str, style="cyan")
        return label

    @classmethod
    def _populate_rich(cls, tree: Tree, value: Any) -> None:
        """Recursively add ``value``'s entries as branches under ``tree``."""
        if isinstance(value, Mapping):
            items: Any = value.items()
        elif _is_listlike(value):
            items = enumerate(value)
        else:
            return
        for k, v in items:
            if isinstance(v, Mapping) or _is_listlike(v):
                child = tree.add(cls._rich_label(k, v))
                cls._populate_rich(child, v)
            else:
                tree.add(cls._rich_label(k, v))

    def _repr_html_(self) -> str:
        """Return the HTML rendered by Jupyter front-ends."""
        uid = f"attrview-{uuid.uuid4().hex[:8]}"
        tree_html = self._render_item("root", self._node, open_=True)
        return (
            f"<style>{_CSS}</style>"
            f'<div class="attrview" id="{uid}">'
            f'<input class="attrview-search" type="search" placeholder="search...">'
            f'<div class="attrview-tree">{tree_html}</div>'
            f"</div>"
            f"<script>{_JS}</script>"
        )

    @classmethod
    def _render_item(cls, key: Any, value: Any, *, open_: bool = False) -> str:
        """Render a single ``(key, value)`` entry into the tree's HTML."""
        key_html = escape(str(key))
        open_attr = " open" if open_ else ""

        if isinstance(value, Mapping):
            children = "".join(cls._render_item(k, v) for k, v in value.items())
            return (
                f"<details{open_attr}>"
                f'<summary><span class="key">{key_html}</span>'
                f'<span class="meta">dict[{len(value)}]</span></summary>'
                f"{children}"
                f"</details>"
            )

        if _is_listlike(value):
            children = "".join(cls._render_item(i, v) for i, v in enumerate(value))
            return (
                f"<details{open_attr}>"
                f'<summary><span class="key">{key_html}</span>'
                f'<span class="meta">{type(value).__name__}[{len(value)}]</span></summary>'
                f"{children}"
                f"</details>"
            )

        repr_str = repr(value)
        if len(repr_str) > _LEAF_REPR_LIMIT:
            repr_str = repr_str[:_LEAF_REPR_LIMIT] + "…"
        return (
            f'<div class="leaf">'
            f'<span class="key">{key_html}</span>: '
            f"<code>{escape(repr_str)}</code>"
            f"</div>"
        )
