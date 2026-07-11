"""``TransformWCS``: the minimal WCS surface the rest of the package consumes.

Almost every WCS consumer in noobfriend only ever asks two questions --
"which frames exist?" and "give me the transform between these two" -- so
that pair *is* the package-wide WCS contract. Anything answering it is a
valid ``wcs`` argument: a :class:`gwcs.WCS`, a compiled
:class:`~noobfriend.core.wcs.NoobWCS`, or a purpose-built adapter (e.g.
:class:`noobfriend.core.io.GrizliCutout` or the tangent-plane adapter in
:mod:`noobfriend.extraction._wcs`). Code that needs more than this
protocol (bounding boxes, ASDF serialisation, astropy coordinate objects)
must ask for a real gwcs object explicitly.
"""

from typing import Any, Callable, Protocol, Sequence


class TransformWCS(Protocol):
    """A multi-frame WCS answering the frame/transform protocol.

    Implementations expose named coordinate frames and return a callable
    transform between any two of them. Transforms accept scalars (returning
    floats) or broadcastable numeric arrays (returning ``float64`` arrays),
    like a gwcs transform.
    """

    @property
    def available_frames(self) -> Sequence[str]:
        """Coordinate-frame names in pipeline order."""
        ...

    def get_transform(self, from_frame: str, to_frame: str) -> Callable[..., Any]:
        """Return the transform mapping ``from_frame`` to ``to_frame``."""
        ...
