"""Line declarations.

:class:`NoobLine` declares one emission or absorption component: where it sits
(``rest`` / ``obs`` and ``unit``), what it is (component, contribution,
profile) and how its parameters are set (:class:`LineRule`,
:class:`LineKernel`). Relations go only from a derived line to its base.
"""

from noobfriend.inference.spectrum.line._line import (
    RESERVED_NAMES,
    ComponentName,
    ContributionName,
    NoobLine,
    ProfileName,
    check_name,
)
from noobfriend.inference.spectrum.line._rules import LineKernel, LineRule

__all__ = [
    "RESERVED_NAMES",
    "ComponentName",
    "ContributionName",
    "LineKernel",
    "LineRule",
    "NoobLine",
    "ProfileName",
    "check_name",
]
