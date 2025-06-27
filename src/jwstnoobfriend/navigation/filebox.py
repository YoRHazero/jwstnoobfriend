from pydantic import BaseModel, Field, field_validator
from typing import ClassVar, Annotated
from functools import lru_cache
from gwcs.wcs import WCS
import re

class JwstInfoBase(BaseModel):
    
    basename_pattern: ClassVar[str] = r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}_[^_]+"
    """ The basename pattern for JWST files, used for validation. """
    
    basename: Annotated[str, Field(
        description="JWST nameing convention is jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>_<filetype>.fits \
                (ref: https://jwst-pipeline.readthedocs.io/en/latest/jwst/data_products/file_naming.html) \
                Here the basename is jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>",
        pattern = r"jw\d{5}\d{3}\d{3}_\d{5}_\d{5}_[^_]+"
    )]
    """jw<ppppp><ooo><vvv>_<gg><s><aa>_<eeeee>_<detector>"""
    @field_validator('basename', mode = 'after')
    @classmethod
    def extract_basename(cls, value: str) -> str:
        """Extracts the basename from the full filename."""
        return re.match(cls.basename_pattern, value).group()
        
