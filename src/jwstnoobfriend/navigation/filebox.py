from pydantic import BaseModel, field_validator, model_validator, computed_field

from jwstnoobfriend.navigation.jwstinfo import JwstInfo
from jwstnoobfriend.navigation.footprint import FootPrint

class FileBox(BaseModel):
    files: list[JwstInfo]
    
    
    
