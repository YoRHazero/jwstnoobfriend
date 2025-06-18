from pydantic import BaseModel, field_validator, ValidationError
from shapely.geometry import Polygon, Point, LineString
from typing import Any, Literal
from astropy.coordinates import SkyCoord

class FootPrint(BaseModel):
    vertices: list[tuple[float, float]]
    
    @field_validator('vertices', mode='before')
    @classmethod
    def convert_from_skycoords(cls, values: Any):
        """ Additionally allow SkyCoord objects to be passed as vertices.
        """
        if not hasattr(values, '__iter__') or not hasattr(values, '__len__'):
            raise TypeError("Vertices must be in a sequence")
        
        first_value = values[0]
        if isinstance(first_value, SkyCoord):
            return [(coord.ra.deg, coord.dec.deg) for coord in values]
        else:
            return values
    
    @field_validator('vertices', mode='after')
    @classmethod
    def valid_vertices(cls, values:list[tuple[float, float]]):
        """ Validate that the number of vertices is 4 and that they form a simple polygon.
        """
        if len(values) != 4:
            raise ValueError("Currently only 4 vertices are supported")
        polygon = Polygon(values)
        if polygon.is_valid:
            return values
        else:
            # switch the 2nd and 3rd vertices
            values[1], values[2] = values[2], values[1]
            polygon = Polygon(values)
            if polygon.is_valid:
                return values
            else:
                raise ValueError("Invalid vertices, cannot form a valid polygon, \
                    check whether the four vertices are on the same line")

    @property
    def polygon(self) -> Polygon:
        """ Returns the shapely.geometry.Polygon object representing the footprint.
        """
        return Polygon(self.vertices)
    
    @property
    def center(self) ->  tuple[float, ...]:
        """ Returns the center of the footprint (geometric centroid).
        """
        return self.polygon.centroid.coords[0]
    