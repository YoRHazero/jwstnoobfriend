from pydantic import BaseModel, field_validator, model_validator, ValidationError
from shapely.geometry import Polygon, Point, LineString
from typing import Any, Literal, Iterable
from astropy.coordinates import SkyCoord
from jwstnoobfriend.utils import log

__all__ = ['FootPrint', 'CompoundFootPrint']

logger = log.getLogger(__name__)

class FootPrint(BaseModel):
    vertices: list[tuple[float, float]]
    vertex_marker: list | None = None
    
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
                
    @model_validator(mode='after')
    def validate_polygon(self):
        if len(self.vertices) != 4:
            raise ValueError("Currently only 4 vertices are supported")
        if self.vertex_marker is not None and len(self.vertex_marker) != 4:
            raise ValueError("If vertex_marker is provided, it must have exactly 4 elements")
        polygon = Polygon(self.vertices)
        if polygon.is_valid:
            return self
        else:
            self.vertices[1], self.vertices[2] = self.vertices[2], self.vertices[1] # switch the 2nd and 3rd vertices
            if self.vertex_marker is not None:
                self.vertex_marker[1], self.vertex_marker[2] = self.vertex_marker[2], self.vertex_marker[1]
            polygon = Polygon(self.vertices)
            if polygon.is_valid:
                return self
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
    

    @property
    def vertices_as_skycoords(self) -> list[SkyCoord]:
        """ Returns the vertices as SkyCoord objects.
        """
        return [SkyCoord(ra=coord[0], dec=coord[1], unit='deg') for coord in self.vertices]


class CompoundFootPrint(FootPrint):
    
    footprints: list[FootPrint] | None = None
    
    @model_validator(mode='before')
    @classmethod
    def validate_footprints(cls, data: Any):
        footprint_iterable = data.get('footprints', None)
        vertex_list = data.get('vertices', None)
        vertex_marker_list = data.get('vertex_marker', None)
        
        if footprint_iterable is None and vertex_list is None:
            raise ValueError("Either 'footprints' or 'vertices' must be provided")
        
        if footprint_iterable is not None:
            if not isinstance(footprint_iterable, Iterable):
                raise TypeError("'footprints' must be an iterable of FootPrint objects, recommend to use a list")
            try:
                first_footprint = next(iter(footprint_iterable))
                result_polygon = first_footprint.polygon
            except StopIteration:
                raise ValueError("'footprints' cannot be empty")
            
            for footprint in footprint_iterable:
                if not isinstance(footprint, FootPrint):
                    raise TypeError("All elements in 'footprints' must be FootPrint objects")
                
                if not result_polygon.intersects(footprint.polygon):
                    raise ValueError("Footprints in 'footprints' must intersect with each other")
                
                result_polygon = result_polygon.union(footprint.polygon)
                
            polygon_vertices = list(result_polygon.exterior.coords)
            
            if vertex_list is not None:
                if sorted(vertex_list) != sorted(polygon_vertices):
                    logger.warning("The provided vertices do not match the vertices of the union of footprints, \
                        the vertices of the union of footprints will be used and the provided vertices and vertex_marker will be ignored")
                    data['vertex_marker'] = None
            data['vertices'] = polygon_vertices
        return data
        
    @model_validator(mode='after')
    def validate_polygon(self):
        polygon = Polygon(self.vertices)
        if not polygon.is_valid:
            raise ValueError("Invalid vertices, cannot form a valid polygon, \
                check whether the sequence of vertices")
        
        if self.vertex_marker is not None and len(self.vertex_marker) != len(self.vertices):
            raise ValueError("If vertex_marker is provided, it must have the same number of elements as vertices\
                , currently {} vertices and {} vertex_marker".format(len(self.vertices), len(self.vertex_marker)))
            
        return self
