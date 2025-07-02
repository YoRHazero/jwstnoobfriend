from pydantic import BaseModel, field_validator, model_validator, computed_field, FilePath, validate_call
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

    @computed_field
    @property
    def center(self) ->  tuple[float, ...]:
        """ Returns the center of the footprint (geometric centroid).
        """
        return self.polygon.centroid.coords[0]
    
    @computed_field
    @property
    def area(self) -> float:
        """ Returns the area of the footprint. In the units of the coordinates provided (e.g., degrees, if RA/Dec is provided).
        """
        return self.polygon.area
    
    @computed_field
    @property
    def radius(self) -> float:
        """ Returns the radius of the footprint, defined as the distance from the center to the furthest vertex.
        """
        return max(Point(self.center).distance(Point(vertex)) for vertex in self.vertices)
    

    @property
    def polygon(self) -> Polygon:
        """ Returns the shapely.geometry.Polygon object representing the footprint.
        """
        return Polygon(self.vertices)

    @property
    def vertices_as_skycoords(self) -> list[SkyCoord]:
        """ Returns the vertices as SkyCoord objects.
        """
        return [SkyCoord(ra=coord[0], dec=coord[1], unit='deg') for coord in self.vertices]
    
    @classmethod
    @validate_call
    def from_file(cls, file_path: FilePath) -> 'FootPrint':
        """ Creates a FootPrint object from a file containing vertices.
        
        The file should contain wcs
        """
        from jwst import datamodels as dm
        import numpy as np
        try:
            with dm.open(file_path) as model:
                wcs = model.meta.wcs
                data_shape = model.data.shape
                pupil = model.meta.instrument.pupil
            
            if pupil == "CLEAR":
                vertices_marker = [(0, 0), (data_shape[1] - 1, 0), 
                                (data_shape[1] - 1, data_shape[0] - 1), (0, data_shape[0] - 1)]
                transform = wcs.get_transform('detector', 'world')
                vertices_marker_array = np.array(vertices_marker)
                vertices = transform(vertices_marker_array[:, 0], vertices_marker_array[:, 1])
                vertices = np.array(vertices).T
                return cls(
                    vertices=vertices,
                    vertex_marker=vertices_marker
                    )
            elif pupil == "GRISMR" or pupil == "GRISMC":
                vertices_marker = [(0, 0), (data_shape[1] - 1, 0), 
                                (data_shape[1] - 1, data_shape[0] - 1), (0, data_shape[0] - 1)]
                transform = wcs.get_transform('detector', 'world')
                vertices_marker_array = np.array(vertices_marker)
                vertices = transform(vertices_marker_array[:, 0], vertices_marker_array[:, 1], [1] * 4, [1] * 4)
                vertices = np.array(vertices).T[:, :2]  # Only take the first two columns (RA, Dec)
                return cls(
                    vertices=vertices,
                    vertex_marker=vertices_marker
                )
        except Exception as e:
            logger.warning(f"Failed to create FootPrint from file {file_path}: {e}. Return None\
                If this is not expected, please check whether the file is assigned a WCS object.")
            return None

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
