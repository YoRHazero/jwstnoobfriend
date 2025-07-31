from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
    validate_call,
    computed_field,
    FilePath,
)
from pathlib import Path
from shapely.geometry import Polygon, Point
from typing import Any, Iterable, Self
from astropy.coordinates import SkyCoord
from jwstnoobfriend.utils import log
from jwstnoobfriend.navigation._cache import (
    _open_and_cache_datamodel,
    _open_and_cache_datamodel_async,
    _open_and_cache_wcs,
    _open_and_cache_wcs_async,
)
import array
import plotly.graph_objects as go
__all__ = ["FootPrint", "CompoundFootPrint"]

logger = log.getLogger(__name__)


class FootPrint(BaseModel):
    vertices: list[tuple[float, float]]
    vertex_marker: list | None = None

    @field_validator("vertices", mode="before")
    @classmethod
    def convert_from_skycoords(cls, values: Any):
        """Additionally allow SkyCoord objects to be passed as vertices."""
        if not hasattr(values, "__iter__") or not hasattr(values, "__len__"):
            raise TypeError("Vertices must be in a sequence")

        first_value = values[0]
        if isinstance(first_value, SkyCoord):
            return [(coord.ra.deg, coord.dec.deg) for coord in values]
        else:
            return values

    @model_validator(mode="after")
    def validate_polygon(self):
        if len(self.vertices) != 4:
            raise ValueError("Currently only 4 vertices are supported")
        if self.vertex_marker is not None and len(self.vertex_marker) != 4:
            raise ValueError(
                "If vertex_marker is provided, it must have exactly 4 elements"
            )
        polygon = Polygon(self.vertices)
        if polygon.is_valid:
            return self
        else:
            self.vertices[1], self.vertices[2] = (
                self.vertices[2],
                self.vertices[1],
            )  # switch the 2nd and 3rd vertices
            if self.vertex_marker is not None:
                self.vertex_marker[1], self.vertex_marker[2] = (
                    self.vertex_marker[2],
                    self.vertex_marker[1],
                )
            polygon = Polygon(self.vertices)
            if polygon.is_valid:
                return self
            else:
                raise ValueError(
                    "Invalid vertices, cannot form a valid polygon, \
                    check whether the four vertices are on the same line"
                )

    @computed_field
    @property
    def center(self) -> tuple[float, ...]:
        """Returns the center of the footprint (geometric centroid)."""
        return self.polygon.centroid.coords[0]

    @computed_field
    @property
    def area(self) -> float:
        """Returns the area of the footprint. In the units of the coordinates provided (e.g., degrees, if RA/Dec is provided)."""
        return self.polygon.area

    @computed_field
    @property
    def radius(self) -> float:
        """Returns the radius of the footprint, defined as the distance from the center to the furthest vertex."""
        return max(
            Point(self.center).distance(Point(vertex)) for vertex in self.vertices
        )

    @property
    def polygon(self) -> Polygon:
        """Returns the shapely.geometry.Polygon object representing the footprint."""
        return Polygon(self.vertices)

    @property
    def vertices_as_skycoords(self) -> list[SkyCoord]:
        """Returns the vertices as SkyCoord objects."""
        return [
            SkyCoord(ra=coord[0], dec=coord[1], unit="deg") for coord in self.vertices
        ]
        
    @property
    def vertices_for_plot(self) -> tuple[array.array, array.array]:
        """Returns the vertices as two separate arrays for plotting."""
        return self.polygon.exterior.xy
    
    @classmethod
    @validate_call
    def new(cls, file_path: FilePath | str) -> Self:
        """Creates a FootPrint object from a file containing vertices.

        The file should contain wcs information, or return None if the file does not have a WCS object assigned.
        """
        import numpy as np

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        try:
            model = _open_and_cache_datamodel(file_path)
            wcs = _open_and_cache_wcs(file_path)
            data_shape = model.data.shape
            pupil = model.meta.instrument.pupil

            if pupil == "CLEAR":
                vertices_marker = [
                    (0, 0),
                    (data_shape[1] - 1, 0),
                    (data_shape[1] - 1, data_shape[0] - 1),
                    (0, data_shape[0] - 1),
                ]
                transform = wcs.get_transform("detector", "world")
                vertices_marker_array = np.array(vertices_marker)
                vertices = transform(
                    vertices_marker_array[:, 0], vertices_marker_array[:, 1]
                )
                vertices = np.array(vertices).T
                return cls(vertices=vertices, vertex_marker=vertices_marker)
            elif pupil == "GRISMR" or pupil == "GRISMC":
                vertices_marker = [
                    (0, 0),
                    (data_shape[1] - 1, 0),
                    (data_shape[1] - 1, data_shape[0] - 1),
                    (0, data_shape[0] - 1),
                ]
                transform = wcs.get_transform("detector", "world")
                vertices_marker_array = np.array(vertices_marker)
                vertices = transform(
                    vertices_marker_array[:, 0],
                    vertices_marker_array[:, 1],
                    [1] * 4,
                    [1] * 4,
                )
                vertices = np.array(vertices).T[
                    :, :2
                ]  # Only take the first two columns (RA, Dec)
                return cls(vertices=vertices, vertex_marker=vertices_marker)
        except Exception as e:
            logger.warning(
                f"Failed to create FootPrint from file {file_path}: {e}. Return None\
                If this is not expected, please check whether the file is assigned a WCS object."
            )
            return None

    @classmethod
    @validate_call
    async def _new_async(cls, file_path: FilePath | str) -> Self:
        import numpy as np

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        try:
            model = await _open_and_cache_datamodel_async(file_path)
            wcs = await _open_and_cache_wcs_async(file_path)
            data_shape = model.data.shape
            pupil = model.meta.instrument.pupil

            if pupil == "CLEAR":
                vertices_marker = [
                    (0, 0),
                    (data_shape[1] - 1, 0),
                    (data_shape[1] - 1, data_shape[0] - 1),
                    (0, data_shape[0] - 1),
                ]
                transform = wcs.get_transform("detector", "world")
                vertices_marker_array = np.array(vertices_marker)
                vertices = transform(
                    vertices_marker_array[:, 0], vertices_marker_array[:, 1]
                )
                vertices = np.array(vertices).T
                return cls(vertices=vertices, vertex_marker=vertices_marker)
            elif pupil == "GRISMR" or pupil == "GRISMC":
                vertices_marker = [
                    (0, 0),
                    (data_shape[1] - 1, 0),
                    (data_shape[1] - 1, data_shape[0] - 1),
                    (0, data_shape[0] - 1),
                ]
                transform = wcs.get_transform("detector", "world")
                vertices_marker_array = np.array(vertices_marker)
                vertices = transform(
                    vertices_marker_array[:, 0],
                    vertices_marker_array[:, 1],
                    [1] * 4,
                    [1] * 4,
                )
                vertices = np.array(vertices).T[
                    :, :2
                ]  # Only take the first two columns (RA, Dec)
                return cls(vertices=vertices, vertex_marker=vertices_marker)
        except Exception as e:
            logger.warning(
                f"Failed to create FootPrint from file {file_path}: {e}. Return None\
                If this is not expected, please check whether the file is assigned a WCS object."
            )
            return None

    def add_trace_in_sky(self, 
                            fig: go.Figure, 
                            color: str = 'teal', 
                            point_hovertemplate: str | None = None,
                            fp_hovertemplate: str | None = None) -> go.Figure:
        """
        Adds the footprint to a Plotly figure as a scattergeo trace.
        
        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to which the footprint will be added.
        color : str, optional
            The color of the footprint line, by default 'teal'.
        point_hovertemplate : str, optional
            The hover template for the points, if None, a default template will be used.
        fp_hovertemplate : str, optional
            The footprint-level information to show in the hover template, if None, no additional information will be shown.
            
        Returns
        -------
        go.Figure
            The Plotly figure with the footprint added as a trace.
        """
        ra_arr, dec_arr = self.vertices_for_plot
        ra_arr = ra_arr.tolist()
        dec_arr = dec_arr.tolist()
        if point_hovertemplate is None:
            point_hovertemplate = '<b>Pixel: %{text}</b><br>' + \
                            'RA: %{lon:.3f}<br>' + \
                            'Dec: %{lat:.3f}<br>' + \
                            '<extra></extra>'
        if fp_hovertemplate is not None:
            hovertemplate = point_hovertemplate + fp_hovertemplate
        else:
            hovertemplate = point_hovertemplate
                                  
        fig.add_trace(
            go.Scattergeo(
                lat=dec_arr,
                lon=ra_arr,
                mode="lines",
                line=dict(color=color, dash='dash'),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scattergeo(
                lat=dec_arr[:-1],
                lon=ra_arr[:-1],
                mode='markers',
                showlegend=False,
                text=self.vertex_marker if self.vertex_marker else [str(i) for i in range(len(ra_arr[:-1]))],
                hovertemplate=hovertemplate,
            )
        )
        return fig
    
    def add_trace_in_cartesian(self, 
                            fig: go.Figure, 
                            color: str = 'teal', 
                            point_hovertemplate: str | None = None,
                            fp_hovertemplate: str | None = None) -> go.Figure:
        """
        Adds the footprint to a Plotly figure as a scatter trace in Cartesian coordinates.
        
        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to which the footprint will be added.
        color : str, optional
            The color of the footprint line, by default 'teal'.
        point_hovertemplate : str, optional
            The hover template for the points, if None, a default template will be used.
        fp_hovertemplate : str, optional
            The footprint-level information to show in the hover template, if None, no additional information will be shown.
        
        Returns
        -------
        go.Figure
            The Plotly figure with the footprint added as a trace.
        """
        ra_arr, dec_arr = self.vertices_for_plot
        ra_arr = ra_arr.tolist()
        dec_arr = dec_arr.tolist()
        if point_hovertemplate is None:
            point_hovertemplate = '<b>Pixel: %{text}</b><br>' + \
                            'RA: %{x:.3f}<br>' + \
                            'Dec: %{y:.3f}<br>' + \
                            '<extra></extra>'
        if fp_hovertemplate is not None:
            hovertemplate = point_hovertemplate + fp_hovertemplate
        else:
            hovertemplate = point_hovertemplate
        
        fig.add_trace(
            go.Scatter(
                x=ra_arr,
                y=dec_arr,
                mode="lines",
                line=dict(color=color, dash='dash'),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ra_arr[:-1],
                y=dec_arr[:-1],
                mode='markers',
                showlegend=False,
                text=self.vertex_marker if self.vertex_marker else [str(i) for i in range(len(ra_arr[:-1]))],
                hovertemplate=hovertemplate,
            )
        )
        return fig


class CompoundFootPrint(FootPrint):
    """
    A compound footprint that can be constructed from multiple footprints or vertices.
    It is suggested to only pass the footprints or vertices, not both, and currently the best practice is to pass footprints.
    The usage of this class is still under development.

    To do
    - Support passing vertices and vertex_marker, and check if they match the union of footprints.
    - Add property and method to utilize the union of footprints.
    """

    footprints: list[FootPrint] | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_footprints(cls, data: Any):
        footprint_iterable = data.get("footprints", None)
        vertex_list = data.get("vertices", None)
        vertex_marker_list = data.get('vertex_marker', None)

        ## At least one of footprints or vertices must be provided
        if footprint_iterable is None and vertex_list is None:
            raise ValueError("Either 'footprints' or 'vertices' must be provided")

        ## If footprints are provided, use polygon to get the union of all footprints
        if footprint_iterable is not None:
            if not isinstance(footprint_iterable, Iterable):
                raise TypeError(
                    "'footprints' must be an iterable of FootPrint objects, recommend to use a list"
                )

            ## check if they are overlapped
            try:
                first_footprint = next(iter(footprint_iterable))
                result_polygon = first_footprint.polygon
            except StopIteration:
                raise ValueError("'footprints' cannot be empty")
            for footprint in footprint_iterable:
                if not isinstance(footprint, FootPrint):
                    raise TypeError(
                        "All elements in 'footprints' must be FootPrint objects"
                    )
                if not result_polygon.intersects(footprint.polygon):
                    raise ValueError(
                        "Footprints in 'footprints' must intersect with each other"
                    )

                result_polygon = result_polygon.union(footprint.polygon)

            ## get the vertices of the union of footprints
            polygon_vertices = list(result_polygon.exterior.coords)

            ## if vertex_list is provided simultaneously, check if they match. Note passing the matching check does not mean the vertices order are correct.
            if vertex_list is not None:
                if sorted(vertex_list) != sorted(polygon_vertices):
                    logger.warning(
                        "The provided vertices do not match the vertices of the union of footprints, \
                        the vertices of the union of footprints will be used and the provided vertices and vertex_marker will be ignored"
                    )
                    data["vertex_marker"] = None
            data["vertices"] = polygon_vertices
        return data

    @model_validator(mode="after")
    def validate_polygon(self):
        polygon = Polygon(self.vertices)
        if not polygon.is_valid:
            raise ValueError(
                "Invalid vertices, cannot form a valid polygon, \
                check whether the sequence of vertices"
            )

        if self.vertex_marker is not None and len(self.vertex_marker) != len(
            self.vertices
        ):
            raise ValueError(
                "If vertex_marker is provided, it must have the same number of elements as vertices\
                , currently {} vertices and {} vertex_marker".format(
                    len(self.vertices), len(self.vertex_marker)
                )
            )

        return self
