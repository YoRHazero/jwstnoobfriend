import pytest
from pydantic import ValidationError
from jwstnoobfriend.navigation import FootPrint
import logging


class TestFootPrint:
    def test_valid_vertices(self):
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        footprint = FootPrint(vertices=vertices) # type: ignore
        assert footprint.polygon.is_valid
        assert footprint.vertices == vertices

    def test_invalid_vertices(self):
        with pytest.raises(ValidationError):
            FootPrint(vertices=[(0, 0), (1, 1), (1, 0)])
            
    def test_invalid_polygon(self):
        vertices = [(0, 0), (1, 1), (1, 0), (0, 1)]
        footprint = FootPrint(vertices=vertices) # type: ignore
        assert footprint.polygon.is_valid
        assert footprint.vertices != vertices
    
