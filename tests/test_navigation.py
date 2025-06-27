import pytest
from pydantic import ValidationError
from jwstnoobfriend.navigation import FootPrint, CompoundFootPrint
from jwstnoobfriend.navigation import JwstInfoBase

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
        
    def test_valid_vertex_with_marker(self):
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        vertex_marker = ['A', 'B', 'C', 'D']
        footprint = FootPrint(vertices=vertices, vertex_marker=vertex_marker) # type: ignore
        assert footprint.vertex_marker == vertex_marker
        
    def test_invalid_vertex_with_marker(self):
        vertices = [(0, 0), (1, 1), (1, 0), (0, 1)]
        vertex_marker = ['A', 'B', 'C', 'D']
        footprint = FootPrint(vertices=vertices, vertex_marker=vertex_marker) # type: ignore
        expected_vertex_marker = vertex_marker.copy()
        expected_vertex_marker[1], expected_vertex_marker[2] = expected_vertex_marker[2], expected_vertex_marker[1]
        assert footprint.vertex_marker == expected_vertex_marker
    

class TestCompoundFootPrint:
    def test_valid_footprints(self):
        footprints = [
            FootPrint(vertices=[(0, 0), (2, 0), (2, 2), (0, 2)]), # type: ignore
            FootPrint(vertices=[(1, 1), (3, 1), (3, 3), (1, 3)]) # type: ignore
        ]
        compound_footprint = CompoundFootPrint(footprints=footprints) # type: ignore
        assert len(compound_footprint.footprints) == 2
        assert all(fp.polygon.is_valid for fp in compound_footprint.footprints)
        
    def test_disjoint_footprints(self):
        footprints = [
            FootPrint(vertices=[(0, 0), (2, 0), (2, 2), (0, 2)]), # type: ignore
            FootPrint(vertices=[(3, 3), (5, 3), (5, 5), (3, 5)]) # type: ignore
        ]
        with pytest.raises(ValidationError):
            CompoundFootPrint(footprints=footprints)
            
    def test_compound_footprint_with_vertices(self):
        vertices = [(0, 0), (2, 0), (2, 2), (0, 2)]
        vertex_marker = ['A', 'B', 'C', 'D']
        compound_footprint = CompoundFootPrint(vertices=vertices, vertex_marker=vertex_marker) # type: ignore
        assert compound_footprint.polygon.is_valid
        assert compound_footprint.vertices == vertices
        assert compound_footprint.vertex_marker == vertex_marker
        
    def test_compound_footprint_invalid_vertices(self):
        vertices = [(0, 0), (1, 1), (1, 0), (0, 1)]
        vertex_marker = ['A', 'B', 'C', 'D']
        with pytest.raises(ValidationError):
            CompoundFootPrint(vertices=vertices, vertex_marker=vertex_marker)
    