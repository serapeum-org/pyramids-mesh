"""Shared pytest fixtures for the pyramids-mesh test suite.

The `import pyramids` below is order-critical and must stay the first import in
this module. The comment on that line carries the rationale.
"""

# Import pyramids FIRST — before anything else, and keep it first.
# The osgeo bindings live inside the pyramids wheel; importing pyramids is what
# puts `_vendor/osgeo` on sys.path. Test modules that do `from osgeo import gdal`
# at module scope, before they import pyramids_mesh, rely on this line to make
# those imports resolve. Nothing installs a top-level osgeo, so deleting this
# line, or sorting it below an osgeo import, breaks collection for the whole
# directory.
import pyramids  # noqa: F401  # isort:skip

import pytest


@pytest.fixture(scope="session")
def unit_square_mesh() -> tuple:
    """Two triangles covering the unit square, as bare UGRID arrays.

    The smallest topology that is still a mesh: four nodes at the corners of the
    unit square and two triangular faces splitting it along the diagonal. It is
    returned as the `(node_x, node_y, face_node_connectivity)` triple that
    `pyramids.netcdf.ugrid.UgridDataset.from_arrays` takes, so a test decides for
    itself whether to build a `Mesh2d` from it or work on the raw arrays.

    Returns:
        tuple: `(node_x, node_y, face_node_connectivity)` — two `float64` arrays
            of four node coordinates each, and a `(2, 3)` `int32` array of node
            indices per face.
    """
    import numpy as np

    node_x = np.array([0.0, 1.0, 1.0, 0.0])
    node_y = np.array([0.0, 0.0, 1.0, 1.0])
    face_node_connectivity = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return node_x, node_y, face_node_connectivity
