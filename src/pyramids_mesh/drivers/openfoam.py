"""OpenFOAM ``polyMesh`` reader.

An OpenFOAM mesh is not a single file but a **directory** -- ``polyMesh`` -- of
plain-text, extensionless component files: ``points`` (the node coordinates),
``faces`` (the face-point connectivity), ``owner`` and ``neighbour`` (the cell
either side of each face) and ``boundary`` (the named surface patches). Each file
opens with a ``FoamFile`` dictionary header describing its ``class`` and ``format``.

OpenFOAM stores a 3D **volume** mesh as an unstructured list of cells, each bounded
by faces. The faces split into two groups: the ``nInternalFaces`` shared between two
cells, and the boundary faces that bound exactly one cell and are grouped into the
named patches of ``boundary``. ADR-0005 keeps file-format drivers to the readable 2D
surface and drops the interior volume connectivity, so this driver exposes only the
boundary faces -- the ones the patches enumerate through their
``[startFace, startFace + nFaces)`` ranges. Those point indices are compacted to the
nodes they actually reference, and the result is a fill-padded face-node
connectivity (triangles, quads or mixed polygons).

The coordinates are CFD local coordinates with no georeference, so ``crs`` is
``None``. The third coordinate becomes a ``node_z`` node variable.

A directory has no file suffix, so ``read_file`` cannot dispatch a bare directory
path, and the component files are extensionless, so there is no suffix to claim
either. The driver is therefore reached by passing the ``polyMesh`` directory with
``driver="openfoam"``; ``can_open`` additionally accepts a path to the ``points``
component inside it, so a dispatcher that probes candidate files still recognises it.

Only the ASCII ``format`` is parsed. A ``FoamFile`` header declaring
``format binary`` is refused with a clear error rather than mis-parsed.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

import os
import re
import warnings

import numpy as np

from pyramids.base._errors import FileFormatNotSupportedError
from pyramids.netcdf.ugrid import MESH_DRIVER_API_VERSION

from pyramids_mesh._base import BaseDriver, ParsedMesh

# The component files that make a directory an OpenFOAM polyMesh. ``owner`` and
# ``neighbour`` describe the interior volume, which this reader drops, so they are not
# required to recognise the surface; ``points``, ``faces`` and ``boundary`` are.
_REQUIRED = ("points", "faces", "boundary")


def _strip_foam(text: str) -> str:
    """Remove the ``FoamFile`` header, block comment and line comments from ``text``.

    What remains is the data body -- a leading count and a parenthesised list -- with
    nothing a parser must step around. The ``FoamFile`` dictionary is removed because
    it carries its own braces, which would otherwise be mistaken for a boundary patch.

    Args:
        text: The raw file contents.

    Returns:
        str: The contents with the banner comment, the ``FoamFile`` block and every
            ``//`` line comment replaced by whitespace.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"FoamFile\s*\{.*?\}", " ", text, flags=re.DOTALL)
    return re.sub(r"//.*", " ", text)


def _foam_list(text: str) -> tuple[int, str]:
    """Return a FoamFile list's declared count and the body between its outer parens.

    A FoamFile data section is ``<count> ( <entries> )``. The body is matched greedily
    to the final ``)``, so a nested list -- a boundary patch's ``inGroups 1(cyclic)``,
    say -- stays inside it rather than ending the match early.

    Args:
        text: The file contents, already passed through :func:`_strip_foam`.

    Returns:
        tuple[int, str]: The declared entry count and the raw body string.

    Raises:
        FileFormatNotSupportedError: If no ``<count> ( ... )`` list is present.
    """
    match = re.search(r"(\d+)\s*\((.*)\)", text, re.DOTALL)
    if match is None:
        raise FileFormatNotSupportedError(
            "expected an OpenFOAM '<count> ( ... )' list but found none"
        )
    return int(match.group(1)), match.group(2)


def _require_ascii(text: str, component: str) -> None:
    """Refuse a component whose ``FoamFile`` header declares a non-ASCII format.

    Args:
        text: The raw file contents, before stripping.
        component: The component name, for the error message.

    Raises:
        FileFormatNotSupportedError: If the header's ``format`` is anything but
            ``ascii`` -- binary polyMesh files are not supported.
    """
    header = re.search(r"FoamFile\s*\{.*?\}", text, re.DOTALL)
    if header is None:
        return
    fmt = re.search(r"\bformat\s+(\w+)", header.group(0))
    if fmt is not None and fmt.group(1).lower() != "ascii":
        raise FileFormatNotSupportedError(
            f"the OpenFOAM '{component}' component is {fmt.group(1)!r} format; this "
            f"reader parses ascii polyMesh files only"
        )


def _parse_points(text: str) -> np.ndarray:
    """Parse a ``points`` file into an ``(n_point, 3)`` float array.

    Args:
        text: The raw ``points`` file contents.

    Returns:
        np.ndarray: The node coordinates as ``float64``.

    Raises:
        FileFormatNotSupportedError: If the point count in the header does not match the
            number of coordinate triples found.
    """
    _require_ascii(text, "points")
    count, body = _foam_list(_strip_foam(text))
    triples = re.findall(r"\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)", body)
    if len(triples) != count:
        raise FileFormatNotSupportedError(
            f"the OpenFOAM 'points' header declares {count} points but {len(triples)} "
            f"coordinate triples were found"
        )
    return np.array(triples, dtype=np.float64)


def _parse_faces(text: str) -> list[list[int]]:
    """Parse a ``faces`` file into a list of 0-based point-index lists.

    Each entry is ``n(i0 i1 ... i{n-1})`` -- a face's node count followed by that many
    point indices. The indices are already 0-based, as OpenFOAM writes them.

    Args:
        text: The raw ``faces`` file contents.

    Returns:
        list[list[int]]: One list of point indices per face, in file order.

    Raises:
        FileFormatNotSupportedError: If the face count in the header does not match the
            number of face entries, or an entry's declared size disagrees with the
            indices it lists.
    """
    _require_ascii(text, "faces")
    count, body = _foam_list(_strip_foam(text))
    faces: list[list[int]] = []
    for size, inside in re.findall(r"(\d+)\s*\(([^)]*)\)", body):
        indices = [int(token) for token in inside.split()]
        if len(indices) != int(size):
            raise FileFormatNotSupportedError(
                f"an OpenFOAM face declares {size} nodes but lists {len(indices)}"
            )
        faces.append(indices)
    if len(faces) != count:
        raise FileFormatNotSupportedError(
            f"the OpenFOAM 'faces' header declares {count} faces but {len(faces)} "
            f"entries were found"
        )
    return faces


def _parse_boundary(text: str) -> list[tuple[str, int, int]]:
    """Parse a ``boundary`` file into its named patches.

    Args:
        text: The raw ``boundary`` file contents.

    Returns:
        list[tuple[str, int, int]]: One ``(name, start_face, n_faces)`` per patch, in
            file order.

    Raises:
        FileFormatNotSupportedError: If a patch is missing its ``startFace`` or
            ``nFaces`` entry.
    """
    _require_ascii(text, "boundary")
    _, body = _foam_list(_strip_foam(text))
    patches: list[tuple[str, int, int]] = []
    for name, block in re.findall(r"(\w+)\s*\{([^}]*)\}", body):
        start = re.search(r"\bstartFace\s+(\d+)", block)
        n_faces = re.search(r"\bnFaces\s+(\d+)", block)
        if start is None or n_faces is None:
            raise FileFormatNotSupportedError(
                f"the OpenFOAM boundary patch {name!r} is missing startFace or nFaces"
            )
        patches.append((name, int(start.group(1)), int(n_faces.group(1))))
    return patches


class OpenFoamDriver(BaseDriver):
    """Reader for the OpenFOAM ``polyMesh`` directory format.

    An OpenFOAM mesh is a ``polyMesh`` directory of plain-text component files. This
    driver reads the nodes and the boundary-patch faces, compacts the referenced
    points, and returns the 2D surface the patches describe -- the interior volume
    connectivity is dropped (ADR-0005). Coordinates are CFD local coordinates, so
    ``crs`` is ``None`` and the third coordinate becomes a ``node_z`` variable.

    Documented limitations:
        - **The input is a directory.** ``read_file`` cannot dispatch a bare directory,
          and the component files are extensionless, so there is no suffix to claim.
          Pass the ``polyMesh`` directory with ``driver="openfoam"``; ``can_open`` also
          accepts a path to the ``points`` component inside it.
        - **Only the boundary surface is read.** The ``nInternalFaces`` interior faces,
          and ``owner``/``neighbour``, describe the 3D volume and are not represented;
          the mesh is the union of the boundary patches' faces.
        - **ASCII only.** A component whose ``FoamFile`` header declares
          ``format binary`` is refused.
        - **Point indices are already 0-based**, as OpenFOAM writes them.
        - Every face may be a triangle, a quad or a larger polygon; the connectivity is
          fill-padded to the widest face.

    Example:
        Read an OpenFOAM mesh by its ``polyMesh`` directory path (shown, not run as a
        doctest -- it needs a real polyMesh directory)::

            from pyramids.netcdf.ugrid import UgridDataset

            mesh = UgridDataset.read_file("case/constant/polyMesh", driver="openfoam")
            elevation = mesh.get("node_z")
    """

    name = "openfoam"
    long_name = "OpenFOAM polyMesh"
    # A polyMesh is a directory of extensionless component files. The recognised *file* is
    # the ParaView case marker ``<case>.foam``, which points at the case directory;
    # ``can_open`` additionally accepts a ``polyMesh`` directory or its ``points`` component
    # directly, and dispatch also works through ``driver="openfoam"``.
    suffixes = (".foam",)
    api_version = MESH_DRIVER_API_VERSION
    capabilities = frozenset({"read_mesh"})

    def can_open(self, path: str) -> bool:
        """Report whether ``path`` is a polyMesh directory or its ``points`` component.

        Overrides the base check, which declines every directory and only ever sniffs a
        file: an OpenFOAM mesh is a directory, reached either directly or through its
        ``points`` component file. Never raises -- a sniff error is warned and declined.

        Args:
            path: The filesystem path to test.

        Returns:
            bool: ``True`` for a polyMesh directory, or a ``points`` file whose
                directory is one.
        """
        poly_dir = self._poly_dir(path)
        if poly_dir is None:
            return False
        try:
            return self._sniff(poly_dir)
        except Exception as error:  # noqa: BLE001 - the contract is never to raise
            # Defensive: os.listdir on a readable directory does not raise, so this
            # guard mirrors the base class's never-raise contract rather than a
            # reachable path.
            warnings.warn(  # pragma: no cover
                f"{self.name} failed to sniff {path!r} and is treated as declining it: "
                f"{type(error).__name__}: {error}",
                stacklevel=2,
            )
            return False  # pragma: no cover

    @staticmethod
    def _poly_dir(path: str) -> str | None:
        """Resolve the polyMesh directory from a directory, ``points`` file, or ``.foam``.

        A directory is taken as the polyMesh itself; a ``points`` file resolves to its
        directory; a ``<case>.foam`` ParaView marker resolves to the case's
        ``constant/polyMesh`` (or ``polyMesh``, or the case directory) when one exists.
        """
        if os.path.isdir(path):
            return path
        if os.path.isfile(path):
            base = os.path.basename(path).lower()
            if base == "points":
                return os.path.dirname(path)
            if base.endswith(".foam"):
                case = os.path.dirname(path)
                for candidate in (
                    os.path.join(case, "constant", "polyMesh"),
                    os.path.join(case, "polyMesh"),
                    case,
                ):
                    if os.path.isdir(candidate):
                        return candidate
        return None

    def _sniff(self, path: str) -> bool:
        """Report whether the directory holds the required polyMesh components."""
        present = {entry.lower() for entry in os.listdir(path)}
        return all(component in present for component in _REQUIRED)

    def _parse(self, path: str, *, lazy: bool) -> ParsedMesh:
        """Read a polyMesh into a :class:`ParsedMesh`, from its directory or ``points``.

        Args:
            path: The polyMesh directory, or its ``points`` component file.
            lazy: Unused -- a polyMesh surface is read whole.

        Returns:
            ParsedMesh: The boundary surface, with the third coordinate as ``node_z``.

        Raises:
            FileFormatNotSupportedError: If the path is not a polyMesh or a required
                component is missing or malformed.
        """
        poly_dir = self._poly_dir(path)
        if poly_dir is None:
            raise FileFormatNotSupportedError(
                f"{path!r} is not an OpenFOAM polyMesh directory or a 'points' file "
                f"inside one"
            )
        return _OpenFoamReader(poly_dir).build()


class _OpenFoamReader:
    """One polyMesh directory: read the components and assemble the boundary surface."""

    def __init__(self, path: str) -> None:
        self.path = path

    def build(self) -> ParsedMesh:
        """Read the components and build the compacted boundary-surface mesh.

        Returns:
            ParsedMesh: The boundary surface, 0-based and fill-padded, with ``node_z``
                carrying the third coordinate and ``crs`` ``None``.

        Raises:
            FileFormatNotSupportedError: If a required component is missing or malformed.
        """
        self._require()
        points = _parse_points(self._read("points"))
        faces = _parse_faces(self._read("faces"))
        patches = _parse_boundary(self._read("boundary"))
        boundary = self._boundary_faces(faces, patches)
        node_x, node_y, node_z, connectivity = self._compact(points, boundary)
        return ParsedMesh(
            node_x=node_x,
            node_y=node_y,
            faces=connectivity,
            fill_value=-1,
            start_index=0,
            original_start_index=0,  # OpenFOAM point indices are 0-based
            node_z=node_z,
            crs=None,  # CFD local coordinates carry no georeference
            variables={},
            global_attributes={"source": "OpenFOAM polyMesh"},
            topology_dimension=2,
            mesh_name="mesh2d",
        )

    def _require(self) -> None:
        """Refuse a directory missing a required component."""
        present = {entry.lower() for entry in os.listdir(self.path)}
        missing = [name for name in _REQUIRED if name not in present]
        if missing:
            raise FileFormatNotSupportedError(
                f"{self.path!r} is not an OpenFOAM polyMesh: missing {missing}"
            )

    def _read(self, component: str) -> str:
        """Read one component file as text, tolerating undecodable bytes."""
        with open(
            os.path.join(self.path, component), "r", encoding="utf-8", errors="replace"
        ) as handle:
            return handle.read()

    def _boundary_faces(
        self, faces: list[list[int]], patches: list[tuple[str, int, int]]
    ) -> list[list[int]]:
        """Gather the faces named by the boundary patches, in patch order.

        Each patch names the contiguous range ``[startFace, startFace + nFaces)`` of
        the face list; their union is the readable 2D surface.

        Args:
            faces: Every face's point indices, in file order.
            patches: The ``(name, start_face, n_faces)`` triples.

        Returns:
            list[list[int]]: The boundary faces' point-index lists.

        Raises:
            FileFormatNotSupportedError: If a patch range reaches past the face list.
        """
        gathered: list[list[int]] = []
        for name, start, n_faces in patches:
            stop = start + n_faces
            if stop > len(faces):
                raise FileFormatNotSupportedError(
                    f"the OpenFOAM boundary patch {name!r} spans faces "
                    f"[{start}, {stop}) but there are only {len(faces)} faces"
                )
            gathered.extend(faces[start:stop])
        return gathered

    @staticmethod
    def _compact(
        points: np.ndarray, boundary: list[list[int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compact the referenced points and fill-pad the face connectivity.

        Only the points the boundary faces reference are kept, renumbered to a dense
        0-based range; the faces are rewritten to that numbering and padded with ``-1``
        to the widest face, so a mixed triangle/quad surface is rectangular.

        Args:
            points: The full ``(n_point, 3)`` coordinate array.
            boundary: The boundary faces' point-index lists.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ``node_x``,
                ``node_y``, ``node_z`` and the ``(n_face, max_nodes)`` fill-padded
                face-node connectivity.

        Raises:
            FileFormatNotSupportedError: If the boundary names no faces at all.
        """
        if not boundary:
            raise FileFormatNotSupportedError(
                "the OpenFOAM boundary names no faces, so there is no surface to read"
            )
        used = np.unique(np.concatenate([np.asarray(face) for face in boundary]))
        remap = {int(old): new for new, old in enumerate(used)}
        width = max(len(face) for face in boundary)
        connectivity = np.full((len(boundary), width), -1, dtype=np.int64)
        for row, face in enumerate(boundary):
            connectivity[row, : len(face)] = [remap[index] for index in face]
        kept = points[used]
        return (
            kept[:, 0].astype(np.float64),
            kept[:, 1].astype(np.float64),
            kept[:, 2].astype(np.float64),
            connectivity,
        )
