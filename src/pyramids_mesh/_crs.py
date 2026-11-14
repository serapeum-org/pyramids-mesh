"""Shared coordinate-reference-system helpers for the file-format drivers.

Several mesh formats store their CRS as a WKT string in Esri's dialect rather than strict
OGC form. The conversion from that string to a usable CRS is identical across drivers, so it
lives here once instead of being copied into each driver that needs it.
"""

from __future__ import annotations

import pyramids  # noqa: F401  # isort:skip

from osgeo import osr


def crs_from_esri_wkt(wkt: str | None) -> int | str | None:
    """Convert an Esri-dialect WKT string to an EPSG code, OGC WKT, or ``None``.

    Formats such as the Esri TIN ``prj.adf`` and the HEC-RAS root ``Projection`` attribute
    store the coordinate system as WKT in Esri's dialect (``GCS_WGS_1984``, ``D_WGS_1984``,
    ``NAD_1983_StatePlane_...``), which strict OGC validation rejects. The string is morphed
    from the Esri form first; when the result names an authority its EPSG code is returned,
    otherwise the normalised OGC WKT, and only when it parses to a real geographic or
    projected system -- never a guess. A missing, blank, or unparseable value yields ``None``
    rather than a guess.

    Args:
        wkt: The raw WKT string, or ``None`` when the source carried no projection. A
            non-string (e.g. an attribute that reads back as ``None``) is treated as absent.

    Returns:
        int | str | None: An EPSG code, a normalised OGC WKT string, or ``None``.
    """
    wkt = (wkt or "").strip()
    if not wkt:
        return None
    reference = osr.SpatialReference()
    try:
        if reference.ImportFromWkt(wkt) != 0:
            return None
        reference.MorphFromESRI()  # GCS_WGS_1984 -> an OGC-recognised name
    except RuntimeError:
        return None
    code = reference.GetAuthorityCode(None)
    if code:
        return int(code)
    if reference.IsGeographic() or reference.IsProjected():
        return reference.ExportToWkt()
    return None
