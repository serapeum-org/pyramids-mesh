"""Branch tests for the shared ``crs_from_esri_wkt`` helper.

The helper is the single conversion step the Esri TIN and HEC-RAS drivers both use, so its
branches (absent / blank / unparseable / authority-coded / geographic-without-authority /
non-georeferenced) are covered here once rather than through each driver.
"""

import pyramids  # noqa: F401  # isort:skip

from osgeo import osr

from pyramids_mesh._crs import crs_from_esri_wkt


class TestCrsFromEsriWkt:
    """Tests for pyramids_mesh._crs.crs_from_esri_wkt."""

    def test_none_returns_none(self):
        """A ``None`` value (an attribute that read back as ``None``) yields ``None``.

        Test scenario:
            The source carried no projection; the helper must not raise on ``None``.
        """
        assert crs_from_esri_wkt(None) is None, "None input should map to None"

    def test_blank_returns_none(self):
        """A whitespace-only value is treated as absent.

        Test scenario:
            "   " strips to empty, so the helper returns None.
        """
        assert crs_from_esri_wkt("   ") is None, "Blank WKT should map to None"

    def test_unparseable_returns_none(self):
        """An unparseable string resolves to ``None`` rather than raising.

        Test scenario:
            "not-a-wkt" fails ``ImportFromWkt`` (non-zero or RuntimeError), both -> None.
        """
        assert crs_from_esri_wkt("not-a-wkt") is None, "Garbage WKT should map to None"

    def test_authority_coded_wkt_returns_epsg_int(self):
        """A WKT carrying an EPSG authority returns that code as an int.

        Test scenario:
            EPSG:4326 exported to WKT carries ``AUTHORITY["EPSG","4326"]`` -> int 4326.
        """
        reference = osr.SpatialReference()
        reference.ImportFromEPSG(4326)
        result = crs_from_esri_wkt(reference.ExportToWkt())
        assert result == 4326, f"Expected authority code 4326, got {result!r}"

    def test_geographic_without_authority_returns_wkt_string(self):
        """A valid Esri geographic WKT with no authority returns the normalised WKT.

        Test scenario:
            An Esri-dialect GCS_WGS_1984 WKT (no AUTHORITY node) morphs to a real geographic
            system, so the helper returns the OGC WKT string rather than ``None``.
        """
        esri_wkt = (
            'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
            'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
            'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
        )
        result = crs_from_esri_wkt(esri_wkt)
        assert result is not None, "A valid geographic WKT must not map to None"
        assert "WGS" in str(result), f"Expected a WGS-84 WKT/code, got {result!r}"

    def test_nonzero_import_without_exceptions_returns_none(self):
        """A malformed WKT that returns a non-zero code (exceptions off) yields ``None``.

        Test scenario:
            With ``DontUseExceptions`` active, ``ImportFromWkt`` returns a non-zero OGRERR
            for malformed input instead of raising, exercising the ``!= 0`` guard. The
            exception mode is restored afterwards so other tests are unaffected.
        """
        osr.DontUseExceptions()
        try:
            result = crs_from_esri_wkt("PROJCS[broken")
        finally:
            osr.UseExceptions()
        assert result is None, (
            f"A non-zero ImportFromWkt should map to None, got {result!r}"
        )

    def test_non_georeferenced_system_returns_none(self):
        """A parseable but non-geographic, non-projected system yields ``None``.

        Test scenario:
            A ``LOCAL_CS`` (engineering) WKT has no authority and is neither geographic nor
            projected, so the final guard returns None rather than a guess.
        """
        result = crs_from_esri_wkt('LOCAL_CS["unreferenced",UNIT["metre",1.0]]')
        assert result is None, (
            f"A non-georeferenced CRS should map to None, got {result!r}"
        )
