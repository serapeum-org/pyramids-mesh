# Reference

The API surface of `pyramids_mesh`, generated from the docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

What this package publishes is **drivers**, not an API you call directly. A driver is registered
with `pyramids-gis` through the `pyramids.mesh_drivers` entry-point group, so installing this
package is the whole integration: `pyramids` then dispatches a read of a format it does not handle
itself to the driver that claims it.

```python
from pyramids.netcdf.ugrid import UgridDataset

mesh = UgridDataset.read_file("domain.2dm")   # resolved to this package's 2dm driver
```

So the entry points below are the surface that matters, and each driver has its own page listing
its suffixes, capabilities and documented limitations:

| Driver | Format | Page |
|---|---|---|
| `2dm` | SMS / TUFLOW 2DM mesh frame | [2DM](2dm.md) |
| `dat` | SMS ASCII results, attached to a mesh | [DAT](dat.md) |
| `mike21` | DHI Mike21 mesh frame | [Mike21](mike21.md) |
| `selafin` | TELEMAC Selafin / Serafin | [Selafin](selafin.md) |
| `sww` | ANUGA SWW (NetCDF) | [SWW](sww.md) |
| `xmdf` | XMDF HDF5 results | [XMDF](xmdf.md) |
| `3di` | 3Di results (NetCDF) | [3Di](3di.md) |
| `tuflowfv` | TUFLOW FV results (NetCDF, 2D) | [TUFLOW FV](tuflowfv.md) |
| `dfsu` | DHI MIKE DFSU (needs `dhi`) | [DFSU](dfsu.md) |
| `esri_tin` | Esri TIN (directory of `.adf`) | [Esri TIN](esri-tin.md) |
| `ply` | Stanford PLY polygon | [PLY](ply.md) |
| `xmstin` | XMS / SMS TIN | [XMS TIN](xmstin.md) |
| `xdmf` | XDMF (XML index over HDF5) | [XDMF](xdmf.md) |
| `flo2d` | FLO-2D (directory of files) | [FLO-2D](flo2d.md) |
| `hecras` | HEC-RAS 2D (HDF5 plan file) | [HEC-RAS 2D](hecras.md) |
| `adcirc` | ADCIRC (`fort.14` + `fort.63.nc`) | [ADCIRC](adcirc.md) |
| `deltares1d` | Deltares 1D network (NetCDF, `Network1d`) | [Deltares 1D (D-Flow FM)](deltares1d.md) |
| `schism` | SCHISM hgrid.gr3 (ASCII) | [SCHISM (.gr3)](schism.md) |
| `gmsh` | Gmsh .msh (v2.2 + v4.1) | [Gmsh (.msh)](gmsh.md) |
| `vtk` | VTK / VTU unstructured grid | [VTK / VTU](vtk.md) |
| `modflow6_disv` | MODFLOW 6 DISV (groundwater vertex grid) | [MODFLOW 6 DISV](modflow6.md) |
| `mpas` | MPAS Voronoi mesh (NetCDF) | [MPAS (NetCDF)](mpas.md) |
| `icon` | ICON icosahedral grid (NetCDF) | [ICON (NetCDF)](icon.md) |
| `fvcom` | FVCOM native `*_grd.dat` (ASCII) | [FVCOM (native grid)](fvcom.md) |
| `triangle` | Triangle/TetGen `.node`+`.ele` | [Triangle / TetGen](triangle.md) |
| `exodus` | Exodus II (NetCDF-based FEM) | [Exodus II (.exo)](exodus.md) |
| `su2` | SU2 `.su2` (ASCII FEM) | [SU2 (.su2)](su2.md) |
| `srh2d` | SRH-2D (`.srhgeom`+`.srhhydro`) | [SRH-2D](srh2d.md) |
| `modflow6_disu` | MODFLOW 6 DISU (groundwater, unstructured) | [MODFLOW 6 DISU](modflow6_disu.md) |
| `feflow` | FEFLOW `.fem` (groundwater FEM) | [FEFLOW (.fem)](feflow.md) |
| `scrip` | SCRIP remap grid (NetCDF) | [SCRIP grid](scrip.md) |
| `esmf` | ESMF unstructured mesh (NetCDF) | [ESMF mesh](esmf.md) |
| `geos` | GEOS cubed-sphere native (NetCDF) | [GEOS cubed-sphere](geos.md) |
| `cgns` | CGNS (CFD, HDF5) | [CGNS (.cgns)](cgns.md) |
| `med` | MED / Salome (FEM, HDF5) | [MED (.med)](med.md) |
| `nastran` | Nastran `.bdf`/`.nas`/`.fem` | [Nastran bulk data](nastran.md) |
| `abaqus` | Abaqus `.inp` (FEM) | [Abaqus (.inp)](abaqus.md) |
| `avsucd` | AVS-UCD `.inp`/`.avs` | [AVS-UCD](avsucd.md) |
| `medit` | Medit `.mesh` (INRIA) | [Medit (.mesh)](medit.md) |
| `netgen` | Netgen `.vol` | [Netgen (.vol)](netgen.md) |
| `tecplot` | Tecplot FE ASCII | [Tecplot FE](tecplot.md) |
| `flac3d` | FLAC3D `.f3grid` (geomech) | [FLAC3D (.f3grid)](flac3d.md) |
| `kratos` | Kratos `.mdpa` (FEM) | [Kratos MDPA](kratos.md) |
| `openfoam` | OpenFOAM polyMesh (CFD boundary) | [OpenFOAM polyMesh](openfoam.md) |
| `obj` | OBJ surface (no georef) | [Wavefront OBJ](obj.md) |
| `stl` | STL surface (no georef) | [STL](stl.md) |
| `off` | OFF surface (no georef) | [OFF (Object File Format)](off.md) |
| `gid` | GiD / Iber mesh (`.msh`) | [GiD / Iber](gid.md) |

The package root deliberately re-exports nothing (`__all__` is empty): it carries `__version__` and
the import contract that activates the vendored osgeo bindings. Driver classes are reached by their
module path — `pyramids_mesh.drivers.sms.TwoDmDriver` — which is also how their entry points name
them, though normal use goes through `pyramids` rather than importing them at all.

For the mesh data model itself — `Mesh2d`, `Network1d`, `UgridDataset`, connectivity, graph
algorithms, regridding, mesh ↔ vector — see the
[pyramids UGRID reference](https://serapeum-org.github.io/pyramids/latest/). The split between the
two is [ADR-0005](../adr/0005-file-format-drivers.md), which supersedes ADR-0003 in full.

::: pyramids_mesh
    options:
        show_root_heading: true
        show_source: true
        heading_level: 2
