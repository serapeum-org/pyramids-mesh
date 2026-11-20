# Reference

The API surface of `pyramids_mesh`, generated from the docstrings by
[mkdocstrings](https://mkdocstrings.github.io/).

What this package publishes is **drivers**, not an API you call directly. A driver is registered
with `pyramids-core` through the `pyramids.mesh_drivers` entry-point group, so installing this
package is the whole integration: `pyramids` then dispatches a read of a format it does not handle
itself to the driver that claims it.

```python
from pyramids.netcdf.ugrid import MeshDataset

mesh = MeshDataset.read_file("domain.2dm")   # resolved to this package's 2dm driver
```

So the entry points below are the surface that matters, and each driver has its own page listing
its suffixes, capabilities and documented limitations:

| Driver | Format | Page |
|---|---|---|
| `2dm` | SMS / TUFLOW 2DM mesh frame | [2DM](river_flood/2dm.md) |
| `dat` | SMS ASCII results, attached to a mesh | [DAT](river_flood/dat.md) |
| `mike21` | DHI Mike21 mesh frame | [Mike21](coastal_ocean/mike21.md) |
| `selafin` | TELEMAC Selafin / Serafin | [Selafin](coastal_ocean/selafin.md) |
| `sww` | ANUGA SWW (NetCDF) | [SWW](coastal_ocean/sww.md) |
| `xmdf` | XMDF HDF5 results | [XMDF](river_flood/xmdf.md) |
| `3di` | 3Di results (NetCDF) | [3Di](river_flood/3di.md) |
| `tuflowfv` | TUFLOW FV results (NetCDF, 2D) | [TUFLOW FV](coastal_ocean/tuflowfv.md) |
| `dfsu` | DHI MIKE DFSU (needs `dhi`) | [DFSU](coastal_ocean/dfsu.md) |
| `arcgis-tin` | Esri TIN (directory of `.adf`) | [Esri TIN](geometry/arcgis-tin.md) |
| `ply` | Stanford PLY polygon | [PLY](geometry/ply.md) |
| `xms-tin` | XMS / SMS TIN | [XMS TIN](geometry/xms-tin.md) |
| `xdmf` | XDMF (XML index over HDF5) | [XDMF](cfd/xdmf.md) |
| `flo2d` | FLO-2D (directory of files) | [FLO-2D](river_flood/flo2d.md) |
| `hecras` | HEC-RAS 2D (HDF5 plan file) | [HEC-RAS 2D](river_flood/hecras.md) |
| `adcirc` | ADCIRC (`fort.14` + `fort.63.nc`) | [ADCIRC](coastal_ocean/adcirc.md) |
| `delft3d-1d` | Delft3D FM / D-HYDRO 1D network (NetCDF, `Mesh1d`) | [Delft3D FM / D-HYDRO 1D](river_flood/delft3d-1d.md) |
| `dhydro-1d` | Alias of `delft3d-1d` (same reader, D-HYDRO name) | [Delft3D FM / D-HYDRO 1D](river_flood/delft3d-1d.md) |
| `schism` | SCHISM hgrid.gr3 (ASCII) | [SCHISM (.gr3)](coastal_ocean/schism.md) |
| `gmsh` | Gmsh .msh (v2.2 + v4.1) | [Gmsh (.msh)](mesh_generators/gmsh.md) |
| `vtk` | VTK / VTU unstructured grid | [VTK / VTU](geometry/vtk.md) |
| `modflow6-disv` | MODFLOW 6 DISV (groundwater vertex grid) | [MODFLOW 6 DISV](subsurface/modflow6.md) |
| `mpas` | MPAS Voronoi mesh (NetCDF) | [MPAS (NetCDF)](earth_system/mpas.md) |
| `icon` | ICON icosahedral grid (NetCDF) | [ICON (NetCDF)](earth_system/icon.md) |
| `fvcom` | FVCOM native `*_grd.dat` (ASCII) | [FVCOM (native grid)](coastal_ocean/fvcom.md) |
| `triangle` | Triangle/TetGen `.node`+`.ele` | [Triangle / TetGen](mesh_generators/triangle.md) |
| `easymesh` | SWAN Easymesh `.n`+`.e` | [SWAN Easymesh](mesh_generators/easymesh.md) |
| `exodus` | Exodus II (NetCDF-based FEM) | [Exodus II (.exo)](finite_element/exodus.md) |
| `su2` | SU2 `.su2` (ASCII FEM) | [SU2 (.su2)](cfd/su2.md) |
| `srh2d` | SRH-2D (`.srhgeom`+`.srhhydro`) | [SRH-2D](river_flood/srh2d.md) |
| `modflow6-disu` | MODFLOW 6 DISU (groundwater, unstructured) | [MODFLOW 6 DISU](subsurface/modflow6-disu.md) |
| `feflow` | FEFLOW `.fem` (groundwater FEM) | [FEFLOW (.fem)](subsurface/feflow.md) |
| `scrip` | SCRIP remap grid (NetCDF) | [SCRIP grid](earth_system/scrip.md) |
| `esmf` | ESMF unstructured mesh (NetCDF) | [ESMF mesh](earth_system/esmf.md) |
| `geos` | GEOS cubed-sphere native (NetCDF) | [GEOS cubed-sphere](earth_system/geos.md) |
| `cgns` | CGNS (CFD, HDF5) | [CGNS (.cgns)](cfd/cgns.md) |
| `med` | MED / Salome (FEM, HDF5) | [MED (.med)](finite_element/med.md) |
| `nastran` | Nastran `.bdf`/`.nas`/`.fem` | [Nastran bulk data](finite_element/nastran.md) |
| `abaqus` | Abaqus `.inp` (FEM) | [Abaqus (.inp)](finite_element/abaqus.md) |
| `avs-ucd` | AVS-UCD `.inp`/`.avs` | [AVS-UCD](finite_element/avs-ucd.md) |
| `medit` | Medit `.mesh` (INRIA) | [Medit (.mesh)](mesh_generators/medit.md) |
| `netgen` | Netgen `.vol` | [Netgen (.vol)](mesh_generators/netgen.md) |
| `tecplot` | Tecplot FE ASCII | [Tecplot FE](cfd/tecplot.md) |
| `flac3d` | FLAC3D `.f3grid` (geomech) | [FLAC3D (.f3grid)](finite_element/flac3d.md) |
| `kratos` | Kratos `.mdpa` (FEM) | [Kratos MDPA](finite_element/kratos.md) |
| `openfoam` | OpenFOAM polyMesh (CFD boundary) | [OpenFOAM polyMesh](cfd/openfoam.md) |
| `obj` | OBJ surface (no georef) | [Wavefront OBJ](geometry/obj.md) |
| `stl` | STL surface (no georef) | [STL](geometry/stl.md) |
| `off` | OFF surface (no georef) | [OFF (Object File Format)](geometry/off.md) |
| `gid` | GiD / Iber mesh (`.msh`) | [GiD / Iber](river_flood/gid.md) |
| `riverflow2d` | RiverFlow2D `.fed` (triangular hydraulic mesh) | [RiverFlow2D](river_flood/riverflow2d.md) |

The package root deliberately re-exports nothing (`__all__` is empty): it carries `__version__` and
the import contract that activates the vendored osgeo bindings. Driver classes are reached by their
module path — `pyramids_mesh.drivers.river_flood.sms.TwoDmDriver` — which is also how their entry points name
them, though normal use goes through `pyramids` rather than importing them at all.

For the mesh data model itself — `Mesh2d`, `Mesh1d`, `MeshDataset`, connectivity, graph
algorithms, regridding, mesh ↔ vector — see the
[pyramids UGRID reference](https://serapeum-org.github.io/pyramids/latest/). The split between the
two is [ADR-0005](../adr/0005-file-format-drivers.md), which supersedes ADR-0003 in full.

::: pyramids_mesh
    options:
        show_root_heading: true
        show_source: true
        heading_level: 2
