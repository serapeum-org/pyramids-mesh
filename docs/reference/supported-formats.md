# Supported formats

Every format `pyramids_mesh` reads, grouped by the driver subpackage it lives in, with the file
suffix(es) each driver claims. The **driver name** is what you pass to
`MeshDataset.read_file(path, driver="…")`; see each format's own
[reference page](index.md) for capabilities and documented limitations.

Two naming axes are at work (see the per-format pages for detail):

- the **module** is named for the *vendor / toolchain* that owns the parser (so formats that share a
  parser live together, e.g. `sms` holds both `2dm` and `dat`);
- the **driver name** is the *format or model handle* a user asks for — the format name where it has a
  well-known one (`2dm`, `dfsu`, `selafin`), the model name where the file is that model's native
  output (`hecras`, `adcirc`, `3di`), or a vendor + variant where disambiguation is needed
  (`deltares1d`, `modflow6_disv`).

## coastal_ocean — coastal, estuarine and ocean circulation

| Driver | Suffix(es) | Format / model |
|---|---|---|
| `adcirc` | `.14` | ADCIRC (`fort.14`) |
| `schism` | `.gr3` | SCHISM (`hgrid.gr3`) |
| `fvcom` | `.dat` | FVCOM native grid |
| `selafin` | `.slf`, `.srf`, `.ser`, `.sel` | TELEMAC Selafin / Serafin |
| `sww` | `.sww` | ANUGA SWW (NetCDF) |
| `tuflowfv` | `.nc` | TUFLOW FV results (2D) |
| `dfsu` | `.dfsu` | DHI MIKE DFSU (needs the `dhi` extra) |
| `mike21` | `.mesh` | DHI MIKE 21 mesh |
| `3di` | `.nc` | 3Di results |

## river_flood — inland river and flood models

| Driver | Suffix(es) | Format / model |
|---|---|---|
| `hecras` | `.hdf`, `.hdf5` | HEC-RAS 2D (HDF5 plan file) |
| `flo2d` | `.out` | FLO-2D |
| `riverflow2d` | `.fed` | RiverFlow2D mesh |
| `srh2d` | `.srhgeom`, `.srhhydro` | SRH-2D mesh |
| `2dm` | `.2dm` | SMS / TUFLOW 2DM mesh |
| `dat` | `.dat` | SMS ASCII DAT results |
| `xmdf` | `.xmdf`, `.h5` | XMDF HDF5 results |
| `deltares1d` | `.nc` | Deltares 1D network (Delft3D FM / D-HYDRO) |
| `gid` | `.msh` | GiD / CIMNE ASCII mesh (Iber) |

## earth_system — global atmosphere/ocean grids and remapping weights

| Driver | Suffix(es) | Format / model |
|---|---|---|
| `mpas` | `.nc` | MPAS mesh |
| `icon` | `.nc` | ICON icosahedral grid |
| `geos` | `.nc`, `.nc4` | GEOS cubed-sphere grid |
| `esmf` | `.nc` | ESMF unstructured mesh |
| `scrip` | `.nc` | SCRIP remapping grid |

## subsurface — groundwater models

| Driver | Suffix(es) | Format / model |
|---|---|---|
| `modflow6_disv` | `.disv` | MODFLOW 6 DISV (discretization by vertices) |
| `modflow6_disu` | `.disu` | MODFLOW 6 DISU (unstructured discretization) |
| `feflow` | `.fem` | DHI FEFLOW (classic ASCII) |

## engineering — FEM / CAE and CFD solver meshes

| Driver | Suffix(es) | Format / model |
|---|---|---|
| `gmsh` | `.msh` | Gmsh mesh (v2.2 + v4.1) |
| `abaqus` | `.inp` | Abaqus keyword deck |
| `nastran` | `.bdf`, `.nas`, `.fem` | Nastran bulk data |
| `avsucd` | `.inp`, `.avs` | AVS-UCD unstructured cell data |
| `exodus` | `.exo`, `.e` | Exodus II (NetCDF-based FEM) |
| `med` | `.med` | MED-file (Salome / code_aster) |
| `medit` | `.mesh` | Medit (INRIA) mesh |
| `netgen` | `.vol`, `.vol.gz` | Netgen volume mesh |
| `su2` | `.su2` | SU2 native mesh |
| `openfoam` | `.foam` | OpenFOAM polyMesh |
| `kratos` | `.mdpa` | Kratos Multiphysics model part |
| `flac3d` | `.f3grid` | FLAC3D grid |
| `tecplot` | `.dat`, `.tec` | Tecplot finite-element ASCII |

## generic — format-neutral interchange, CAD geometry and triangulators

| Driver | Suffix(es) | Format / model |
|---|---|---|
| `vtk` | `.vtk`, `.vtu` | VTK / VTU unstructured grid |
| `xdmf` | `.xdmf`, `.xmf` | XDMF (XML index over HDF5) |
| `cgns` | `.cgns` | CGNS (CFD, HDF5) |
| `obj` | `.obj` | Wavefront OBJ |
| `stl` | `.stl` | STL (stereolithography) |
| `off` | `.off` | Object File Format |
| `ply` | `.ply` | Stanford PLY |
| `xmstin` | `.tin` | XMS / SMS TIN |
| `esri_tin` | `.adf` | Esri TIN (directory of `.adf` files) |
| `triangle` | `.node`, `.ele` | Triangle / TetGen |
| `easymesh` | `.n`, `.e` | SWAN Easymesh |

## Suffixes are not unique — pass `driver=` to disambiguate

Several extensions are claimed by more than one driver, so the suffix alone cannot pick a reader:

| Suffix | Claimed by |
|---|---|
| `.nc` | `3di`, `tuflowfv`, `deltares1d`, `mpas`, `icon`, `geos`, `esmf`, `scrip` |
| `.dat` | `dat`, `fvcom`, `tecplot` |
| `.inp` | `abaqus`, `avsucd` |
| `.msh` | `gmsh`, `gid` |
| `.fem` | `feflow`, `nastran` |
| `.mesh` | `mike21`, `medit` |
| `.e` | `exodus`, `easymesh` |

For an ambiguous suffix, name the driver explicitly:

```python
from pyramids.netcdf.ugrid import MeshDataset

mesh = MeshDataset.read_file("model.nc", driver="deltares1d")
```
