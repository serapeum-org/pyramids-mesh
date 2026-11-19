# Supported formats

Every format `pyramids_mesh` reads. The **driver name** is the handle you pass to
`MeshDataset.read_file(path, driver="…")`; the remaining columns give the software that writes the
file, its driver subpackage, what that software is, and the file format. See each format's own
[reference page](index.md) for capabilities and documented limitations.

Two naming axes are at work:

- the **module** (the `.py` file) is named for the **software/model** that writes the file (`hecras`, `adcirc`,
  `telemac`…), or for the **format** itself when it is a vendor-neutral standard
  with no single owning tool (`vtk`, `cgns`, `obj`…). A module can hold several drivers (`sms` holds
  both `2dm` and `dat`).
- the **driver name** is the handle passed to `driver=` — the format name where it has a well-known
  one (`2dm`, `dfsu`, `selafin`), the software name where the file is that software's native output
  (`hecras`, `adcirc`, `3di`), or a name plus a variant where one toolchain emits several grids
  (`modflow6-disv`, `modflow6-disu`). Multi-word driver names are **hyphenated**, never underscored
  (`delft3d-1d`, `dhydro-1d`, `arcgis-tin`, `modflow6-disv`); a single compact token stays one word
  (`2dm`, `3di`, `flo2d`, `srh2d`). Subpackage and package names, being Python identifiers, keep
  underscores (`coastal_ocean`, `pyramids_mesh`).

| Driver          | Software            | Subpackage      | What the software is                                     | Format                                             |
|-----------------|---------------------|-----------------|----------------------------------------------------------|----------------------------------------------------|
| `adcirc`        | ADCIRC              | `coastal_ocean` | Coastal-ocean circulation / storm-surge model            | ADCIRC `fort.14` ASCII grid (`.14`)                |
| `schism`        | SCHISM              | `coastal_ocean` | Cross-scale ocean/coastal circulation model              | `hgrid.gr3` ASCII grid (`.gr3`)                    |
| `fvcom`         | FVCOM               | `coastal_ocean` | Finite-volume coastal-ocean model                        | FVCOM native grid (`.dat`)                         |
| `selafin`       | TELEMAC             | `coastal_ocean` | Free-surface hydrodynamics suite (rivers/coasts)         | Selafin / Serafin (`.slf`, `.srf`, `.ser`, `.sel`) |
| `sww`           | ANUGA               | `coastal_ocean` | Shallow-water flood & tsunami model                      | SWW, NetCDF (`.sww`)                               |
| `tuflowfv`      | TUFLOW FV           | `coastal_ocean` | Flexible-mesh flood/coastal hydrodynamic solver          | TUFLOW FV results, NetCDF (`.nc`)                  |
| `dfsu`          | DHI MIKE            | `coastal_ocean` | DHI's coastal/estuarine hydrodynamic suite               | DFSU, needs `dhi` extra (`.dfsu`)                  |
| `mike21`        | DHI MIKE 21         | `coastal_ocean` | DHI's 2D coastal/estuarine hydrodynamic model            | MIKE 21 mesh (`.mesh`)                             |
| `3di`           | 3Di                 | `coastal_ocean` | Urban/rural flood-modelling platform                     | 3Di results, NetCDF (`.nc`)                        |
| `hecras`        | HEC-RAS             | `river_flood`   | USACE river hydraulics & flood model                     | HEC-RAS HDF5 plan file (`.hdf`, `.hdf5`)           |
| `flo2d`         | FLO-2D              | `river_flood`   | Flood-routing / overland-flow model                      | FLO-2D (`.out`)                                    |
| `riverflow2d`   | RiverFlow2D         | `river_flood`   | 2D river/flood hydrodynamic & sediment model             | RiverFlow2D mesh (`.fed`)                          |
| `srh2d`         | SRH-2D              | `river_flood`   | USBR 2D river hydraulics & sedimentation model           | SRH-2D (`.srhgeom`, `.srhhydro`)                   |
| `2dm`           | SMS                 | `river_flood`   | Aquaveo Surface-water Modeling System (GUI)              | SMS/TUFLOW 2DM mesh (`.2dm`)                       |
| `dat`           | SMS                 | `river_flood`   | Aquaveo Surface-water Modeling System (GUI)              | SMS ASCII DAT results (`.dat`)                     |
| `xmdf`          | Aquaveo XMS         | `river_flood`   | Results container for the XMS suite (SMS/SRH-2D)         | XMDF HDF5 results (`.xmdf`, `.h5`)                 |
| `delft3d-1d`    | Delft3D FM          | `river_flood`   | Deltares' flexible-mesh hydrodynamic suite               | UGRID-1.0 + Deltares-extension NetCDF (`.nc`)      |
| `dhydro-1d`     | D-HYDRO             | `river_flood`   | Same software, D-HYDRO name; alias of `delft3d-1d`       | UGRID-1.0 + Deltares-extension NetCDF (`.nc`)      |
| `gid`           | GiD (Iber)          | `river_flood`   | CIMNE pre/post-processor; reads the Iber 2D model export | GiD / CIMNE ASCII mesh (`.msh`)                    |
| `mpas`          | MPAS                | `earth_system`  | Model for Prediction Across Scales (atmos/ocean)         | MPAS mesh, NetCDF (`.nc`)                          |
| `icon`          | ICON                | `earth_system`  | ICOsahedral Nonhydrostatic weather/climate model         | ICON icosahedral grid, NetCDF (`.nc`)              |
| `geos`          | GEOS                | `earth_system`  | NASA global atmosphere / earth-system model              | GEOS cubed-sphere grid, NetCDF (`.nc`, `.nc4`)     |
| `esmf`          | ESMF                | `earth_system`  | Earth System Modeling Framework (model coupling)         | ESMF unstructured mesh, NetCDF (`.nc`)             |
| `scrip`         | SCRIP               | `earth_system`  | Remapping-weights package / grid convention              | SCRIP remapping grid, NetCDF (`.nc`)               |
| `modflow6-disv` | MODFLOW 6           | `subsurface`    | USGS groundwater-flow model                              | DISV discretization-by-vertices (`.disv`)          |
| `modflow6-disu` | MODFLOW 6           | `subsurface`    | USGS groundwater-flow model                              | DISU unstructured discretization (`.disu`)         |
| `feflow`        | FEFLOW              | `subsurface`    | DHI subsurface flow & transport model                    | FEFLOW classic ASCII (`.fem`)                      |
| `gmsh`          | Gmsh                | `engineering`   | Open-source finite-element mesh generator                | Gmsh mesh v2.2/v4.1 (`.msh`)                       |
| `abaqus`        | Abaqus              | `engineering`   | Dassault Systemes FEA solver                             | Abaqus keyword deck (`.inp`)                       |
| `nastran`       | Nastran             | `engineering`   | Structural FEA solver                                    | Nastran bulk data (`.bdf`, `.nas`, `.fem`)         |
| `avsucd`        | AVS                 | `engineering`   | Advanced Visual Systems visualization                    | AVS-UCD unstructured cell data (`.inp`, `.avs`)    |
| `exodus`        | Sandia SEACAS       | `engineering`   | Sandia FE pre/post toolkit                               | Exodus II, NetCDF-based (`.exo`, `.e`)             |
| `med`           | SALOME / code_aster | `engineering`   | EDF/CEA simulation platform                              | MED-file (`.med`)                                  |
| `medit`         | Medit (INRIA)       | `engineering`   | INRIA mesh viewer                                        | Medit mesh (`.mesh`)                               |
| `netgen`        | Netgen              | `engineering`   | Open-source mesh generator                               | Netgen volume mesh (`.vol`, `.vol.gz`)             |
| `su2`           | SU2                 | `engineering`   | Open-source CFD / multiphysics suite                     | SU2 native mesh (`.su2`)                           |
| `openfoam`      | OpenFOAM            | `engineering`   | Open-source CFD toolbox                                  | OpenFOAM polyMesh (`.foam`)                        |
| `kratos`        | Kratos Multiphysics | `engineering`   | Open-source multiphysics FEM framework                   | Kratos model part (`.mdpa`)                        |
| `flac3d`        | FLAC3D              | `engineering`   | Itasca geomechanics solver                               | FLAC3D grid (`.f3grid`)                            |
| `tecplot`       | Tecplot             | `engineering`   | CFD / engineering visualization                          | Tecplot FE ASCII (`.dat`, `.tec`)                  |
| `vtk`           | VTK (Kitware)       | `generic`       | Visualization Toolkit / ParaView                         | VTK / VTU unstructured grid (`.vtk`, `.vtu`)       |
| `xdmf`          | XDMF                | `generic`       | Open XML + HDF5 data model                               | XDMF (`.xdmf`, `.xmf`)                             |
| `cgns`          | CGNS                | `generic`       | Open CFD data standard                                   | CGNS / HDF5 (`.cgns`)                              |
| `obj`           | Wavefront           | `generic`       | Geometry interchange format                              | Wavefront OBJ (`.obj`)                             |
| `stl`           | -                   | `generic`       | 3D-printing / CAD surface format                         | STL stereolithography (`.stl`)                     |
| `off`           | -                   | `generic`       | Simple geometry format (Geomview)                        | Object File Format (`.off`)                        |
| `ply`           | Stanford            | `generic`       | Polygon geometry format                                  | Stanford PLY (`.ply`)                              |
| `xmstin`        | Aquaveo XMS         | `generic`       | XMS suite triangulated surface                           | XMS / SMS TIN (`.tin`)                             |
| `arcgis-tin`      | Esri (ArcGIS)       | `generic`       | GIS platform; TIN terrain surface                        | Esri TIN, directory of `.adf`                      |
| `triangle`      | Triangle / TetGen   | `generic`       | 2D/3D mesh generators (Shewchuk / Si)                    | Triangle node+ele (`.node`, `.ele`)                |
| `easymesh`      | EasyMesh            | `generic`       | 2D triangular mesh generator (for SWAN)                  | SWAN Easymesh (`.n`, `.e`)                         |

## Suffixes are not unique — pass `driver=` to disambiguate

Several extensions are claimed by more than one driver, so the suffix alone cannot pick a reader:

| Suffix  | Claimed by                                                                            |
|---------|---------------------------------------------------------------------------------------|
| `.nc`   | `3di`, `tuflowfv`, `delft3d-1d`, `dhydro-1d`, `mpas`, `icon`, `geos`, `esmf`, `scrip` |
| `.dat`  | `dat`, `fvcom`, `tecplot`                                                             |
| `.inp`  | `abaqus`, `avsucd`                                                                    |
| `.msh`  | `gmsh`, `gid`                                                                         |
| `.fem`  | `feflow`, `nastran`                                                                   |
| `.mesh` | `mike21`, `medit`                                                                     |
| `.e`    | `exodus`, `easymesh`                                                                  |

For an ambiguous suffix, name the driver explicitly:

```python
from pyramids.netcdf.ugrid import MeshDataset


mesh = MeshDataset.read_file("model.nc", driver="delft3d-1d")
```
