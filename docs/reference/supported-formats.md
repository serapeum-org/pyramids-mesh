# Supported formats

Every format `pyramids_mesh` reads. The **driver name** is the handle you pass to
`MeshDataset.read_file(path, driver="…")`; the remaining columns give the software that writes the
file, what that software is, the driver's subpackage, and the file format. See each format's own
[reference page](index.md) for capabilities and documented limitations.

Two naming axes are at work:

- the **module** (the `.py` file) is named for the **software/model** that writes the file
  (`hecras`, `adcirc`, `telemac`…), or for the **format** itself when it is a vendor-neutral standard
  with no single owning tool (`vtk`, `cgns`, `obj`…). A module can hold several drivers (`sms` holds
  both `2dm` and `dat`).
- the **driver name** is the handle passed to `driver=` — the format name where it has a well-known
  one (`2dm`, `dfsu`, `selafin`), the software name where the file is that software's native output
  (`hecras`, `adcirc`, `3di`), or a software + variant where one toolchain emits several grids
  (`modflow6-disv`, `modflow6-disu`, `delft3d-1d`).

| Driver          | Software             | What the software is                                     | Subpackage      | Format                                             |
|-----------------|----------------------|----------------------------------------------------------|-----------------|----------------------------------------------------|
| `adcirc`        | ADCIRC               | Coastal-ocean circulation / storm-surge model            | `coastal_ocean` | ADCIRC `fort.14` ASCII grid (`.14`)                |
| `schism`        | SCHISM               | Cross-scale ocean/coastal circulation model              | `coastal_ocean` | `hgrid.gr3` ASCII grid (`.gr3`)                    |
| `fvcom`         | FVCOM                | Finite-volume coastal-ocean model                        | `coastal_ocean` | FVCOM native grid (`.dat`)                         |
| `selafin`       | TELEMAC              | Free-surface hydrodynamics suite (rivers/coasts)         | `coastal_ocean` | Selafin / Serafin (`.slf`, `.srf`, `.ser`, `.sel`) |
| `sww`           | ANUGA                | Shallow-water flood & tsunami model                      | `coastal_ocean` | SWW, NetCDF (`.sww`)                               |
| `tuflowfv`      | TUFLOW FV            | Flexible-mesh flood/coastal hydrodynamic solver          | `coastal_ocean` | TUFLOW FV results, NetCDF (`.nc`)                  |
| `dfsu`          | DHI MIKE             | DHI's coastal/estuarine hydrodynamic suite               | `coastal_ocean` | DFSU, needs `dhi` extra (`.dfsu`)                  |
| `mike21`        | DHI MIKE 21          | DHI's 2D coastal/estuarine hydrodynamic model            | `coastal_ocean` | MIKE 21 mesh (`.mesh`)                             |
| `3di`           | 3Di                  | Urban/rural flood-modelling platform                     | `coastal_ocean` | 3Di results, NetCDF (`.nc`)                        |
| `hecras`        | HEC-RAS              | USACE river hydraulics & flood model                     | `river_flood`   | HEC-RAS HDF5 plan file (`.hdf`, `.hdf5`)           |
| `flo2d`         | FLO-2D               | Flood-routing / overland-flow model                      | `river_flood`   | FLO-2D (`.out`)                                    |
| `riverflow2d`   | RiverFlow2D          | 2D river/flood hydrodynamic & sediment model             | `river_flood`   | RiverFlow2D mesh (`.fed`)                          |
| `srh2d`         | SRH-2D               | USBR 2D river hydraulics & sedimentation model           | `river_flood`   | SRH-2D (`.srhgeom`, `.srhhydro`)                   |
| `2dm`           | SMS                  | Aquaveo Surface-water Modeling System (GUI)              | `river_flood`   | SMS/TUFLOW 2DM mesh (`.2dm`)                       |
| `dat`           | SMS                  | Aquaveo Surface-water Modeling System (GUI)              | `river_flood`   | SMS ASCII DAT results (`.dat`)                     |
| `xmdf`          | Aquaveo XMS          | Results container for the XMS suite (SMS/SRH-2D)         | `river_flood`   | XMDF HDF5 results (`.xmdf`, `.h5`)                 |
| `delft3d-1d`    | Delft3D FM / D-HYDRO | Deltares' flexible-mesh hydrodynamic suite               | `river_flood`   | UGRID-1.0 + Deltares-extension NetCDF (`.nc`)      |
| `dhydro-1d`     | Delft3D FM / D-HYDRO | Same software, D-HYDRO name; alias of `delft3d-1d`       | `river_flood`   | UGRID-1.0 + Deltares-extension NetCDF (`.nc`)      |
| `gid`           | GiD (Iber)           | CIMNE pre/post-processor; reads the Iber 2D model export | `river_flood`   | GiD / CIMNE ASCII mesh (`.msh`)                    |
| `mpas`          | MPAS                 | Model for Prediction Across Scales (atmos/ocean)         | `earth_system`  | MPAS mesh, NetCDF (`.nc`)                          |
| `icon`          | ICON                 | ICOsahedral Nonhydrostatic weather/climate model         | `earth_system`  | ICON icosahedral grid, NetCDF (`.nc`)              |
| `geos`          | GEOS                 | NASA global atmosphere / earth-system model              | `earth_system`  | GEOS cubed-sphere grid, NetCDF (`.nc`, `.nc4`)     |
| `esmf`          | ESMF                 | Earth System Modeling Framework (model coupling)         | `earth_system`  | ESMF unstructured mesh, NetCDF (`.nc`)             |
| `scrip`         | SCRIP                | Remapping-weights package / grid convention              | `earth_system`  | SCRIP remapping grid, NetCDF (`.nc`)               |
| `modflow6-disv` | MODFLOW 6            | USGS groundwater-flow model                              | `subsurface`    | DISV discretization-by-vertices (`.disv`)          |
| `modflow6-disu` | MODFLOW 6            | USGS groundwater-flow model                              | `subsurface`    | DISU unstructured discretization (`.disu`)         |
| `feflow`        | FEFLOW               | DHI subsurface flow & transport model                    | `subsurface`    | FEFLOW classic ASCII (`.fem`)                      |
| `gmsh`          | Gmsh                 | Open-source finite-element mesh generator                | `engineering`   | Gmsh mesh v2.2/v4.1 (`.msh`)                       |
| `abaqus`        | Abaqus               | Dassault Systemes FEA solver                             | `engineering`   | Abaqus keyword deck (`.inp`)                       |
| `nastran`       | Nastran              | Structural FEA solver                                    | `engineering`   | Nastran bulk data (`.bdf`, `.nas`, `.fem`)         |
| `avsucd`        | AVS                  | Advanced Visual Systems visualization                    | `engineering`   | AVS-UCD unstructured cell data (`.inp`, `.avs`)    |
| `exodus`        | Sandia SEACAS        | Sandia FE pre/post toolkit                               | `engineering`   | Exodus II, NetCDF-based (`.exo`, `.e`)             |
| `med`           | SALOME / code_aster  | EDF/CEA simulation platform                              | `engineering`   | MED-file (`.med`)                                  |
| `medit`         | Medit (INRIA)        | INRIA mesh viewer                                        | `engineering`   | Medit mesh (`.mesh`)                               |
| `netgen`        | Netgen               | Open-source mesh generator                               | `engineering`   | Netgen volume mesh (`.vol`, `.vol.gz`)             |
| `su2`           | SU2                  | Open-source CFD / multiphysics suite                     | `engineering`   | SU2 native mesh (`.su2`)                           |
| `openfoam`      | OpenFOAM             | Open-source CFD toolbox                                  | `engineering`   | OpenFOAM polyMesh (`.foam`)                        |
| `kratos`        | Kratos Multiphysics  | Open-source multiphysics FEM framework                   | `engineering`   | Kratos model part (`.mdpa`)                        |
| `flac3d`        | FLAC3D               | Itasca geomechanics solver                               | `engineering`   | FLAC3D grid (`.f3grid`)                            |
| `tecplot`       | Tecplot              | CFD / engineering visualization                          | `engineering`   | Tecplot FE ASCII (`.dat`, `.tec`)                  |
| `vtk`           | VTK (Kitware)        | Visualization Toolkit / ParaView                         | `generic`       | VTK / VTU unstructured grid (`.vtk`, `.vtu`)       |
| `xdmf`          | XDMF                 | Open XML + HDF5 data model                               | `generic`       | XDMF (`.xdmf`, `.xmf`)                             |
| `cgns`          | CGNS                 | Open CFD data standard                                   | `generic`       | CGNS / HDF5 (`.cgns`)                              |
| `obj`           | Wavefront            | Geometry interchange format                              | `generic`       | Wavefront OBJ (`.obj`)                             |
| `stl`           | -                    | 3D-printing / CAD surface format                         | `generic`       | STL stereolithography (`.stl`)                     |
| `off`           | -                    | Simple geometry format (Geomview)                        | `generic`       | Object File Format (`.off`)                        |
| `ply`           | Stanford             | Polygon geometry format                                  | `generic`       | Stanford PLY (`.ply`)                              |
| `xmstin`        | Aquaveo XMS          | XMS suite triangulated surface                           | `generic`       | XMS / SMS TIN (`.tin`)                             |
| `esri-tin`      | Esri (ArcGIS)        | GIS platform; TIN terrain surface                        | `generic`       | Esri TIN, directory of `.adf`                      |
| `triangle`      | Triangle / TetGen    | 2D/3D mesh generators (Shewchuk / Si)                    | `generic`       | Triangle node+ele (`.node`, `.ele`)                |
| `easymesh`      | EasyMesh             | 2D triangular mesh generator (for SWAN)                  | `generic`       | SWAN Easymesh (`.n`, `.e`)                         |

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
