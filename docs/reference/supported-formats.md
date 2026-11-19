# Supported formats

Every format `pyramids_mesh` reads. The **driver name** is the handle you pass to
`MeshDataset.read_file(path, driver="…")`; the remaining columns give the software that writes the
file, its driver subpackage, what that software is, and the file format. See each format's own
[reference page](index.md) for capabilities and documented limitations.

## Naming conventions

Three identifiers, three rules — and each is named for the **software or the format, never the
company**.

**Module** — the `.py` file. A Python identifier, so `snake_case` (underscores, never hyphens).

- Named for the **software/model** that writes the file (`hecras`, `adcirc`, `modflow`), or for the
  **format** itself when it is a vendor-neutral standard with no single owning tool (`vtk`, `cgns`,
  `obj`).
- **Never a company name** — use the software, not the vendor: `delft3d` (not `deltares`), `mike`
  (not `dhi`), `arcgis` (not `esri`).
- One module may expose several drivers (`sms` holds both `2dm` and `dat`).

**Driver name** — the handle you pass to `driver=`.

- The **format's own name** when the format has a distinctive, well-known one (`selafin`, `dfsu`,
  `2dm`, `vtk`, `obj`); otherwise the **software/model name** when the file is that software's native
  output (`hecras`, `adcirc`, `3di`); or **software + variant** when one toolchain emits several grids
  (`modflow6-disv`, `modflow6-disu`).
- The software name also **disambiguates a shared extension** (`.msh` → `gmsh`/`gid`, `.inp` →
  `abaqus`/`avs-ucd`).
- **Never a company name** (`delft3d-1d` / `dhydro-1d`, not `deltares…`; `arcgis-tin`, not `esri-tin`).
- **Multi-word names are hyphenated, never underscored** (`delft3d-1d`, `arcgis-tin`,
  `modflow6-disv`); a single compact token stays one word (`2dm`, `3di`, `flo2d`, `srh2d`).

**Subpackage** — the domain bucket. A Python identifier, so `snake_case`.

- A domain/realm word: `coastal_ocean`, `river_flood`, `earth_system`, `subsurface`, `engineering`,
  `generic`. **Never a company name.**

| Driver          | Module        | Software            | Subpackage      | What the software is                                     | Format                                             |
|-----------------|---------------|---------------------|-----------------|----------------------------------------------------------|----------------------------------------------------|
| `adcirc`        | `adcirc`      | ADCIRC              | `coastal_ocean` | Coastal-ocean circulation / storm-surge model            | ADCIRC `fort.14` ASCII grid (`.14`)                |
| `schism`        | `schism`      | SCHISM              | `coastal_ocean` | Cross-scale ocean/coastal circulation model              | `hgrid.gr3` ASCII grid (`.gr3`)                    |
| `fvcom`         | `fvcom`       | FVCOM               | `coastal_ocean` | Finite-volume coastal-ocean model                        | FVCOM native grid (`.dat`)                         |
| `selafin`       | `telemac`     | TELEMAC             | `coastal_ocean` | Free-surface hydrodynamics suite (rivers/coasts)         | Selafin / Serafin (`.slf`, `.srf`, `.ser`, `.sel`) |
| `sww`           | `anuga`       | ANUGA               | `coastal_ocean` | Shallow-water flood & tsunami model                      | SWW, NetCDF (`.sww`)                               |
| `tuflowfv`      | `tuflowfv`    | TUFLOW FV           | `coastal_ocean` | Flexible-mesh flood/coastal hydrodynamic solver          | TUFLOW FV results, NetCDF (`.nc`)                  |
| `dfsu`          | `mike`        | DHI MIKE            | `coastal_ocean` | DHI's coastal/estuarine hydrodynamic suite               | DFSU, needs `dhi` extra (`.dfsu`)                  |
| `mike21`        | `mike21`      | DHI MIKE 21         | `coastal_ocean` | DHI's 2D coastal/estuarine hydrodynamic model            | MIKE 21 mesh (`.mesh`)                             |
| `3di`           | `threedi`     | 3Di                 | `coastal_ocean` | Urban/rural flood-modelling platform                     | 3Di results, NetCDF (`.nc`)                        |
| `hecras`        | `hecras`      | HEC-RAS             | `river_flood`   | USACE river hydraulics & flood model                     | HEC-RAS HDF5 plan file (`.hdf`, `.hdf5`)           |
| `flo2d`         | `flo2d`       | FLO-2D              | `river_flood`   | Flood-routing / overland-flow model                      | FLO-2D (`.out`)                                    |
| `riverflow2d`   | `riverflow2d` | RiverFlow2D         | `river_flood`   | 2D river/flood hydrodynamic & sediment model             | RiverFlow2D mesh (`.fed`)                          |
| `srh2d`         | `srh2d`       | SRH-2D              | `river_flood`   | USBR 2D river hydraulics & sedimentation model           | SRH-2D (`.srhgeom`, `.srhhydro`)                   |
| `2dm`           | `sms`         | SMS                 | `river_flood`   | Aquaveo Surface-water Modeling System (GUI)              | SMS/TUFLOW 2DM mesh (`.2dm`)                       |
| `dat`           | `sms`         | SMS                 | `river_flood`   | Aquaveo Surface-water Modeling System (GUI)              | SMS ASCII DAT results (`.dat`)                     |
| `xmdf`          | `xms`         | Aquaveo XMS         | `river_flood`   | Results container for the XMS suite (SMS/SRH-2D)         | XMDF HDF5 results (`.xmdf`, `.h5`)                 |
| `delft3d-1d`    | `delft3d`     | Delft3D FM          | `river_flood`   | Deltares' flexible-mesh hydrodynamic suite               | UGRID-1.0 + Deltares-extension NetCDF (`.nc`)      |
| `dhydro-1d`     | `delft3d`     | D-HYDRO             | `river_flood`   | Same software, D-HYDRO name; alias of `delft3d-1d`       | UGRID-1.0 + Deltares-extension NetCDF (`.nc`)      |
| `gid`           | `gid`         | GiD (Iber)          | `river_flood`   | CIMNE pre/post-processor; reads the Iber 2D model export | GiD / CIMNE ASCII mesh (`.msh`)                    |
| `mpas`          | `mpas`        | MPAS                | `earth_system`  | Model for Prediction Across Scales (atmos/ocean)         | MPAS mesh, NetCDF (`.nc`)                          |
| `icon`          | `icon`        | ICON                | `earth_system`  | ICOsahedral Nonhydrostatic weather/climate model         | ICON icosahedral grid, NetCDF (`.nc`)              |
| `geos`          | `geos`        | GEOS                | `earth_system`  | NASA global atmosphere / earth-system model              | GEOS cubed-sphere grid, NetCDF (`.nc`, `.nc4`)     |
| `esmf`          | `esmf`        | ESMF                | `earth_system`  | Earth System Modeling Framework (model coupling)         | ESMF unstructured mesh, NetCDF (`.nc`)             |
| `scrip`         | `scrip`       | SCRIP               | `earth_system`  | Remapping-weights package / grid convention              | SCRIP remapping grid, NetCDF (`.nc`)               |
| `modflow6-disv` | `modflow`     | MODFLOW 6           | `subsurface`    | USGS groundwater-flow model                              | DISV discretization-by-vertices (`.disv`)          |
| `modflow6-disu` | `modflow`     | MODFLOW 6           | `subsurface`    | USGS groundwater-flow model                              | DISU unstructured discretization (`.disu`)         |
| `feflow`        | `feflow`      | FEFLOW              | `subsurface`    | DHI subsurface flow & transport model                    | FEFLOW classic ASCII (`.fem`)                      |
| `gmsh`          | `gmsh`        | Gmsh                | `engineering`   | Open-source finite-element mesh generator                | Gmsh mesh v2.2/v4.1 (`.msh`)                       |
| `abaqus`        | `abaqus`      | Abaqus              | `engineering`   | Dassault Systemes FEA solver                             | Abaqus keyword deck (`.inp`)                       |
| `nastran`       | `nastran`     | Nastran             | `engineering`   | Structural FEA solver                                    | Nastran bulk data (`.bdf`, `.nas`, `.fem`)         |
| `avs-ucd`       | `avs`         | AVS                 | `engineering`   | Advanced Visual Systems visualization                    | AVS-UCD unstructured cell data (`.inp`, `.avs`)    |
| `exodus`        | `seacas`      | Sandia SEACAS       | `engineering`   | Sandia FE pre/post toolkit                               | Exodus II, NetCDF-based (`.exo`, `.e`)             |
| `med`           | `salome`      | SALOME / code_aster | `engineering`   | EDF/CEA simulation platform                              | MED-file (`.med`)                                  |
| `medit`         | `medit`       | Medit (INRIA)       | `engineering`   | INRIA mesh viewer                                        | Medit mesh (`.mesh`)                               |
| `netgen`        | `netgen`      | Netgen              | `engineering`   | Open-source mesh generator                               | Netgen volume mesh (`.vol`, `.vol.gz`)             |
| `su2`           | `su2`         | SU2                 | `cfd`           | Open-source CFD / multiphysics suite                     | SU2 native mesh (`.su2`)                           |
| `openfoam`      | `openfoam`    | OpenFOAM            | `cfd`           | Open-source CFD toolbox                                  | OpenFOAM polyMesh (`.foam`)                        |
| `kratos`        | `kratos`      | Kratos Multiphysics | `engineering`   | Open-source multiphysics FEM framework                   | Kratos model part (`.mdpa`)                        |
| `flac3d`        | `flac3d`      | FLAC3D              | `engineering`   | Itasca geomechanics solver                               | FLAC3D grid (`.f3grid`)                            |
| `tecplot`       | `tecplot`     | Tecplot             | `cfd`           | CFD / engineering visualization                          | Tecplot FE ASCII (`.dat`, `.tec`)                  |
| `vtk`           | `vtk`         | VTK (Kitware)       | `generic`       | Visualization Toolkit / ParaView                         | VTK / VTU unstructured grid (`.vtk`, `.vtu`)       |
| `xdmf`          | `xdmf`        | XDMF                | `cfd`           | Open XML + HDF5 data model                               | XDMF (`.xdmf`, `.xmf`)                             |
| `cgns`          | `cgns`        | CGNS                | `cfd`           | Open CFD data standard                                   | CGNS / HDF5 (`.cgns`)                              |
| `obj`           | `obj`         | Wavefront           | `generic`       | Geometry interchange format                              | Wavefront OBJ (`.obj`)                             |
| `stl`           | `stl`         | -                   | `generic`       | 3D-printing / CAD surface format                         | STL stereolithography (`.stl`)                     |
| `off`           | `off`         | -                   | `generic`       | Simple geometry format (Geomview)                        | Object File Format (`.off`)                        |
| `ply`           | `interchange` | Stanford            | `generic`       | Polygon geometry format                                  | Stanford PLY (`.ply`)                              |
| `xms-tin`       | `interchange` | Aquaveo XMS         | `generic`       | XMS suite triangulated surface                           | XMS / SMS TIN (`.tin`)                             |
| `arcgis-tin`    | `arcgis`      | Esri (ArcGIS)       | `generic`       | GIS platform; TIN terrain surface                        | Esri TIN, directory of `.adf`                      |
| `triangle`      | `triangle`    | Triangle / TetGen   | `generic`       | 2D/3D mesh generators (Shewchuk / Si)                    | Triangle node+ele (`.node`, `.ele`)                |
| `easymesh`      | `easymesh`    | EasyMesh            | `generic`       | 2D triangular mesh generator (for SWAN)                  | SWAN Easymesh (`.n`, `.e`)                         |

## Suffixes are not unique — pass `driver=` to disambiguate

Several extensions are claimed by more than one driver, so the suffix alone cannot pick a reader:

| Suffix  | Claimed by                                                                            |
|---------|---------------------------------------------------------------------------------------|
| `.nc`   | `3di`, `tuflowfv`, `delft3d-1d`, `dhydro-1d`, `mpas`, `icon`, `geos`, `esmf`, `scrip` |
| `.dat`  | `dat`, `fvcom`, `tecplot`                                                             |
| `.inp`  | `abaqus`, `avs-ucd`                                                                    |
| `.msh`  | `gmsh`, `gid`                                                                         |
| `.fem`  | `feflow`, `nastran`                                                                   |
| `.mesh` | `mike21`, `medit`                                                                     |
| `.e`    | `exodus`, `easymesh`                                                                  |

For an ambiguous suffix, name the driver explicitly:

```python
from pyramids.netcdf.ugrid import MeshDataset


mesh = MeshDataset.read_file("model.nc", driver="delft3d-1d")
```
