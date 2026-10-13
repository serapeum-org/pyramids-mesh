[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/serapeum-org/pyramids-mesh/branch/main/graph/badge.svg)](https://codecov.io/gh/serapeum-org/pyramids-mesh)

# pyramids-mesh

**pyramids-mesh** is the unstructured-mesh tier of the
[pyramids](https://github.com/serapeum-org/pyramids) GIS stack. It extends the UGRID support that
`pyramids-gis` already ships — `pyramids.netcdf.ugrid` — with the operations that tier deliberately
leaves out.

What `pyramids-gis` owns today, and this package therefore does not reimplement:

- **The data model** — `Mesh2d`, `Network1d`, `UgridDataset`, `Connectivity`, and UGRID-compliant
  NetCDF read/write.
- **Topology operations on a mesh already in hand** — connectivity derivation, graph algorithms
  (components, RCM ordering, dilation/erosion, Voronoi, circumcenters, boundary extraction), facet
  remapping, spatial indexing.
- **Regridding and sampling** — point sampling, `rasterize_like`, mesh-to-mesh conservative /
  barycentric / centroid regridding, temporal interpolation.
- **Conversion** — mesh ↔ vector, construction from a raster grid or a polygon layer, xarray/zarr
  hand-off, gap filling.

What this package adds:

- **Generation** — building a mesh where no topology exists yet.
- **Quality** — element metrics (aspect ratio, skewness, orthogonality) and quality-driven
  smoothing.
- **Refinement** — local refinement and coarsening of an existing mesh.

> **Status** — `0.1.0` is the repository scaffold: packaging, CI, docs and the import contract. No
> mesh API is published yet. Follow the [change log](docs/change-log.md) for what lands.

> **Naming note** — the distribution name is `pyramids-mesh` (with hyphen), the Python import name is
> `pyramids_mesh` (with underscore).

## Installation

The package is not published yet. Install it from source with
[Pixi](https://pixi.sh/latest/#installation):

```bash
git clone https://github.com/serapeum-org/pyramids-mesh.git
cd pyramids-mesh
pixi install -e dev      # creates the dev environment
pixi shell -e dev
```

One extra is declared: `viz`, which adds the plotting stack (`pyramids-gis[viz]` + `cleopatra`) that
mesh rendering goes through. The `dev` environment installs it, so the tests run against the same
set an installed user gets.

GDAL is **not** a dependency. The `pyramids-gis` wheel vendors its own osgeo bindings, and
`import pyramids` is what puts them on `sys.path` — which is why `import pyramids` is the first
import in every module here.

Supported Python: **3.11–3.14**.

## Project layout

```
src/pyramids_mesh/   — the package (import name: pyramids_mesh)
tests/               — pytest suite
docs/                — MkDocs sources (MkDocs Material + mkdocstrings)
docs/examples/       — example notebooks, validated by `pixi run notebooks`
examples/data/       — data the notebooks read, via notebook-relative paths
```

## Documentation

- Live site: <https://serapeum-org.github.io/pyramids-mesh/latest/>
- Local preview: `pixi run -e docs mkdocs serve`

## Development

This repository uses [Pixi](https://pixi.sh/) for environment management.

```bash
pixi run main          # main test suite (excludes plot tests)
pixi run plot          # plot/visualization tests
pixi run test-all      # everything, with a terminal coverage report
pixi run notebooks     # validate example notebooks
pre-commit run --all-files
```

Branches and commits follow [Conventional Commits](https://www.conventionalcommits.org/): branch
`<type>/<short-description>`, PR title `<type>: <imperative summary>`. Releases are driven by
commitizen (`pixi run cz-bump`) and the shared `serapeum-org/github-actions` workflows.

## License

GNU General Public License v3 — see [`LICENSE`](./LICENSE).
