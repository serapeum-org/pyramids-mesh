# Installation

This page covers installation of **pyramids-mesh** and its native dependencies.

| Item | Value |
|---|---|
| Distribution name | `pyramids-mesh` |
| Python import name | `pyramids_mesh` |
| Current version | `0.1.0` |
| Supported Python | **3.11 – 3.14** |
| License | GPL v3 |

> **Not published yet.** `0.1.0` exists only in this repository — there is no PyPI or conda-forge
> release to install from. Use the from-source route below.

## Dependencies

### Runtime

- `pyramids-gis >= 0.68.0` (provides the `pyramids` import and the mesh-driver registry; pulled from PyPI)
- `numpy >= 2.0.0`

GDAL is deliberately absent from that list. It is not a dependency of this package: the
`pyramids-gis` wheel vendors its own osgeo bindings, and `import pyramids` is what puts them on
`sys.path`. Nothing here installs or pins a separate GDAL.

`scipy`, `shapely` and `geopandas` are not declared either. The UGRID subsystem requires all three,
so they arrive with `pyramids-gis` — but that is its requirement, not a promise to this package.
Each gets declared here the moment a module under `src/pyramids_mesh/` imports it directly.

### Optional extras

| Extra | Purpose | Pulls |
|---|---|---|
| `viz` | mesh rendering / colour tables | `pyramids-gis[viz]`, `cleopatra[tiles]` |

### Development dependency groups

`dev`, `docs` and `notebook` are [PEP 735](https://peps.python.org/pep-0735/) dependency groups, not
extras — they are local tooling and are deliberately not published in the package metadata. Pixi
resolves a group by name wherever an environment lists it, so an environment is a set of groups and
extras rather than a single same-named group: `dev` is `py314` + `dev` + `viz`, `docs` is `py314` +
`docs`. `notebook` is currently referenced by no environment.

```bash
pixi install -e dev      # tests, linting, build tooling
pixi install -e docs     # mkdocs toolchain
```

## Recommended: Pixi

This repository ships a [Pixi](https://pixi.sh/) configuration that resolves every dependency from
PyPI, GDAL included — the `pyramids-gis` wheel vendors its own, so no conda channel is involved and
the usual GDAL-wheel headaches do not arise.

Prerequisites: install [Pixi](https://pixi.sh/latest/#installation).

```bash
git clone https://github.com/serapeum-org/pyramids-mesh.git
cd pyramids-mesh

# Solve and install the dev environment
pixi install -e dev

# Drop into a shell with everything available
pixi shell -e dev

# Or run a task directly
pixi run main          # main test suite
pixi run plot          # plot/visualization tests
pixi run test-all      # everything, with a terminal coverage report
pixi run notebooks     # validate example notebooks
```

### Available Pixi environments

| Environment | Features | Purpose |
|---|---|---|
| `default` | `py314` | minimal runtime |
| `dev` | `py314`, `dev`, `viz` | tests, linting, build tooling |
| `docs` | `py314`, `docs` | docs site (`mkdocs serve`) |
| `py311` | `py311`, `dev`, `viz` | pinned Python 3.11 |
| `py312` | `py312`, `dev`, `viz` | pinned Python 3.12 |
| `py313` | `py313`, `dev`, `viz` | pinned Python 3.13 |
| `py314` | `py314`, `dev`, `viz` | pinned Python 3.14 |

`default`, `dev` and `docs` are not unpinned — they reuse the `py314` feature, so they resolve
Python 3.14 like the matrix environment of that name. `viz` is a PEP 621 extra; `dev`, `docs` and
`notebook` are PEP 735 dependency groups. The definitions live in `pyproject.toml` under
`[tool.pixi.environments]`.

The 3.13 and 3.14 pins carry a build glob (`*_cp313` / `*_cp314`) that excludes the free-threaded
interpreter: `pyramids-gis` ships no `cp313t` / `cp314t` wheel, and with no conda GDAL underneath,
resolving one of those would fail at import rather than at solve time.

### Where the environments live

`.pixi/config.toml` sets `detached-environments` to `C:/python-environments/pixi/envs`, keeping the
multi-GB solved environments off the Google-Drive-synced checkout. That file is local-only (pixi
writes a `.pixi/.gitignore` holding `*`), so each machine chooses its own location; without it, pixi
puts the environments in `.pixi/envs/` inside the repository.

## Editable / development install

```bash
git clone https://github.com/serapeum-org/pyramids-mesh.git
cd pyramids-mesh
pixi install -e dev
pixi run -e dev pre-commit install
```

The package itself is registered as an editable pixi pypi-dependency, so `pixi install -e dev`
already puts your checkout on the path — there is no separate editable-install step.

## Quick check

```python
>>> import pyramids_mesh
>>> pyramids_mesh.__version__
'0.1.0'
>>> from pyramids.netcdf.ugrid import Mesh2d, UgridDataset   # the tier this extends
```

## Notes

- `pyramids` (conda-forge name) and `pyramids-gis` (PyPI name) are the **same package**.
  pyramids-mesh depends on the PyPI distribution name (`pyramids-gis`) so it works regardless of how
  pyramids itself was installed.
- Documentation: <https://serapeum-org.github.io/pyramids-mesh/latest>
- Source repository: <https://github.com/serapeum-org/pyramids-mesh>
