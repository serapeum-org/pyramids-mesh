# ADR-0005: pyramids-mesh is a file-format driver package

- Status: Accepted
- Date: 2026-10-13
- Supersedes: [ADR-0003](0003-scope-boundary.md)

## Context

ADR-0003 scoped this package to mesh **generation**, **quality** and **refinement**. That scope was
written during scaffolding, before the gap analysis it was meant to serve, and it does not match the
work this repository exists to do.

The actual gap is **format breadth**. A survey of MDAL — the Mesh Data Abstraction Library that
QGIS uses, and the reference implementation in this space — found that `pyramids-gis` reads exactly
one of MDAL's eighteen driver families: UGRID NetCDF. The other seventeen are the native file
formats of hydrodynamic and hydraulic solvers — SMS, TUFLOW, TELEMAC, DHI MIKE, FLO-2D, 3Di, ANUGA,
HEC-RAS — and none of them can be opened today.

That survey also found the comparison runs the other way for everything else: MDAL performs no
interpolation, regridding, reprojection, contouring, gap filling, mesh-to-vector conversion or graph
analysis. `pyramids-gis` already has all of it. So the only substantial capability MDAL has over
this stack is **the ability to open other people's mesh files**.

Reading a format is never out of scope for the stack, but seventeen parsers do not belong in
`pyramids-gis` itself: they carry dependencies the core must not take (`mikeio` alone pulls pandas,
matplotlib, scipy and tqdm), and their test fixtures are binary files measured in megabytes against
a PyPI project that already refills its 10 GB limit roughly every twelve releases.

## Decision

**`pyramids-mesh` reads and writes the file formats of hydrodynamic and hydraulic mesh models —
2DM, Selafin, DFSU, XMDF, PLY and the rest of MDAL's driver table — as `pyramids.netcdf.ugrid`
objects. It owns file-format drivers; every mesh operation belongs upstream in `pyramids-gis`.**

- **Generation, quality and refinement are dropped**, not deferred. They were not an intended scope.
  If mesh generation is wanted later it gets its own package and its own argument.
- Drivers are discovered by `pyramids-gis` through the `pyramids.mesh_drivers` **entry-point group**.
  This package depends on `pyramids-gis`; `pyramids-gis` never imports this package. That registry
  shipped in `pyramids-gis 0.68.0`, which is this package's floor, so registered drivers are
  discovered and dispatched to.
- Each driver is a class declaring `name`, `long_name`, `suffixes`, `api_version` and a
  `capabilities` set, implementing `can_open` and `mesh_names` — which pyramids calls on every
  driver whatever it declares — plus either `read` (a format carrying topology) or `read_data` (a
  results-only format such as DAT, XMDF or XDMF, which carries values but no mesh). Only
  `read_mesh`, `needs_mesh`, `write_mesh` and `write_data` gate behaviour upstream; `read_data`,
  `lazy_data` and `multi_mesh` are **declarative** and require no method of their own.
- The one dependency-backed driver family lives behind an extra — `[dhi]` (`mikeio`) — and a
  driver whose extra is absent is skipped at discovery, so a bare install stays small and useful.
  Mesh reading itself is never delegated to a third-party mesh library: that is `pyramids-gis`'s,
  and a generic capability missing there is added there. The `[interchange]` extra (`meshio`) was
  dropped for this reason; PLY and the TIN formats are parsed here. `[hdf]` (`h5py`) went the
  same way: HDF5 is read through the GDAL multidimensional API over the GDAL vendored in the
  `pyramids-gis` wheel, which supplies group traversal, attributes and typed reads. `mikeio`
  remains because DFSU and DFS2 are proprietary DHI containers with no portable alternative.
- Test fixtures are hand-built minimal files, committed, and excluded from the sdist by
  `MANIFEST.in`'s `prune tests`. Vendor sample
  files are fetched in CI behind a marker, never committed.

## The boundary (carried over from ADR-0003 unchanged)

Every rule below was correct in ADR-0003 and is restated here because that ADR is superseded in
full. Only its three-concerns decision was wrong.

- `pyramids-gis` remains the **single owner** of the mesh data model, its persistence, and every
  operation that follows from a topology already in hand. This package imports those types; it never
  redefines them.
- Public functions take and return `pyramids.netcdf.ugrid` types, so a result can go straight back
  into a `UgridDataset` and be written as UGRID NetCDF without a conversion step.
- A gap or bug on the pyramids side of the boundary is reported and fixed upstream. A local
  workaround, if one is unavoidable, carries a comment naming the upstream issue and is removed when
  that lands.
- Plotting goes through cleopatra, reached via pyramids. This package declares no plotting code —
  and, with the `viz` extra removed, no plotting dependency either: a driver returns data, and the
  caller plots it.

## Consequences

- The `pyramids-gis` floor is an API floor on the **registry**, not on the mesh model. The registry
  shipped in `pyramids-gis 0.68.0`, so the floor is now `>= 0.68.0` and this package's entry points
  are live: a driver it registers is dispatched to. The earlier hold on publishing to PyPI — kept
  while the registry was unreleased and the entry points inert — is lifted.
- Three concerns MDAL supports and `pyramids-gis` does not — ragged vertical levels
  (`DataOnVolumes`), per-timestep active/dry flags, and a dataset reference time — are **upstream**
  work. A driver meeting them carries the value through without fabricating data: it raises on a
  genuinely ragged vertical profile rather than padding it, and attaches an active mask as a sibling
  variable rather than folding it into the values as NaN.
- DFS2 is a structured grid rather than a mesh, so it is the one in-scope format whose natural
  target is a raster `Dataset`. Whether it belongs here at all is deliberately left open.
- The `viz` extra is removed, and generation/quality/refinement leave the documentation.

## References

- `planning/design.md` — the full architecture: driver protocol, registry and dispatch, the model
  contract, per-format specifications, and the task plan.
- `planning/mdal-comparison.md` — MDAL's driver table and the capability comparison this decision
  rests on.
