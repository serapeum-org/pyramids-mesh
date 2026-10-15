# ADR-0003: Scope boundary with pyramids-gis

- Status: Superseded by [ADR-0005](0005-file-format-drivers.md)
- Date: 2026-10-13

> **Superseded 2026-10-13.** The three-concerns decision below (generation, quality,
> refinement) was written during scaffolding and was not the intended scope. This package is
> a file-format driver package; see [ADR-0005](0005-file-format-drivers.md), which restates
> every boundary rule from this ADR unchanged. Kept for the record; do not act on its
> Decision section.

## Context

`pyramids-gis` already ships UGRID support in `pyramids.netcdf.ugrid`: the data model (`Mesh2d`,
`Network1d`, `UgridDataset`, `Connectivity`), UGRID-compliant NetCDF I/O, connectivity derivation
and graph algorithms, sampling and mesh-to-mesh regridding, construction by flattening a raster grid
or a polygon layer, mesh-to-vector conversion, gap filling, spatial indexing, and rendering through
cleopatra. This repository exists because three things are missing from that list, not because the
mesh model needs a second implementation.

Without a stated boundary, the obvious failure mode is re-deriving connectivity or re-wrapping the
NetCDF I/O here, which gives the stack two mesh types that drift apart.

## Decision

- `pyramids-gis` remains the **single owner** of the mesh data model, its persistence, and every
  operation that follows from a topology already in hand. This package imports those types; it never
  redefines them.
- `pyramids-mesh` owns three concerns:
    - **Generation** — producing a topology where none exists yet (as opposed to flattening a grid
      or polygon layer that already carries one, which is `ugrid.construct`).
    - **Quality** — per-element metrics (aspect ratio, skewness, orthogonality) and quality-driven
      smoothing.
    - **Refinement** — local refinement and coarsening of an existing mesh.
- Public functions take and return `pyramids.netcdf.ugrid` types, so a result can go straight back
  into a `UgridDataset` and be written as UGRID NetCDF without a conversion step.
- A gap or bug on the pyramids side of the boundary is reported and fixed upstream. A local
  workaround, if one is unavoidable, carries a comment naming the upstream issue and is removed when
  that lands.
- Plotting goes through cleopatra, reached via pyramids — this package declares no plotting code of
  its own beyond the `viz` extra.

## Consequences

- The dependency floor (`pyramids-gis >= 0.66.0`) is an API floor: raising it is the normal way to
  pick up an upstream mesh fix.
- Some features will be rejected here and routed upstream instead; the table in the
  [docs home page](../index.md) is the quick test for which side a concern falls on.
- Interoperability is free — anything this package produces is already a UGRID mesh that the rest of
  the stack reads.
