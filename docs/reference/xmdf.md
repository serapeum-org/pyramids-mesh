# XMDF — HDF5 results

The XMDF reader. A results-only HDF5 container, read through the vendored GDAL
multidimensional API (no declared dependency, no `h5py`). It carries values and no
topology, so it attaches to a mesh you already have.

| | |
|---|---|
| Driver name | `xmdf` |
| Suffixes | `.xmdf`, `.h5` |
| Capabilities | `needs_mesh`, `read_data`, `lazy_data` |
| Extra required | none |
| Returns | the variables it read, merged onto the dataset by `attach_data` |

```python
from pyramids.netcdf.ugrid import UgridDataset

dataset = UgridDataset.read_file("domain.2dm")
with_results = dataset.attach_data("results.xmdf")
```

## Structure

The file is an HDF5 tree whose root carries a `File Type` of `Xmdf`. Each dataset is a
group with a `Grouptype` attribute beginning `DATASET`, holding `Values`
`(n_time, n_element)`, a `Times` coordinate, an `Active` wet/dry mask and precomputed
`Mins`/`Maxs`. The tree is walked recursively, so the driver is not tied to one container
layout.

## Documented limitations

- **The element-count check is the only thing tying the results to the mesh.** A dataset's
  `Values` width is matched against the mesh's node or face count to place it; a width
  matching neither is refused, because nothing in the file identifies its mesh.
- **Summary datasets are skipped, not read as timesteps.** A dataset nested under a
  `Final`/`Maximums` container is a single-step snapshot; reading it as a time level would
  fabricate data.
- **The active mask is a standalone face-located `<name>_active` variable**, never folded
  into the values as NaN (design §21.2). It is not carried through `ParsedVariable.active`:
  the mask counts faces while a dataset may be node-located, so the sibling cannot take
  the parent's location or shape. It is attached only when its width is the face count.
- **Values are lazy** — the loader reopens the file per read and seeks to the array, so it
  closes over a path, never an open handle; the shape is read from the header.
- `Mins`/`Maxs` are carried as per-timestep `minimum`/`maximum` attributes.
- **A layered 3D dataset is refused** — a `Values` array with more than two axes is a 3D
  result whose ragged vertical levels are not modelled yet (design §21.1).
- A vector dataset is refused for now; the 2D scalar case ships first.

## API

::: pyramids_mesh.drivers.xmdf.XmdfDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
