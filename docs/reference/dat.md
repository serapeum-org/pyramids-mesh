# DAT — SMS ASCII results

The DAT reader. Plain text carrying **values and no topology**, so it is attached to a mesh you
already have rather than opened on its own.

| | |
|---|---|
| Driver name | `dat` |
| Suffixes | `.dat` |
| Capabilities | `needs_mesh`, `read_data` |
| | `needs_mesh` is what makes `read_data` the entry point: there is no topology in the file, so the mesh has to come from the caller. `read()` refuses and names `attach_data`. |
| Extra required | none |
| Returns | the variables it read, merged onto the dataset by `attach_data` |

```python
from pyramids.netcdf.ugrid import UgridDataset

dataset = UgridDataset.read_file("domain.2dm")
with_results = dataset.attach_data("depth.dat")
```

## The element-count check is the whole safeguard

Nothing inside a DAT file identifies the mesh it belongs to — no name, no checksum, no topology
hash. The only defence is the element count the file declares, so it is checked **before any value
is read**:

- `ND` against the dataset's node count
- `NC` against its face count

A mismatch raises, reporting both numbers. A file declaring **neither** count is refused rather
than attached on hope.

What this cannot catch: two different meshes with the same node and face counts. Attaching a DAT
to the wrong one of those is undetectable from the data, and no driver can fix that — it is a
property of the format.

## Two dialects

| Dialect | Recognised by | Structure |
|---|---|---|
| New | first line is `DATASET` | `BEGSCL`/`BEGVEC` groups, `ND`/`NC`, `NAME`, `TS <istat> <time>`, `ENDDS` |
| Old | no `DATASET` line | one group whose type is the header line (`VECTOR` ⇒ vector), `ND`, bare `TS <time>` blocks; no `BEGSCL`/`ENDDS`/`NC`/`NAME` |

The dialect is decided by the first line alone. The `TS` arity differs — new
`TS <istat> <time>`, old `TS <time>` — so reading an
old file with the new arity would take the *time* for the active flag and the first value for the
time, a plausible one-timestep series instead of an error.

The **old dialect** carries a single node-located group named after the file (it has no `NAME`
card), with no active block. Its `ND` is checked against
the mesh just as the new dialect's counts are.

## Cards

| Card | Meaning |
|---|---|
| `DATASET` | marks the new dialect |
| `OBJTYPE <type>` | recorded only; see the location note below |
| `BEGSCL` / `BEGVEC` | open a scalar / vector group |
| `ND <count>` / `NC <count>` | declared node / cell counts — the check above |
| `NAME <name>` | the group's name, and so the variable's |
| `TIMEUNITS <unit>` | carried as a `time_units` variable attribute |
| `RT_JULIAN <time>` | carried as a `reference_time` variable attribute |
| `TS <istat> <time>` | one timestep; `istat` means an active block follows |
| `ENDDS` | close the group |

A card this driver does not model is ignored: DAT has
accumulated optional cards across SMS versions, and refusing an unknown one would reject files
that read fine.

Malformed structure is refused rather than guessed at — a group opened while another is open, an
`ENDDS` with nothing open, a `NAME`/`TIMEUNITS`/`TS` outside a group, a group with no `TS` at all,
a group left unterminated, a short or non-numeric value block, and two groups producing the same
variable name.

## Documented limitations

- **A vector group yields two scalar variables**, `<name>_x` and `<name>_y`, not one
  two-component variable. `MeshVariable` carries a location and a shape over elements and time,
  with no component axis to put a vector in. Both keep the group name as a prefix so they stay
  recognisable as one quantity.
- **Values are node-located when `ND` is declared**, which is what SMS writes. A file declaring
  only `NC` is read as face-located. With both declared the group is taken as node-located, the SMS
  convention for `BEGSCL`/`BEGVEC`. `OBJTYPE` is recorded but not used to resolve the location —
  no SMS writer puts one in it.
- **The active mask becomes a separate face-located variable** `<name>_active`, never folded into
  the values as NaN (design §21.2). Folding is irreversible and indistinguishable from real
  missing data. It is deliberately *not* attached through `ParsedVariable.active`, because that
  machinery gives the sibling its parent's location and shape, and a DAT active block counts
  *cells* while a node-located group counts *nodes*.
- **The active block's position is the one thing no vendor file was available to confirm.** It is
  read immediately after the `TS` card, before the values. A mismatch is made
  loud rather than silent — the flags must all be `0` or `1`, and a block of real values in their
  place is refused. A file whose node and cell counts are *equal* defeats that check, which is
  recorded here because nothing in the data can catch it.
- **`RT_JULIAN` rides on the variables, not the dataset.** `attach_data` merges the variables a
  driver returns and keeps the dataset's own `global_attributes` untouched, so a driver has no
  route to a file-level attribute. Carrying it properly needs the reference time modelled upstream
  in `pyramids-core`.
- DAT carries **no CRS and no topology**, so neither is produced here.
- There is **no lazy path**. DAT is plain text with no record index to seek, so the driver does not
  declare `lazy_data` and `lazy=` is accepted and ignored.

## API

::: pyramids_mesh.drivers.sms.DatDriver
    options:
        show_root_heading: true
        show_source: true
        heading_level: 3
