## 0.1.0 (2026-10-16)

### Feat

- **drivers**: add the DAT, Mike21, Selafin, SWW, PLY and TIN readers (#11)
- **drivers**: add the PLY and XMS TIN geometry readers
- **drivers**: add the ANUGA SWW reader
- **drivers**: add the Selafin (TELEMAC) binary reader
- **drivers**: add the Mike21 mesh reader
- **drivers**: add the DAT results reader
- **drivers**: add the mesh driver framework and the 2DM reader (#3)
- add the 2DM reader
- add the text and binary parsing helpers
- add BaseDriver template method and ParsedMesh intermediate

### Fix

- **dat**: implement the old (pre-DATASET) dialect properly
- **interchange**: validate PLY face properties before reading them
- **selafin**: translate lazy-load errors and drop a dead guard
- **sww,mike21**: catch invalid-CRS validation and detect the base robustly
- **dat**: drop reference-time metadata from the active-mask sibling
- **interchange**: name the missing-count errors in PLY and XMS TIN
- **mike21**: preserve the projection verbatim and scope osr exceptions
- **sww**: resolve dimensions by name, guard node_z, scope GDAL exceptions
- **selafin**: correct the parallel guard, 3D handling and error type
- **validate**: refuse a 2D topology that carries no faces
- correct the package docstring, the warning frames and the sniff guard
- **sms**: bound the material columns and stop inventing mixed elevation
- **docs**: make the multi_mesh example name its topologies
- **base**: refuse a mesh_name the driver does not offer
- **validate**: complete the connectivity checks the network split left behind
- validate the active mask and report the surface mesh a 1D read drops
- **sms**: never let the sniff judge a line it has not finished reading
- **validate**: size the degeneracy tolerance by resolution as well as span
- **test**: make the sniff-limit test exercise the limit it names
- **sms**: read a BOM and a long comment header instead of declining them
- **validate**: measure winding in the face's own frame, not the origin's
- **base**: honour the never-raises contract and reject a half-declared driver
- **validate**: decide degeneracy geometrically and report reversals
- **binary**: reject a wrong record count and diagnose truncation correctly
- **sms**: report malformed fields and align the sniff with the parser
- **base**: correct the synthesised variables and the name collisions they cause
- **validate**: close the three gaps that made shape untrustworthy
- make the 1D topology path reachable and its losses explicit
- **validate**: settle the two index bases so the result cannot contradict itself
- align the 2DM header sniff with the record parser
