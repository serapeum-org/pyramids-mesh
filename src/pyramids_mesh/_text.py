"""Record-oriented parsing helpers for the plain-text formats.

Shared by 2DM, DAT, Mike21 and FLO-2D. Includes ``native_id_map``, which exists because
2DM node and element ids can have gaps and can be out of file order -- a driver assuming
``id == index + 1`` silently mis-wires connectivity on any file a tool has renumbered.

See ``planning/design.md`` section 19. Implemented by task S4.
"""
