"""Record-oriented parsing helpers for the plain-text formats.

Shared by 2DM, DAT, Mike21 and FLO-2D. Includes ``native_id_map``, which exists because
2DM node and element ids can have gaps and can be out of file order -- a driver assuming
``id == index + 1`` silently mis-wires connectivity on any file a tool has renumbered.

These helpers are pure parsing: they import nothing from ``pyramids`` and are testable
without it installed.

See ``planning/design.md`` section 19. Implemented by task S4.
"""

from __future__ import annotations

from typing import Iterator, Sequence


def iter_records(path: str, *, comment: str = "#") -> Iterator[tuple[str, list[str]]]:
    """Yield one ``(keyword, fields)`` pair per significant line of a text file.

    A significant line is one that is neither blank nor a comment. The line is split
    on runs of whitespace; the first token is the keyword and the rest are the fields,
    in file order. Record-based text formats (2DM, DAT, Mike21, FLO-2D) are all read
    through this.

    Args:
        path: Path to the text file to read.
        comment: Line-comment marker. A line whose first non-blank character is this
            marker is skipped. Defaults to ``"#"``.

    Yields:
        tuple[str, list[str]]: The keyword and the remaining whitespace-separated
            fields of each significant line.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(comment):
                continue
            tokens = stripped.split()
            yield tokens[0], tokens[1:]


def native_id_map(ids: Sequence[int]) -> dict[int, int]:
    """Map file-native element or node ids to contiguous 0-based indices.

    The ids a file assigns to nodes and elements may have gaps and may be out of
    order -- a tool that renumbers a mesh leaves ``ND 7`` as the third node in the
    file. Connectivity must be resolved through this map, never by assuming
    ``id == index + 1``, or every element is wired to the wrong nodes.

    Args:
        ids: The native ids, in the order they appear in the file.

    Returns:
        dict[int, int]: A mapping from each native id to its 0-based index, assigned
            in order of first appearance.

    Raises:
        ValueError: If an id appears more than once, which a well-formed file never
            does and which would make the mapping ambiguous.
    """
    mapping: dict[int, int] = {}
    for index, native in enumerate(ids):
        if native in mapping:
            raise ValueError(f"duplicate native id {native!r} at position {index}")
        mapping[native] = index
    return mapping
