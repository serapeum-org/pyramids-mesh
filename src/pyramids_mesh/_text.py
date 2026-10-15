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

    Examples:
        - Read the records of a tiny 2DM file, with a comment and a blank line
          dropped along the way:

            >>> import os
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "mesh.2dm")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         print("MESH2D", file=handle)
            ...         print("", file=handle)
            ...         print("# written by some tool", file=handle)
            ...         print("ND 1 0.0 0.0 1.0", file=handle)
            ...     list(iter_records(path))
            [('MESH2D', []), ('ND', ['1', '0.0', '0.0', '1.0'])]

        - The keyword is separated from its fields, so a driver can switch on it:

            >>> import os
            >>> import tempfile
            >>> with tempfile.TemporaryDirectory() as directory:
            ...     path = os.path.join(directory, "elements.2dm")
            ...     with open(path, "w", encoding="utf-8") as handle:
            ...         print("E3T 1 1 2 3 7", file=handle)
            ...     keyword, fields = next(iter_records(path))
            >>> keyword
            'E3T'
            >>> fields[1:4]
            ['1', '2', '3']
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

    Examples:
        - Gapped ids become contiguous indices in order of appearance:

            >>> native_id_map([1, 5, 9, 12])
            {1: 0, 5: 1, 9: 2, 12: 3}

        - Ids out of file order map to their position, not their value, which is
          what keeps a renumbered mesh wired correctly:

            >>> mapping = native_id_map([7, 3, 5])
            >>> mapping[7]
            0
            >>> [mapping[native] for native in (7, 3, 5)]
            [0, 1, 2]

        - A repeated id is ambiguous and refused:

            >>> native_id_map([4, 4])
            Traceback (most recent call last):
                ...
            ValueError: duplicate native id 4 at position 1
    """
    mapping: dict[int, int] = {}
    for index, native in enumerate(ids):
        if native in mapping:
            raise ValueError(f"duplicate native id {native!r} at position {index}")
        mapping[native] = index
    return mapping
