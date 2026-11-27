"""Every committed example notebook must embed no machine-specific absolute path in output.

Re-executing a notebook bakes the executing checkout's absolute paths into cell output -- a
worktree ``src`` path, a pixi ``site-packages`` path, a resolved data path, a user-home or CI
checkout. Such a path is machine-specific, meaningless on the rendered docs site, and leaks the
local layout. ``tests/drivers/cfd/test_cgns.py`` guards this for ``cgns.ipynb`` alone (PR review
SF1/R2-I3); this module generalises the guard to every notebook under ``docs/examples/`` (both the
top-level tutorials and ``real-data/``), so a contaminated re-execution of any of them is caught,
not only cgns.
"""

import pyramids  # noqa: F401  # isort:skip

import json
import re
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).parents[1] / "docs" / "examples"
_NOTEBOOKS = sorted(_EXAMPLES.rglob("*.ipynb"))
_DRIVE_PATH = re.compile(r"[A-Za-z]:[\\/]")
_MACHINE_TOKENS = ("worktrees", "python-environments", "site-packages")


def _notebook_output_text(path: Path) -> str:
    """Return all rendered output text of a notebook, concatenated.

    Joins stream text, execute-result / display-data ``text/plain`` and ``text/html``, and any
    error traceback -- every place a printed or warned path can land -- so one search covers them.

    Args:
        path: The notebook file to read.

    Returns:
        The notebook's concatenated output text (empty when it has no textual output).
    """
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    parts: list[str] = []
    for cell in cells:
        for out in cell.get("outputs", []):
            kind = out.get("output_type")
            if kind == "stream":
                parts.append("".join(out.get("text", "")))
            elif kind in ("execute_result", "display_data"):
                data = out.get("data", {})
                parts.append("".join(data.get("text/plain", [])))
                parts.append("".join(data.get("text/html", [])))
            elif kind == "error":
                parts.append("\n".join(out.get("traceback", [])))
    return "\n".join(parts)


@pytest.mark.skipif(not _NOTEBOOKS, reason="no example notebooks present")
@pytest.mark.parametrize(
    "notebook", _NOTEBOOKS, ids=lambda p: str(p.relative_to(_EXAMPLES))
)
def test_example_notebook_commits_no_machine_specific_path(notebook: Path):
    """A committed example notebook's output carries only relative module/repo paths.

    Test scenario:
        Concatenate the notebook's rendered output and assert it holds neither a drive-letter
        absolute path nor any machine-layout token a re-execution would leak.
    """
    output = _notebook_output_text(notebook)
    assert not _DRIVE_PATH.search(output), (
        f"{notebook.name} output embeds a drive-letter absolute path"
    )
    for needle in _MACHINE_TOKENS:
        assert needle not in output, (
            f"{notebook.name} output leaks a machine path token: {needle!r}"
        )
