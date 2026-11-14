"""The repo's packages only import in allowed directions, so the two pipelines stay separate."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import REPO

# What each package may import from the repo's own packages.
ALLOWED = {
    "shared": set(),
    "data_pipeline": {"shared"},
    "ml_pipeline": {"shared"},
    "api": {"shared"},
}


def repo_imports(path: Path) -> set[str]:
    """The top-level repo packages a file imports."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found.update(name.split(".")[0] for name in names if name.split(".")[0] in ALLOWED)
    return found


@pytest.mark.parametrize("package", sorted(ALLOWED))
def test_packages_import_only_what_they_are_allowed_to(package: str) -> None:
    for path in sorted((REPO / package).glob("*.py")):
        others = repo_imports(path) - {package}
        assert others <= ALLOWED[package], f"{path.relative_to(REPO)} imports {sorted(others)}"
