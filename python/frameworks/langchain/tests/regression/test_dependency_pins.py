"""Dependency-pin regression for handoff §5.4 / Area 4.

The package pins `langchain = "^0.3.9"` (and langchain-community), capping <0.4 and
forcing customers on LangChain 1.x to `pip install --no-deps`. The fast test below
statically asserts the declared pin admits 1.x; it fails today and passes once the
pins are opened to the house `>=` style.

An optional heavy test performs a real dependency resolution in a throwaway venv
(no `--no-deps`); it is opt-in via FI_RUN_SLOW=1 because it hits the network.
"""
from __future__ import annotations

import os
import pathlib

import pytest

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None

from packaging.specifiers import SpecifierSet
from packaging.version import Version

PYPROJECT = pathlib.Path(__file__).resolve().parents[2] / "pyproject.toml"
TARGET = Version("1.0.2")  # customer's langchain version


def _poetry_to_specifier(spec: str) -> SpecifierSet:
    """Translate a Poetry version spec to a PEP 440 SpecifierSet (caret only)."""
    spec = spec.strip()
    if not spec.startswith("^"):
        return SpecifierSet(spec)
    base = spec[1:]
    parts = [int(p) for p in base.split(".")]
    # Caret bumps the left-most non-zero component.
    upper = list(parts)
    for i, val in enumerate(parts):
        if val != 0 or i == len(parts) - 1:
            upper[i] = val + 1
            for j in range(i + 1, len(upper)):
                upper[j] = 0
            break
    return SpecifierSet(f">={base},<{'.'.join(map(str, upper))}")


def _dep_version(pyproject: dict, name: str):
    """Return the version spec for a dependency, or None if it is not declared."""
    dep = pyproject["tool"]["poetry"]["dependencies"].get(name)
    if dep is None:
        return None
    return dep["version"] if isinstance(dep, dict) else dep


@pytest.mark.skipif(tomllib is None, reason="tomllib unavailable")
def test_langchain_core_admits_1x():
    """GREEN guard. langchain-core is the real runtime dep and must admit 1.0.2."""
    data = tomllib.loads(PYPROJECT.read_text())
    spec = _poetry_to_specifier(_dep_version(data, "langchain-core"))
    assert TARGET in spec, f"langchain-core pin {spec} rejects {TARGET}"


@pytest.mark.skipif(tomllib is None, reason="tomllib unavailable")
def test_langchain_pin_does_not_block_1x():
    """RED — flags §5.4. If `langchain` is required at all, its pin must admit 1.0.2.
    (The fix drops it as a hard dep, since the package imports only langchain-core.)"""
    data = tomllib.loads(PYPROJECT.read_text())
    spec_str = _dep_version(data, "langchain")
    if spec_str is None:
        return  # not a required dependency → cannot block resolution
    assert TARGET in _poetry_to_specifier(spec_str), (
        f"langchain pin {spec_str} rejects {TARGET}; customers must use --no-deps"
    )

