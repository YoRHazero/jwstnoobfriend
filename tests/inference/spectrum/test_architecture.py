"""Architecture rules of the spectrum package (T12.4-T12.6).

- Layers: enforced by the import-linter contract in ``pyproject.toml``.
- Private names: a ``_`` module is importable only inside the package that
  contains it, and a ``_`` name only from within the importing module's own
  subpackage. (Not ruff's PLC2701: it is preview-only and also flags private
  imports inside one subpackage.)
- Importing the package loads neither JAX, NumPyro, ArviZ nor matplotlib.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import noobfriend.inference.spectrum as spectrum

PACKAGE = "noobfriend.inference.spectrum"
ROOT = Path(spectrum.__file__).parent
PROJECT = ROOT.parents[3]


def _module_name(path: Path) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((PACKAGE, *parts))


def _package_of(module: str, path: Path) -> str:
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _subpackage(module: str) -> str:
    parts = module[len(PACKAGE) + 1 :].split(".")
    return parts[0] if parts and parts[0] else ""


def _private_violations() -> list[str]:
    violations = []
    for path in sorted(ROOT.rglob("*.py")):
        module = _module_name(path)
        package = _package_of(module, path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                targets = [(node.module, [alias.name for alias in node.names])]
            elif isinstance(node, ast.Import):
                targets = [(alias.name, []) for alias in node.names]
            else:
                continue
            for target, names in targets:
                if not target.startswith(PACKAGE):
                    continue
                parts = target.split(".")
                for index, part in enumerate(parts):
                    if part.startswith("_"):
                        owner = ".".join(parts[:index])
                        if not (package == owner or package.startswith(owner + ".")):
                            violations.append(
                                f"{module}:{node.lineno} imports private module {target}"
                            )
                        break
                if _subpackage(target) != _subpackage(module):
                    for name in names:
                        if name.startswith("_"):
                            violations.append(
                                f"{module}:{node.lineno} imports private name {name} from {target}"
                            )
    return violations


def test_private_names_stay_inside_their_package() -> None:
    assert _private_violations() == []


def test_layer_contract() -> None:
    importlinter = pytest.importorskip("importlinter.cli")
    code = importlinter.lint_imports(
        config_filename=str(PROJECT / "pyproject.toml"), no_cache=True, no_logo=True
    )
    assert code == 0


def test_import_is_light() -> None:
    script = (
        "import os, sys\n"
        "os.environ.pop('XLA_FLAGS', None)\n"
        "import noobfriend.inference.spectrum, noobfriend.inference.spectrum.mcmc\n"
        "import noobfriend.inference.spectrum.mle\n"
        "heavy = sorted({m.split('.')[0] for m in sys.modules} & "
        "{'jax', 'numpyro', 'arviz', 'matplotlib'})\n"
        "print(heavy, 'xla_force_host_platform_device_count' in os.environ.get('XLA_FLAGS', ''))\n"
    )
    output = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert output == "[] True"


def test_top_level_exports_declaration_names_only() -> None:
    assert sorted(spectrum.__all__) == sorted(
        [
            "AR1",
            "Autocorrelation",
            "LogUniform",
            "Noise",
            "NoobContinuum",
            "NoobLine",
            "NoobSpectrum",
            "Normal",
            "TruncatedNormal",
            "Uniform",
        ]
    )
