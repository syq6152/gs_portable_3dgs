# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Check Windows package startup and its manifest-derived version output."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def check_portable_startup(package_root: Path) -> None:
    executable = package_root / "SwapTexture.exe"
    if not executable.is_file():
        raise RuntimeError(f"Portable executable is missing: {executable}")
    if os.name != "nt":
        raise RuntimeError("Portable executable startup check requires Windows")

    try:
        manifest = json.loads((package_root / "manifest.json").read_text(encoding="utf-8-sig"))
        version = manifest["extension"]["packageVersion"]
        if not isinstance(version, str) or not version:
            raise ValueError("extension.packageVersion must be a nonempty string")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"Invalid package version manifest: {error}") from error

    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join(
        (str(executable.parent), str(system_root / "System32"), str(system_root))
    )
    try:
        result = subprocess.run(
            [str(executable), "--version"], cwd=executable.parent,
            env=environment, capture_output=True, text=True, errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"Portable executable startup failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            f"Portable executable startup failed: exit={result.returncode}; {detail}"
        )
    expected = f"Swaptexture v{version}"
    if result.stdout.strip() != expected:
        raise RuntimeError(
            f"Portable executable version mismatch: expected {expected!r}, "
            f"got {result.stdout.strip()!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    options = parser.parse_args()
    try:
        check_portable_startup(options.package_root.resolve(strict=True))
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        return 1
    print("Portable executable startup passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
