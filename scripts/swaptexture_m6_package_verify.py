# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Audit the minimal native Swaptexture runtime package.

Only the product manifest is plaintext JSON. CMake writes hash inventories
outside the package under ``build/package-audit``.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

TRUSTED_RUNTIME = Path(__file__).resolve().parents[1] / "third_party/runtime/manifest.json"
PRODUCT_MANIFEST = Path(__file__).resolve().parents[1] / "resources/swaptexture_configs/manifest.json"
EXECUTABLES = {"swaptexture.exe", "bin/third_party/colmap-cuda-cli/bin/sfm.exe", "bin/third_party/superresolution/preprocess.exe"}
CONFIGS = {"bin/gs_params_scan_fast.bin", "bin/gs_params_scan_medium.bin", "bin/gs_params_scan_quality.bin", "bin/gs_params_inc_fast.bin", "bin/gs_params_inc_medium.bin", "bin/gs_params_inc_quality.bin", "bin/swaptexture_params.bin"}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)

def safe_relative(name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name, f"unsafe manifest path: {name!r}")
    path = PurePosixPath(name)
    require(not path.is_absolute() and all(p not in ("", ".", "..") for p in name.split("/")), f"unsafe manifest path: {name!r}")
    require(all(p == p.rstrip(" .") for p in path.parts), f"Windows path alias: {name!r}")
    return name

def check_policy(name):
    lower = safe_relative(name).lower()
    parts = set(lower.split("/"))
    require(lower == "swaptexture.exe" or not parts & {"swaptexture.exe", "_internal", "swaptexture_python", ".git", "tests", "scripts"}, f"forbidden legacy/test payload: {name}")
    require(not lower.startswith("share/") and not lower.startswith("licenses/"), f"human-readable package material is forbidden: {name}")
    require(not re.search(r"(^|/)(licenses?|notices?|copying|copyright)([./_-]|$)", lower)
            and (not lower.endswith(".json") or lower == "manifest.json"),
            f"metadata/license payload is forbidden: {name}")
    require(not re.search(r"(^|/)(readme|changelog|authors?|contributors?)([./_-]|$)", lower)
            and not re.search(r"[.](md|txt|rst|html?|pdf|xml|ya?ml)$", lower),
            f"human-readable package material is forbidden: {name}")
    if lower.endswith(".exe"):
        require(lower in EXECUTABLES, f"undeclared executable: {name}")
    require(not lower.endswith(".onnx"), f"external foreground model must not be packaged: {name}")

def inventory(root):
    result = {}
    for path in root.rglob("*"):
        require(not path.is_symlink() and not getattr(path, "is_junction", lambda: False)(), f"linked package payload: {path}")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            check_policy(name)
            require(name.lower() not in result, f"case-alias package file: {name}")
            result[name.lower()] = (name, path)
    return result

def check_records(root, records, *, sizes=False):
    require(isinstance(records, list) and records, "empty or invalid file records")
    names = set()
    for record in records:
        name = safe_relative(record["path"])
        require(name.lower() not in names, f"duplicate manifest path: {name}")
        names.add(name.lower())
        path = root / name
        require(path.is_file() and path.resolve().is_relative_to(root.resolve()), f"missing/escaped file: {name}")
        require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None, f"invalid SHA-256: {name}")
        require(digest(path) == record["sha256"], f"hash mismatch: {name}")
        if sizes:
            require(type(record["byte_size"]) is int and path.stat().st_size == record["byte_size"], f"size mismatch: {name}")
    return names

def verify_provider_tree(root, runtime_manifest):
    runtime = read_json(runtime_manifest)
    require(runtime.get("schema_version") == 1, "unsupported provider manifest")
    components = {row["component"]: row for row in runtime["components"]}
    require(set(components) == {"colmap", "super_resolution"} and len(runtime["components"]) == 2, "wrong provider component set")
    for component in components.values():
        directory = root / "bin/third_party" / safe_relative(component["source_subdirectory"])
        expected = check_records(directory, component["files"], sizes=True)
        actual = {p.relative_to(directory).as_posix().lower() for p in directory.rglob("*") if p.is_file()}
        require(actual == expected, f"extra/missing provider files: {directory.name}")
        entry = safe_relative(component["entrypoint"])
        require(entry.lower() in expected and digest(directory / entry) == component["entrypoint_sha256"], "provider entrypoint is not pinned")

def verify_package(prefix, audit_dir, trusted_runtime=TRUSTED_RUNTIME):
    root = Path(prefix).resolve(strict=True)
    audit = Path(audit_dir).resolve(strict=True)
    files = inventory(root)
    names = {name for name, _ in files.values()}
    lower_names = {name.lower() for name in names}
    require(EXECUTABLES <= lower_names, "missing runtime executable")
    require(CONFIGS <= lower_names, "missing encrypted configuration")
    require("manifest.json" in lower_names, "missing product manifest.json")
    require(read_json(root / "manifest.json") == read_json(PRODUCT_MANIFEST), "product manifest differs from canonical source")
    verify_provider_tree(root, trusted_runtime)
    package_manifest = audit / "package-manifest.json"
    require(package_manifest.is_file(), "missing external package audit manifest")
    package_audit = read_json(package_manifest)
    require(package_audit.get("schema_version") == 1, "unsupported package audit manifest")
    recorded = check_records(root, package_audit["files"], sizes=True)
    require(recorded == lower_names, "package inventory differs from audit manifest")
    dependency_manifest = audit / "runtime-dependencies.json"
    require(dependency_manifest.is_file(), "missing external dependency audit manifest")
    dependencies = read_json(dependency_manifest)
    require(dependencies.get("schema_version") == 1, "unsupported dependency audit manifest")
    dep_names = check_records(root, dependencies["files"], sizes=False)
    return {"schema_name": "lfs.swaptexture.package-verification", "schema_version": 2, "package_root": str(root), "audit_dir": str(audit), "package_valid": True, "release_ready": False, "manifest_sha256": digest(package_manifest), "executable_sha256": digest(root / "SwapTexture.exe"), "file_count": len(files), "native_dependency_count": len(dep_names), "limitations": ["Offline inventory check, not a clean-machine or E2E test.", "Unsigned inventory does not certify supply-chain authenticity.", "Release compatibility and external acceptance remain separate."]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trusted-runtime-manifest", type=Path, default=TRUSTED_RUNTIME)
    options = parser.parse_args()
    require(not options.report.resolve().is_relative_to(options.prefix.resolve()), "report must be outside package")
    require(not options.report.exists(), "report already exists; refusing overwrite")
    try:
        result = verify_package(options.prefix, options.audit_dir, options.trusted_runtime_manifest)
    except (ValueError, OSError, KeyError, TypeError) as error:
        result = {"schema_name": "lfs.swaptexture.package-verification", "schema_version": 2, "package_root": str(options.prefix), "package_valid": False, "release_ready": False, "error": str(error)}
    with options.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result["package_valid"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
