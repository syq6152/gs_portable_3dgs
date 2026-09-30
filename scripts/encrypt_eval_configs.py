#!/usr/bin/env python3
"""Encrypt six training presets and the native Swaptexture settings for delivery."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from encrypt_config_json import DEFAULT_PASSWORD, encrypt_bytes


# Edit this value to choose which strategy's six JSON presets are encrypted.
# Supported values: "adc", "mcmc", "igs+".
ENCRYPT_STRATEGY = "igs+"

CONFIG_PREFIX_BY_STRATEGY = {
    "adc": "adc_optimization_params_pack",
    "mcmc": "mcmc_optimization_params_pack",
    "igs+": "improvedGSplus_optimization_params_pack",
}

PRESET_VARIANTS = (
    "inc_fast",
    "inc_medium",
    "inc_quality",
    "scan_fast",
    "scan_medium",
    "scan_quality",
)


def config_matrix_for_strategy(strategy: str) -> tuple[tuple[str, str, str], ...]:
    try:
        config_prefix = CONFIG_PREFIX_BY_STRATEGY[strategy]
    except KeyError as exc:
        supported = ", ".join(f'"{name}"' for name in CONFIG_PREFIX_BY_STRATEGY)
        raise ValueError(
            f"Unsupported ENCRYPT_STRATEGY {strategy!r}; expected one of {supported}"
        ) from exc

    return tuple(
        (
            variant,
            f"{config_prefix}_{variant}.json",
            f"GS_params_{variant}.bin",
        )
        for variant in PRESET_VARIANTS
    )


def default_eval_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "swaptexture_configs" / "eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-encrypt eval JSON presets into the GS_params_*.bin files "
            "expected by the packaged release."
        )
    )
    parser.add_argument(
        "--strategy", choices=tuple(CONFIG_PREFIX_BY_STRATEGY), default=ENCRYPT_STRATEGY,
        help="Training strategy to package (default: igs+).",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=default_eval_dir(),
        help="Directory containing eval JSON files. Defaults to resources/swaptexture_configs/eval.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory for all seven encrypted files. Defaults to --eval-dir.",
    )
    parser.add_argument(
        "--swaptexture-config", type=Path,
        default=Path(__file__).resolve().parents[1] / "resources" / "swaptexture_configs" / "swaptexture_params.json",
        help="Native Swaptexture settings JSON, encrypted as Swaptexture_params.bin.",
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_PASSWORD,
        help="Encryption password. Defaults to the LichtFeld release password.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned outputs without writing .bin files.",
    )
    return parser.parse_args()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON value: {value}")


def read_json(input_path: Path) -> bytes:
    plaintext = input_path.read_text(encoding="utf-8-sig")
    value = json.loads(plaintext, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {input_path}")
    return plaintext.encode("utf-8")


def write_encrypted(output_path: Path, encrypted: bytes) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output_path.parent, prefix=".config-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encrypted)
        os.replace(temporary, output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def encrypt_json_file(input_path: Path, output_path: Path, password: str, dry_run: bool) -> None:
    plaintext = read_json(input_path)

    if dry_run:
        return

    write_encrypted(output_path, encrypt_bytes(plaintext, password))


def main() -> int:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    output_dir = (args.output_dir or eval_dir).resolve()

    try:
        config_matrix = config_matrix_for_strategy(args.strategy)
    except ValueError as exc:
        print(exc)
        return 2

    inputs = [(label, eval_dir / json_name, output_dir / bin_name)
              for label, json_name, bin_name in config_matrix]
    inputs.append(("swaptexture", args.swaptexture_config.resolve(), output_dir / "Swaptexture_params.bin"))
    missing_files = [str(source) for _, source, _ in inputs if not source.is_file()]
    if missing_files:
        for path in missing_files:
            print(f"Missing input: {path}")
        return 1

    try:
        # Validate the entire batch before replacing any deployed configuration.
        validated = [(label, source, target, read_json(source)) for label, source, target in inputs]
        payloads = [] if args.dry_run else [encrypt_bytes(data, args.password) for _, _, _, data in validated]
        print(f"Encrypting strategy: {args.strategy}")
        for index, (label, source, target, _) in enumerate(validated):
            if not args.dry_run:
                write_encrypted(target, payloads[index])
            action = "would write" if args.dry_run else "wrote"
            print(f"{label}: {source} -> {target} ({action})")
    except (OSError, ValueError) as exc:
        print(f"Cannot encrypt configuration: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
