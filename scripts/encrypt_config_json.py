#!/usr/bin/env python3
"""Convert a LichtFeld --config JSON file to an encrypted binary .bin file."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path


DEFAULT_PASSWORD = f"{851479}obc{781532}"


def derive_key(password: str) -> bytes:
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
    except ImportError:
        import hashlib
        return hashlib.sha256(password.encode("utf-8")).digest()

    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(password.encode("utf-8"))
    return digest.finalize()


def encrypt_bytes(plaintext: bytes, password: str) -> bytes:
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError:
        if os.name != "nt":
            raise RuntimeError("Missing dependency: cryptography (install it with: pip install cryptography)")
        with tempfile.TemporaryDirectory(prefix="lfs-encrypt-") as directory:
            root = Path(directory)
            source = root / "plain.bin"
            target = root / "encrypted.bin"
            source.write_bytes(plaintext)
            script = r'''
param([string]$InputPath, [string]$OutputPath, [string]$Password)
$InputPath = $env:LFS_ENCRYPT_INPUT
$OutputPath = $env:LFS_ENCRYPT_OUTPUT
$Password = $env:LFS_ENCRYPT_PASSWORD
$plain = [IO.File]::ReadAllBytes($InputPath)
$sha = [Security.Cryptography.SHA256]::Create()
$key = $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($Password))
$iv = New-Object byte[] 16
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($iv)
$aes = [Security.Cryptography.Aes]::Create()
$aes.Mode = [Security.Cryptography.CipherMode]::CBC
$aes.Padding = [Security.Cryptography.PaddingMode]::PKCS7
$aes.Key = $key
$aes.IV = $iv
$enc = $aes.CreateEncryptor().TransformFinalBlock($plain, 0, $plain.Length)
[IO.File]::WriteAllBytes($OutputPath, $iv + $enc)
'''
            environment = os.environ.copy()
            environment.update({"LFS_ENCRYPT_INPUT": str(source), "LFS_ENCRYPT_OUTPUT": str(target),
                                "LFS_ENCRYPT_PASSWORD": password})
            result = subprocess.run(
                ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, env=environment)
            if result.returncode != 0 or not target.is_file():
                raise RuntimeError("PowerShell AES fallback failed: " + result.stderr.strip())
            return target.read_bytes()

    iv = os.urandom(16)
    key = derive_key(password)

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_plaintext = padder.update(plaintext) + padder.finalize()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()

    return iv + ciphertext


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix:
        return input_path.with_suffix(".bin")
    return input_path.with_name(input_path.name + ".bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encrypt a LichtFeld --config JSON file as binary iv+ciphertext. "
            "The output can be passed back through --config."
        )
    )
    parser.add_argument("json_file", type=Path, help="Input --config JSON file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .bin path. Defaults to the input path with .bin suffix.",
    )
    parser.add_argument(
        "--password",
        default=DEFAULT_PASSWORD,
        help="Encryption password. Defaults to the LichtFeld release password.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.json_file
    output_path = args.output or default_output_path(input_path)

    plaintext = input_path.read_text(encoding="utf-8")
    json.loads(plaintext)

    encrypted = encrypt_bytes(plaintext.encode("utf-8"), args.password)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encrypted)

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
