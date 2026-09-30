"""Regression tests for the Assimp PLY importer used by SwapTexture.

Run with the project's Python, optionally selecting another built Assimp DLL::

    python scripts/tests/test_assimp_ply_import.py --assimp-dll path/to/assimp.dll

Only the Python standard library is required. Fixtures live in a temporary
subdirectory of --work-dir (or the system temporary directory) and are removed
after the run. Tests exercise the real C importer, without postprocessing by
default. --postprocess-flags additionally verifies processed triangle geometry.
"""

import argparse
import ctypes as C
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest


BLOCK_SIZE = 1024 * 1024
DEFAULT_DLL = Path(__file__).resolve().parents[2] / "SwapTexture/assimp-vc143-mt.dll"
VERTICES = ((1.0, 2.0, 3.0), (2.0, 2.0, 3.0),
            (1.0, 3.0, 3.0), (1.0, 2.0, 4.0))
FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
VERTEX_LINES = (b"1 2 3", b"2 2 3", b"1 3 3", b"1 2 4")
FACE_LINES = (b"3 0 2 1", b"3 0 1 3", b"3 1 2 3", b"3 2 0 3")


class AiVector3(C.Structure):
    _fields_ = [("x", C.c_float), ("y", C.c_float), ("z", C.c_float)]


class AiFace(C.Structure):
    _fields_ = [("count", C.c_uint), ("indices", C.POINTER(C.c_uint))]


class AiMesh(C.Structure):
    # Prefix of aiMesh through mFaces; arrays have AI_MAX_NUMBER_* == 8.
    _fields_ = [
        ("primitive_types", C.c_uint), ("vertex_count", C.c_uint),
        ("face_count", C.c_uint), ("vertices", C.POINTER(AiVector3)),
        ("normals", C.c_void_p), ("tangents", C.c_void_p),
        ("bitangents", C.c_void_p), ("colors", C.c_void_p * 8),
        ("texture_coords", C.c_void_p * 8), ("uv_components", C.c_uint * 8),
        ("faces", C.POINTER(AiFace)),
    ]


class AiScene(C.Structure):
    _fields_ = [("flags", C.c_uint), ("root", C.c_void_p),
                ("mesh_count", C.c_uint),
                ("meshes", C.POINTER(C.POINTER(AiMesh)))]


def header(newline, encoding=b"ascii", padding=0):
    """Add an exact number of bytes using short, valid PLY comment lines."""
    comments = bytearray()
    minimum = len(b"comment ") + len(newline)
    while padding:
        length = min(4096, padding)
        if 0 < padding - length < minimum:
            length -= minimum
        if length < minimum:
            raise ValueError("Padding cannot form a complete PLY comment")
        comments.extend(b"comment " + b"p" * (length - minimum) + newline)
        padding -= length
    return (b"ply" + newline + b"format " + encoding + b" 1.0" + newline
            + comments + newline.join((
                b"element vertex 4", b"property float x", b"property float y",
                b"property float z", b"element face 4",
                b"property list uchar int vertex_indices", b"end_header"))
            + newline)


def ascii_fixture(record=None, record_offset=None, newline=b"\r\n",
                  final_newline=True, long_vertex=False):
    rows = list(VERTEX_LINES + FACE_LINES)
    if long_vertex:
        rows[1] = b"2" + b" " * (2 * BLOCK_SIZE + 17) + b"2 3"
    prefix = header(newline)
    if record is not None:
        unpadded_offset = len(prefix) + sum(len(row) + len(newline)
                                           for row in rows[:record])
        prefix = header(newline, padding=record_offset - unpadded_offset)
    return prefix + newline.join(rows) + (newline if final_newline else b"")


class AssimpPlyImportTests(unittest.TestCase):
    dll_path = DEFAULT_DLL
    work_dir = None
    postprocess_flags = 0

    @classmethod
    def setUpClass(cls):
        dll_path = cls.dll_path.resolve()
        if not dll_path.is_file():
            raise FileNotFoundError("Select a built Assimp DLL with --assimp-dll: "
                                    + str(dll_path))
        if hasattr(os, "add_dll_directory"):
            dll_directory = os.add_dll_directory(str(dll_path.parent))
            cls.addClassCleanup(dll_directory.close)
        cls.dll = C.CDLL(str(dll_path))
        cls.dll.aiImportFile.argtypes = [C.c_char_p, C.c_uint]
        cls.dll.aiImportFile.restype = C.POINTER(AiScene)
        cls.dll.aiGetErrorString.argtypes = []
        cls.dll.aiGetErrorString.restype = C.c_char_p
        cls.dll.aiReleaseImport.argtypes = [C.POINTER(AiScene)]
        cls.dll.aiReleaseImport.restype = None
        cls.scratch = tempfile.TemporaryDirectory(prefix="assimp-ply-", dir=cls.work_dir)
        cls.addClassCleanup(cls.scratch.cleanup)
        cls.fixture_dir = Path(cls.scratch.name)

    def import_mesh(self, path, flags):
        scene = self.dll.aiImportFile(str(path).encode("utf-8"), flags)
        self.assertTrue(scene, self.dll.aiGetErrorString())
        try:
            self.assertEqual(scene.contents.mesh_count, 1)
            mesh = scene.contents.meshes[0].contents
            # Bound reads before dereferencing pointers, also detecting bad ABI.
            self.assertLessEqual(mesh.vertex_count, 12)
            self.assertLessEqual(mesh.face_count, 4)
            vertices = tuple((v.x, v.y, v.z)
                             for v in mesh.vertices[:mesh.vertex_count])
            faces = []
            for face in mesh.faces[:mesh.face_count]:
                self.assertLessEqual(face.count, 4)
                faces.append(tuple(face.indices[:face.count]))
            return vertices, tuple(faces)
        finally:
            self.dll.aiReleaseImport(scene)

    def check_fixture(self, data, vertices=VERTICES):
        path = self.fixture_dir / (self.id().rsplit(".", 1)[-1] + ".ply")
        path.write_bytes(data)
        actual_vertices, actual_faces = self.import_mesh(path, 0)
        self.assertTrue(all(math.isfinite(v) for p in actual_vertices for v in p))
        # Compare the entire mesh, including the final record: no zero/default
        # replacement, missing row, malformed face, or unintended renumbering.
        self.assertEqual(actual_vertices, vertices)
        self.assertEqual(actual_faces, FACES)
        if self.postprocess_flags:
            actual_vertices, actual_faces = self.import_mesh(path, self.postprocess_flags)
            self.assertTrue(all(math.isfinite(v) for p in actual_vertices for v in p))
            self.assertEqual(sorted(actual_vertices), sorted(vertices))
            self.assertTrue(all(len(f) == 3 for f in actual_faces))
            self.assertTrue(all(0 <= i < len(actual_vertices)
                                for f in actual_faces for i in f))
            self.assertEqual(sorted(tuple(actual_vertices[i] for i in f)
                                    for f in actual_faces),
                             sorted(tuple(vertices[i] for i in f) for f in FACES))

    def split_fixture(self, record):
        data = ascii_fixture(record, BLOCK_SIZE + 1)
        self.assertEqual(data[BLOCK_SIZE - 1:BLOCK_SIZE + 1], b"\r\n")
        return data

    def test_crlf_split_before_vertex_preserves_every_record(self):
        self.check_fixture(self.split_fixture(1))

    def test_crlf_split_before_face_preserves_every_record(self):
        self.check_fixture(self.split_fixture(5))

    def test_lf_equivalent_of_vertex_boundary_preserves_every_record(self):
        self.check_fixture(self.split_fixture(1).replace(b"\r\n", b"\n"))

    def test_lf_equivalent_of_face_boundary_preserves_every_record(self):
        self.check_fixture(self.split_fixture(5).replace(b"\r\n", b"\n"))

    def test_vertex_starting_at_block_boundary(self):
        data = ascii_fixture(1, BLOCK_SIZE)
        self.assertEqual(data[BLOCK_SIZE - 2:BLOCK_SIZE], b"\r\n")
        self.check_fixture(data)

    def test_face_starting_at_block_boundary(self):
        self.check_fixture(ascii_fixture(5, BLOCK_SIZE))

    def test_last_face_without_newline(self):
        self.check_fixture(ascii_fixture(final_newline=False))

    def test_split_crlf_with_last_face_without_newline(self):
        self.check_fixture(self.split_fixture(5)[:-2])

    def test_vertex_line_spanning_multiple_blocks(self):
        self.check_fixture(ascii_fixture(long_vertex=True))

    def test_binary_crlf_bytes_across_block_boundary_are_not_text(self):
        # Float32 0x3f800a0d is finite and begins with CRLF in little endian.
        value = struct.unpack("<f", b"\r\n\x80\x3f")[0]
        vertices = ((value, 2.0, 3.0),) + VERTICES[1:]
        encoding = b"binary_little_endian"
        padding = BLOCK_SIZE - 1 - len(header(b"\r\n", encoding))
        data = header(b"\r\n", encoding, padding)
        data += b"".join(struct.pack("<fff", *v) for v in vertices)
        data += b"".join(struct.pack("<Biii", 3, *f) for f in FACES)
        self.assertEqual(data[BLOCK_SIZE - 1:BLOCK_SIZE + 1], b"\r\n")
        self.check_fixture(data, vertices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assimp-dll", type=Path, default=DEFAULT_DLL)
    parser.add_argument("--work-dir", type=Path,
                        help="Existing parent directory for temporary fixtures")
    parser.add_argument("--postprocess-flags", type=lambda s: int(s, 0), default=0,
                        help="Also verify processed geometry (integer or hex bitmask)")
    args, unittest_args = parser.parse_known_args()
    AssimpPlyImportTests.dll_path = args.assimp_dll
    AssimpPlyImportTests.work_dir = args.work_dir
    AssimpPlyImportTests.postprocess_flags = args.postprocess_flags
    print("Assimp DLL:", args.assimp_dll.resolve(), flush=True)
    unittest.main(argv=[sys.argv[0]] + unittest_args, verbosity=2)


if __name__ == "__main__":
    main()
