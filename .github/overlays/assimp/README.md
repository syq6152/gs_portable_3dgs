# Assimp 6.0.4 stream boundary fix

This overlay retains the Assimp 6.0.4 port from the repository's locked
vcpkg baseline `62159a45e18f3a9ac0548628dcaf74fcb60c6ff9`, including Draco
support and the original build fixes. Local port revision 1 adds only the
reader patch.

`IOStreamBuffer::getNextLine` previously discarded a data row when the CR
and LF of the preceding line fell on opposite sides of a cache boundary.
The PLY importer uses 1 MiB blocks and silently replaced the lost vertex
or face with default zero values. The patch consumes only line-ending
characters across blocks and retains a final line without a terminator.
Binary block reads, PLY values, postprocessing flags, and user files are
unchanged.

The repository's `vcpkg-configuration.json` already enables this overlay
directory. Reconfigure and rebuild to obtain the corrected library;
editing the patch alone does not update an existing portable package.

Verify a built Windows DLL using the project Python:

```bat
C:\Users\shiboke\AppData\Local\anaconda3\python.exe scripts\tests\test_assimp_ply_import.py --assimp-dll SwapTexture\assimp-vc143-mt.dll --postprocess-flags 0x80804b
```

`scripts/tests/assimp_stream_regression.cpp` additionally exercises small
cache sizes and line boundaries directly against the installed header.
