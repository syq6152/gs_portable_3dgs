# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
<#
.SYNOPSIS
Build, audit, publish, and optionally archive a Windows Run-GS package.

.DESCRIPTION
Uses Visual Studio 2022 Community, the repository vcpkg toolchain, and conda
base Python. The selected strategy controls which six eval JSON files are
encrypted. Build caches are retained per strategy under the repository's
build-swaptexture-work directory and reused by subsequent runs. Packages are
installed into fresh per-run staging directories, audited, published by
default, and audited again. Successful runs clean only distribution scratch files.
#>
[CmdletBinding()]
param(
    [ValidateSet('igs+', 'mcmc', 'adc')]
    [string]$Strategy = 'igs+',
    [ValidateRange(1, 64)]
    [int]$Parallel = 8,
    [ValidateRange(1, 64)]
    [int]$VcpkgConcurrency = 8,

    [string]$PublishDirectory = '',
    [string]$ArtifactDirectory = '',
    [switch]$BuildOnly,
    [switch]$SkipTests,
    [switch]$Publish = $true,
    [switch]$Archive,
    [string]$VsDevCmd = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat',
    [string]$CMakeExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe',
    [string]$NinjaExe = 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe',
    [string]$VcpkgExe = '',
    [string]$PythonExe = 'C:\Users\shiboke\AppData\Local\anaconda3\python.exe',
    [string]$VcpkgRoot = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$strategySlug = $Strategy.Replace('+', 'plus')

function Resolve-TaskPath {
    param([string]$Path, [string]$Default)
    $selected = if ([string]::IsNullOrWhiteSpace($Path)) { $Default } else { $Path }
    if ([IO.Path]::IsPathRooted($selected)) {
        return [IO.Path]::GetFullPath($selected)
    }
    return [IO.Path]::GetFullPath((Join-Path $repoRoot $selected))
}

function Assert-File {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label was not found: $Path"
    }
}

function Invoke-Native {
    param([string]$Executable, [string[]]$NativeArguments, [string]$Label)
    Write-Host ''
    Write-Host "[$Label]" -ForegroundColor Cyan
    Write-Host "$Executable $($NativeArguments -join ' ')" -ForegroundColor DarkGray
    & $Executable @NativeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Import-VsDeveloperEnvironment {
    Assert-File $VsDevCmd 'VS 2022 developer command file'
    $command = 'call "' + $VsDevCmd + '" -arch=x64 -host_arch=x64 >nul && set'
    $environmentLines = & $env:ComSpec /d /s /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "VsDevCmd.bat failed with exit code $LASTEXITCODE"
    }
    foreach ($line in $environmentLines) {
        $separator = $line.IndexOf('=')
        if ($separator -le 0) {
            continue
        }
        $name = $line.Substring(0, $separator)
        $value = $line.Substring($separator + 1)
        Set-Item -LiteralPath "Env:$name" -Value $value
    }
}

function Invoke-PackageAudit {
    param([string]$Prefix, [string]$ReportName)
    $report = Join-Path $AuditDirectory $ReportName
    if (Test-Path -LiteralPath $report) {
        throw "Verification report already exists: $report"
    }
    Invoke-Native $PythonExe @(
        (Join-Path $repoRoot 'scripts\swaptexture_m6_package_verify.py'),
        '--prefix', $Prefix,
        '--audit-dir', $AuditDirectory,
        '--trusted-runtime-manifest', (Join-Path $repoRoot 'third_party\runtime\manifest.json'),
        '--report', $report
    ) "Audit package $Prefix"
    $result = Get-Content -LiteralPath $report -Encoding UTF8 -Raw | ConvertFrom-Json
    if (-not $result.package_valid) {
        throw "Package did not pass verification: $report"
    }
    return $report
}

$defaultPublishName = 'SwapTexture'
$PublishDirectory = Resolve-TaskPath $PublishDirectory $defaultPublishName
$ArtifactDirectory = Resolve-TaskPath $ArtifactDirectory 'build-swaptexture-artifacts'
$runId = [Guid]::NewGuid().ToString('N')
$workRoot = Join-Path $repoRoot 'build-swaptexture-work'
$strategyRoot = Join-Path $workRoot $strategySlug
$BuildDirectory = Join-Path $strategyRoot 'build'
$runRoot = Join-Path $strategyRoot 'runs'
$workDirectory = Join-Path $runRoot "$timestamp-$runId"
$StageDirectory = Join-Path $workDirectory 'stage'
$AuditDirectory = Join-Path $workDirectory 'package-audit'
$backupPath = Join-Path $workDirectory 'previous-publish'
$backupNeedsRecovery = $false
$deliverySucceeded = $false
$buildLock = $null
if ([string]::IsNullOrWhiteSpace($VcpkgRoot)) {
    $VcpkgRoot = [IO.Path]::GetFullPath((Join-Path (Split-Path $repoRoot -Parent) 'vcpkg'))
} else {
    $VcpkgRoot = [IO.Path]::GetFullPath($VcpkgRoot)
}

if ([string]::IsNullOrWhiteSpace($VcpkgExe)) {
    $VcpkgExe = Join-Path $VcpkgRoot 'vcpkg.exe'
} else {
    $VcpkgExe = [IO.Path]::GetFullPath($VcpkgExe)
}

Assert-File $CMakeExe 'CMake executable'
Assert-File $NinjaExe 'Ninja executable'
Assert-File $VcpkgExe 'vcpkg executable'
Assert-File $PythonExe 'Conda base Python'
$toolchain = Join-Path $VcpkgRoot 'scripts\buildsystems\vcpkg.cmake'
Assert-File $toolchain 'vcpkg toolchain'
Assert-File (Join-Path $repoRoot 'scripts\swaptexture_m6_package_verify.py') 'Package verifier'
if (-not $Publish -and -not $BuildOnly) {
    throw 'Use -BuildOnly for a build without publishing; publishing is enabled by default.'
}
# Publishing must not move or replace any persistent cache or its ancestors.
$publishRoot = $PublishDirectory.TrimEnd('\', '/')
$cacheRoot = $workRoot.TrimEnd('\', '/')
if ([string]::Equals($publishRoot, $cacheRoot, [StringComparison]::OrdinalIgnoreCase) -or
    $publishRoot.StartsWith($cacheRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
    $cacheRoot.StartsWith($publishRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'PublishDirectory must not overlap the persistent build workspace'
}
New-Item -ItemType Directory -Path $strategyRoot -Force | Out-Null
try {
    $buildLock = [IO.File]::Open((Join-Path $strategyRoot 'build.lock'),
        [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
} catch {
    throw "Cannot lock build workspace for strategy '$Strategy'; another build may be running. $($_.Exception.Message)"
}

try {
New-Item -ItemType Directory -Path $workDirectory -Force | Out-Null
if (Test-Path -LiteralPath (Join-Path $BuildDirectory 'CMakeCache.txt')) {
    Write-Host "Reusing build cache: $BuildDirectory" -ForegroundColor Cyan
} else {
    Write-Host "Initializing build cache: $BuildDirectory" -ForegroundColor Cyan
}
Import-VsDeveloperEnvironment
$env:VCPKG_ROOT = $VcpkgRoot

$dependencyPrefix = Join-Path $BuildDirectory 'vcpkg_installed'
    $previousConcurrency = $env:VCPKG_MAX_CONCURRENCY
    try {
        $env:VCPKG_MAX_CONCURRENCY = $VcpkgConcurrency.ToString()
        Invoke-Native $VcpkgExe @(
        'install',
        '--triplet', 'x64-windows',
        '--x-manifest-root', $repoRoot,
        '--x-install-root', $dependencyPrefix,
        '--clean-buildtrees-after-build',
        '--clean-packages-after-build'
        ) 'Reconcile manifest dependencies with retained installation'
    } finally {
        $env:VCPKG_MAX_CONCURRENCY = $previousConcurrency
    }
$configureArguments = @(
    '-S', $repoRoot,
    '-B', $BuildDirectory,
    '-G', 'Ninja',
    "-DCMAKE_MAKE_PROGRAM:FILEPATH=$NinjaExe",
    '-DCMAKE_BUILD_TYPE=Release',
    '-DBUILD_PORTABLE=ON',
    '-DBUILD_CUDA_PTX_ONLY=ON',
    '-DBUILD_TESTS=OFF',
    '-DLFS_ENABLE_CLI_HELP=OFF',
    '-DVCPKG_MANIFEST_INSTALL=OFF',
    "-DVCPKG_INSTALLED_DIR:PATH=$dependencyPrefix",
    '-DVCPKG_TARGET_TRIPLET=x64-windows',
    "-DCMAKE_TOOLCHAIN_FILE=$toolchain",
    "-DPython3_EXECUTABLE:FILEPATH=$PythonExe",
    "-DLFS_SWAPTEXTURE_TRAINING_STRATEGY=$Strategy"
)
Invoke-Native $CMakeExe $configureArguments 'Configure Release build'
$buildArguments = @(
    '--build', $BuildDirectory,
    '--config', 'Release',
    '--target', 'Run-GS',
    '--parallel', $Parallel.ToString()
)
Invoke-Native $CMakeExe $buildArguments 'Build Run-GS and seven encrypted configs'

$executableCandidates = @(
    (Join-Path $BuildDirectory 'SwapTexture.exe'),
    (Join-Path $BuildDirectory 'Release\SwapTexture.exe'),
    (Join-Path $BuildDirectory 'bin\SwapTexture.exe')
)
$builtExecutable = $executableCandidates | Where-Object {
    Test-Path -LiteralPath $_ -PathType Leaf
} | Select-Object -First 1
if (-not $builtExecutable) {
    throw "Built SwapTexture executable was not found under $BuildDirectory"
}

# This source workspace intentionally omits the developer test suite.
if (-not $SkipTests -and -not (Test-Path -LiteralPath (Join-Path $repoRoot 'tests\python') -PathType Container)) {
    Write-Host 'Developer tests are not included in this source workspace; package audits and startup checks remain enabled.'
    $SkipTests = $true
}

if (-not $SkipTests) {
    $oldCMake = [Environment]::GetEnvironmentVariable('LFS_M6_CMAKE', 'Process')
    $oldExecutable = [Environment]::GetEnvironmentVariable('LFS_SWAPTEXTURE_EXECUTABLE', 'Process')
    try {
        $env:LFS_M6_CMAKE = $CMakeExe
        $env:LFS_SWAPTEXTURE_EXECUTABLE = $builtExecutable
        Invoke-Native $PythonExe @(
            '-m', 'unittest',
            'tests.python.test_encrypt_eval_configs',
            'tests.python.test_swaptexture_m6_package',
            'tests.python.test_swaptexture_delivery_incremental',
            'tests.python.test_swaptexture_native_config',
            '-v'
        ) 'Run focused encryption/package/config tests'
    } finally {
        [Environment]::SetEnvironmentVariable('LFS_M6_CMAKE', $oldCMake, 'Process')
        [Environment]::SetEnvironmentVariable('LFS_SWAPTEXTURE_EXECUTABLE', $oldExecutable, 'Process')
    }
}

if ($BuildOnly) {
    Write-Host ''
    Write-Host "Build completed: $builtExecutable" -ForegroundColor Green
    Write-Host "Encrypted strategy: $Strategy"
    exit 0
}

$stageParent = Split-Path $StageDirectory -Parent
New-Item -ItemType Directory -Path $stageParent -Force | Out-Null
$installArguments = @(
    '--install', $BuildDirectory,
    '--config', 'Release',
    '--prefix', $StageDirectory
)
Invoke-Native $CMakeExe $installArguments 'Install fresh runtime-only package'

New-Item -ItemType Directory -Path $AuditDirectory -Force | Out-Null
# CMake regenerates these manifests on install. Snapshot them with this run's
# verification reports so a later install cannot overwrite failed-run evidence.
foreach ($name in @('package-manifest.json', 'runtime-dependencies.json')) {
    Copy-Item -LiteralPath (Join-Path $BuildDirectory "package-audit\$name") -Destination $AuditDirectory
}
$verificationReport = Invoke-PackageAudit $StageDirectory "verification-$strategySlug-$timestamp.json"
Invoke-Native $PythonExe @(
    (Join-Path $repoRoot 'scripts\check_portable_startup.py'),
    '--package-root', $StageDirectory
) 'Verify staged Run-GS starts without build-tree DLLs'

$packagePath = $StageDirectory
# Do not treat the old package as a failed candidate if its backup move fails.
if (Test-Path -LiteralPath $PublishDirectory) {
    Move-Item -LiteralPath $PublishDirectory -Destination $backupPath
    $backupNeedsRecovery = $true
}
try {
    $publishParent = Split-Path $PublishDirectory -Parent
    New-Item -ItemType Directory -Path $publishParent -Force | Out-Null
    Move-Item -LiteralPath $StageDirectory -Destination $PublishDirectory
    $packagePath = $PublishDirectory
    $null = Invoke-PackageAudit $packagePath "published-$strategySlug-$timestamp.json"
    Invoke-Native $PythonExe @(
        (Join-Path $repoRoot 'scripts\check_portable_startup.py'),
        '--package-root', $packagePath
    ) 'Verify published Run-GS starts without build-tree DLLs'
    $backupNeedsRecovery = $false
} catch {
    $failedPath = Join-Path $workDirectory 'failed-publish'
    if (Test-Path -LiteralPath $PublishDirectory) {
        Move-Item -LiteralPath $PublishDirectory -Destination $failedPath
    }
    if ($backupNeedsRecovery -and (Test-Path -LiteralPath $backupPath)) {
        Move-Item -LiteralPath $backupPath -Destination $PublishDirectory
        $backupNeedsRecovery = $false
    }
    throw
}

$archivePath = $null
if ($Archive) {
    $tarCommand = Get-Command tar.exe -ErrorAction SilentlyContinue
    if (-not $tarCommand) {
        throw 'tar.exe is required for -Archive'
    }
    New-Item -ItemType Directory -Path $ArtifactDirectory -Force | Out-Null
    $archivePath = Join-Path $ArtifactDirectory "Run-GS-$strategySlug-$timestamp.zip"
    if (Test-Path -LiteralPath $archivePath) {
        throw "Archive already exists: $archivePath"
    }
    $packageParent = Split-Path $packagePath -Parent
    $packageLeaf = Split-Path $packagePath -Leaf
    $archiveArguments = @(
        '-a', '-cf', $archivePath,
        '-C', $packageParent,
        $packageLeaf
    )
    Invoke-Native $tarCommand.Source $archiveArguments 'Create distribution ZIP'
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $archivePath).Hash.ToLowerInvariant()
    $hashLine = "$hash  $([IO.Path]::GetFileName($archivePath))" + [Environment]::NewLine
    [IO.File]::WriteAllText("$archivePath.sha256", $hashLine, [Text.UTF8Encoding]::new($false))
}

Write-Host ''
$deliverySucceeded = $true
Write-Host 'SwapTexture delivery audited successfully.' -ForegroundColor Green
Write-Host "Strategy : $Strategy"
Write-Host "Package  : $packagePath"
Write-Host 'Audit    : passed; cleaning distribution scratch files'
if ($archivePath) { Write-Host "Archive  : $archivePath" }
} finally {
    try {
        Write-Host "Build cache retained: $BuildDirectory"
        if ($BuildOnly) {
            Write-Host "Run workspace retained: $workDirectory"
        } elseif (Test-Path -LiteralPath $workDirectory) {
            if (-not $deliverySucceeded -or $backupNeedsRecovery) {
                Write-Warning "Delivery failed; workspace retained for diagnosis or recovery: $workDirectory"
            } else {
                $resolvedWork = (Resolve-Path -LiteralPath $workDirectory).ProviderPath
                $resolvedRoot = (Resolve-Path -LiteralPath $runRoot).ProviderPath
                if (-not [string]::Equals((Split-Path $resolvedWork -Parent), $resolvedRoot, [StringComparison]::OrdinalIgnoreCase) -or
                    ((Get-Item -LiteralPath $resolvedWork).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                    throw "Refusing to clean a workspace outside the expected run root: $resolvedWork"
                }
                Remove-Item -LiteralPath $workDirectory -Recurse -Force
                Write-Host 'Distribution scratch files removed; build cache retained.' -ForegroundColor Green
            }
        }
    } finally {
        $buildLock.Dispose()
    }
}
