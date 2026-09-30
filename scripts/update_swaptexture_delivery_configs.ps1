# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
<#
.SYNOPSIS
Re-encrypt selected SwapTexture JSON files into an existing delivery package.

.DESCRIPTION
This is a JSON-only fast path for an already-built Run-GS package. It stages
the canonical seven encrypted configurations with the production encryptor,
then replaces only the selected packaged BIN files. It does not build C++ or
create a new delivery package.

.EXAMPLE
./scripts/update_swaptexture_delivery_configs.ps1 `
    -ConfigPath resources/swaptexture_configs/swaptexture_params.json

.EXAMPLE
./scripts/update_swaptexture_delivery_configs.ps1 `
    -ConfigPath resources/swaptexture_configs/eval/improvedGSplus_optimization_params_pack_scan_medium.json

.EXAMPLE
./scripts/update_swaptexture_delivery_configs.ps1 -All -Strategy igs+
#>
[CmdletBinding(DefaultParameterSetName = 'Selected', SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [Parameter(Mandatory = $true, ParameterSetName = 'Selected', Position = 0)]
    [ValidateNotNullOrEmpty()]
    [string[]]$ConfigPath,

    [Parameter(Mandatory = $true, ParameterSetName = 'All')]
    [switch]$All,

    [ValidateSet('igs+', 'mcmc', 'adc')]
    [string]$Strategy = 'igs+',

    [string]$PackageDirectory = '',
    [string]$PythonExe = 'C:\Users\shiboke\AppData\Local\anaconda3\python.exe'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$configRoot = Join-Path $repoRoot 'resources\swaptexture_configs'
$evalRoot = Join-Path $configRoot 'eval'
$encryptor = Join-Path $repoRoot 'scripts\encrypt_eval_configs.py'

function Resolve-TaskPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$BaseDirectory
    )
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $BaseDirectory $Path))
}

function Assert-File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label was not found: $Path"
    }
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$NativeArguments,
        [Parameter(Mandatory = $true)][string]$Label
    )
    Write-Host "[$Label]" -ForegroundColor Cyan
    & $Executable @NativeArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Remove-TaskTemporaryDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedParent
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    $resolvedPath = (Resolve-Path -LiteralPath $Path).ProviderPath
    $resolvedParent = (Resolve-Path -LiteralPath $ExpectedParent).ProviderPath
    $item = Get-Item -LiteralPath $resolvedPath
    if (-not [string]::Equals((Split-Path $resolvedPath -Parent), $resolvedParent,
            [StringComparison]::OrdinalIgnoreCase) -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing to remove an unexpected temporary directory: $resolvedPath"
    }
    Remove-Item -LiteralPath $resolvedPath -Recurse -Force
}

$prefixByStrategy = @{
    'igs+' = 'improvedGSplus_optimization_params_pack'
    'mcmc' = 'mcmc_optimization_params_pack'
    'adc' = 'adc_optimization_params_pack'
}
$variants = @(
    'inc_fast',
    'inc_medium',
    'inc_quality',
    'scan_fast',
    'scan_medium',
    'scan_quality'
)

$nativeConfig = [IO.Path]::GetFullPath((Join-Path $configRoot 'swaptexture_params.json'))
$canonical = [ordered]@{}
$canonical[$nativeConfig] = 'Swaptexture_params.bin'
$prefix = $prefixByStrategy[$Strategy]
foreach ($variant in $variants) {
    $source = [IO.Path]::GetFullPath((Join-Path $evalRoot "${prefix}_${variant}.json"))
    $canonical[$source] = "GS_params_${variant}.bin"
}

if ([string]::IsNullOrWhiteSpace($PackageDirectory)) {
    $PackageDirectory = Join-Path $repoRoot 'SwapTexture'
} else {
    $PackageDirectory = Resolve-TaskPath $PackageDirectory $repoRoot
}
$PackageDirectory = [IO.Path]::GetFullPath($PackageDirectory)
$binDirectory = Join-Path $PackageDirectory 'bin'

if (-not (Test-Path -LiteralPath $PackageDirectory -PathType Container)) {
    throw "Package directory was not found: $PackageDirectory"
}
if (-not (Test-Path -LiteralPath $binDirectory -PathType Container)) {
    throw "Package bin directory was not found: $binDirectory"
}
$packagedExecutable = Join-Path $PackageDirectory 'SwapTexture.exe'
if (-not (Test-Path -LiteralPath $packagedExecutable -PathType Leaf)) {
    # Explicitly selected older packages remain eligible for config-only updates.
    $packagedExecutable = Join-Path $PackageDirectory 'Run-GS.exe'
}
if (-not (Test-Path -LiteralPath $packagedExecutable -PathType Leaf)) {
    $packagedExecutable = Join-Path $binDirectory 'Run-GS.exe'
}
Assert-File $packagedExecutable 'Packaged SwapTexture executable'
Assert-File $PythonExe 'Python executable'
Assert-File $encryptor 'Configuration encryptor'

$selected = [Collections.Generic.List[object]]::new()
$seenTargets = @{}
if ($All) {
    foreach ($entry in $canonical.GetEnumerator()) {
        $selected.Add([PSCustomObject]@{ Source = $entry.Key; Name = $entry.Value })
        $seenTargets[$entry.Value] = $true
    }
} else {
    foreach ($requestedPath in $ConfigPath) {
        $resolvedSource = Resolve-TaskPath $requestedPath $repoRoot
        if (-not $canonical.Contains($resolvedSource)) {
            throw "ConfigPath must be one of the seven canonical JSON inputs for strategy '$Strategy': $resolvedSource"
        }
        $targetName = $canonical[$resolvedSource]
        if (-not $seenTargets.ContainsKey($targetName)) {
            $selected.Add([PSCustomObject]@{ Source = $resolvedSource; Name = $targetName })
            $seenTargets[$targetName] = $true
        }
    }
}

foreach ($entry in $canonical.GetEnumerator()) {
    Assert-File $entry.Key 'Canonical JSON input'
}
foreach ($entry in $selected) {
    Assert-File (Join-Path $binDirectory $entry.Name) 'Packaged encrypted configuration'
}

$targetSummary = ($selected | ForEach-Object { $_.Name }) -join ', '
if (-not $PSCmdlet.ShouldProcess($binDirectory, "Encrypt and replace $targetSummary")) {
    return
}

$temporaryParent = Join-Path ([IO.Path]::GetTempPath()) 'LichtFeldStudio\swaptexture-config-update'
$temporaryDirectory = Join-Path $temporaryParent ([Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryDirectory -Force | Out-Null

$transactionId = [Guid]::NewGuid().ToString('N')
$prepared = [Collections.Generic.List[object]]::new()
$completed = [Collections.Generic.List[object]]::new()
$deploymentSucceeded = $false
try {
    Invoke-Native $PythonExe @(
        $encryptor,
        '--strategy', $Strategy,
        '--eval-dir', $evalRoot,
        '--output-dir', $temporaryDirectory,
        '--swaptexture-config', $nativeConfig
    ) 'Encrypt canonical SwapTexture configuration batch'

    foreach ($entry in $selected) {
        $staged = Join-Path $temporaryDirectory $entry.Name
        Assert-File $staged 'Staged encrypted configuration'
        $target = Join-Path $binDirectory $entry.Name
        $candidate = Join-Path $binDirectory ".lfs-config-$transactionId-$($entry.Name).new"
        $backup = Join-Path $binDirectory ".lfs-config-$transactionId-$($entry.Name).bak"
        [IO.File]::Copy($staged, $candidate, $false)
        $prepared.Add([PSCustomObject]@{
            Source = $entry.Source
            Name = $entry.Name
            Target = $target
            Candidate = $candidate
            Backup = $backup
        })
    }

    try {
        foreach ($entry in $prepared) {
            [IO.File]::Replace($entry.Candidate, $entry.Target, $entry.Backup, $true)
            $completed.Add($entry)
            Write-Host "$($entry.Source) -> $($entry.Target)"
        }
    } catch {
        $replacementError = $_.Exception.Message
        $rollbackErrors = [Collections.Generic.List[string]]::new()
        for ($index = $completed.Count - 1; $index -ge 0; --$index) {
            $entry = $completed[$index]
            try {
                [IO.File]::Replace($entry.Backup, $entry.Target, $entry.Candidate, $true)
            } catch {
                $rollbackErrors.Add("$($entry.Name): $($_.Exception.Message)")
            }
        }
        if ($rollbackErrors.Count -gt 0) {
            throw "Configuration replacement failed: $replacementError; rollback incomplete: $($rollbackErrors -join '; ')"
        }
        throw "Configuration replacement failed; completed replacements were rolled back: $replacementError"
    }

    $deploymentSucceeded = $true
    Write-Host "Updated $($selected.Count) encrypted configuration file(s) in $binDirectory" -ForegroundColor Green
    [Console]::Error.WriteLine(
        'WARNING: Packaged configuration hashes changed; previous external package audit manifests and reports are stale. Run build_swaptexture_delivery.ps1 before formal release.'
    )
} finally {
    foreach ($entry in $prepared) {
        if (Test-Path -LiteralPath $entry.Candidate) {
            Remove-Item -LiteralPath $entry.Candidate -Force
        }
        if (Test-Path -LiteralPath $entry.Backup) {
            if ($deploymentSucceeded) {
                Remove-Item -LiteralPath $entry.Backup -Force
            } else {
                [Console]::Error.WriteLine("Original configuration backup retained for manual recovery: $($entry.Backup)")
            }
        }
    }
    Remove-TaskTemporaryDirectory $temporaryDirectory $temporaryParent
}
