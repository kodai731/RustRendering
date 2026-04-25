[CmdletBinding()]
param(
    [string]$WheelsDir = "blender_addon/wheels"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot/.."
$AbsWheels = Join-Path $RepoRoot $WheelsDir
$ManifestPath = Join-Path $AbsWheels "HASHES.txt"

if (-not (Test-Path $ManifestPath)) {
    throw "HASHES.txt not found at $ManifestPath. Run scripts/collect_wheels.ps1 first."
}

$Expected = @{}
foreach ($line in Get-Content $ManifestPath) {
    if ($line -match '^\s*$' -or $line.StartsWith('#')) { continue }
    $parts = $line -split '\s+', 2
    if ($parts.Length -ne 2) {
        throw "Malformed HASHES.txt line: $line"
    }
    $Expected[$parts[0].Trim()] = $parts[1].Trim().ToLower()
}

$Mismatches = New-Object System.Collections.Generic.List[string]
$Missing = New-Object System.Collections.Generic.List[string]
$Unexpected = New-Object System.Collections.Generic.List[string]

foreach ($wheel in Get-ChildItem $AbsWheels -Filter "*.whl") {
    if (-not $Expected.ContainsKey($wheel.Name)) {
        $Unexpected.Add($wheel.Name)
        continue
    }
    $actual = (Get-FileHash $wheel.FullName -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $Expected[$wheel.Name]) {
        $Mismatches.Add("$($wheel.Name): expected $($Expected[$wheel.Name]), got $actual")
    }
}

foreach ($name in $Expected.Keys) {
    if (-not (Test-Path (Join-Path $AbsWheels $name))) {
        $Missing.Add($name)
    }
}

if ($Mismatches.Count -gt 0 -or $Missing.Count -gt 0) {
    foreach ($m in $Mismatches) { Write-Error $m }
    foreach ($m in $Missing) { Write-Error "missing wheel: $m" }
    throw "Wheel hash verification failed"
}

if ($Unexpected.Count -gt 0) {
    foreach ($u in $Unexpected) {
        Write-Warning "unexpected wheel (not in HASHES.txt): $u"
    }
}

Write-Host "[verify_wheel_hashes] All wheel hashes match ($($Expected.Count) entries)" -ForegroundColor Green
