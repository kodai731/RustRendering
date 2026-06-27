# ビルドとテストを順次実行するスクリプト

param(
    [switch]$SkipTests,
    [switch]$Release
)

# Idempotently vendor a patched copy of the openusd crate (workaround for its
# USDC compressed integer-valued float-array decode bug; see patches/README.md).
# The workspace [patch.crates-io] points at vendor/openusd, which is git-ignored
# and regenerated here so `cargo build` never fails on a fresh checkout.
function Ensure-PatchedOpenUsd {
    $old = 'let ints: Vec<i32> = self.read_compressed(count)?;'
    $new = 'let ints: Vec<i32> = self.read_encoded_ints(count)?;'
    $dest = "vendor/openusd"
    $target = "$dest/src/usdc/reader.rs"

    $m = [regex]::Match((Get-Content Cargo.lock -Raw),
        '(?ms)^\[\[package\]\]\r?\nname = "openusd"\r?\nversion = "([^"]+)"')
    if (-not $m.Success) { Write-Host "openusd not in Cargo.lock; run 'cargo fetch'" -ForegroundColor Red; exit 1 }
    $ver = $m.Groups[1].Value

    if ((Test-Path "$dest/Cargo.toml") -and
        (Select-String -Path "$dest/Cargo.toml" -Pattern "^version = `"$ver`"" -Quiet) -and
        (Test-Path $target) -and (Select-String -Path $target -SimpleMatch $new -Quiet)) {
        Write-Host "[vendor_openusd] openusd $ver already patched; skip" -ForegroundColor Gray
        return
    }

    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE ".cargo" }
    $src = Get-ChildItem -Path "$cargoHome/registry/src" -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Get-ChildItem -Path $_.FullName -Directory -Filter "openusd-$ver" -ErrorAction SilentlyContinue } |
        Select-Object -First 1
    if (-not $src) { Write-Host "[vendor_openusd] openusd-$ver not in registry; run 'cargo fetch'" -ForegroundColor Red; exit 1 }

    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Path "vendor" -Force | Out-Null
    Copy-Item -Recurse -Force $src.FullName $dest
    Get-ChildItem $dest -Recurse -File | ForEach-Object { $_.IsReadOnly = $false }

    $content = Get-Content $target -Raw
    if ($content -notmatch [regex]::Escape($new)) {
        if ($content -notmatch [regex]::Escape($old)) {
            Write-Host "[vendor_openusd] target line not found; upstream changed - update the fix" -ForegroundColor Red; exit 1
        }
        Set-Content -Path $target -Value $content.Replace($old, $new) -NoNewline
    }
    Write-Host "[vendor_openusd] patched openusd $ver -> $dest" -ForegroundColor Gray
}

Ensure-PatchedOpenUsd

Write-Host "=== Building project ===" -ForegroundColor Cyan

# ビルドオプションを設定
$buildArgs = @()
if ($Release) {
    $buildArgs += "--release"
    Write-Host "Release mode" -ForegroundColor Yellow
} else {
    Write-Host "Debug mode" -ForegroundColor Yellow
}

# ビルド実行
Write-Host "Running: cargo build $buildArgs" -ForegroundColor Gray
cargo build @buildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n=== Build failed! ===" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=== Build succeeded! ===" -ForegroundColor Green

# テストをスキップする場合は終了
if ($SkipTests) {
    Write-Host "Tests skipped (--SkipTests flag)" -ForegroundColor Yellow
    exit 0
}

Write-Host "`n=== Running tests ===" -ForegroundColor Cyan

# log ディレクトリが存在しない場合は作成
if (-not (Test-Path "log")) {
    New-Item -ItemType Directory -Path "log" | Out-Null
}

# 既存の log_test.txt を削除
if (Test-Path "log/log_test.txt") {
    Remove-Item "log/log_test.txt"
}

Write-Host "Running lib tests (with ML)..." -ForegroundColor Gray
cargo test --lib --no-fail-fast 2>&1 | Tee-Object -FilePath "log/log_test.txt"
$libResult = $LASTEXITCODE

Write-Host "`nRunning integration tests..." -ForegroundColor Gray
cargo test --test ecs_tests --no-default-features --no-fail-fast 2>&1 | Tee-Object -Append -FilePath "log/log_test.txt"
$integrationResult = $LASTEXITCODE

Write-Host "`nRunning thyllore-grpc-client CI gates (addon ZIP smoke + ABI SSOT)..." -ForegroundColor Gray
cargo test -p thyllore-grpc-client --no-default-features --no-fail-fast 2>&1 | Tee-Object -Append -FilePath "log/log_test.txt"
$grpcClientResult = $LASTEXITCODE

$LASTEXITCODE = if ($libResult -ne 0) { $libResult }
                elseif ($integrationResult -ne 0) { $integrationResult }
                else { $grpcClientResult }

# 結果を表示
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== All tests passed! ===" -ForegroundColor Green
} else {
    Write-Host "`n=== Some tests failed! ===" -ForegroundColor Red
}

Write-Host "Test results saved to log/log_test.txt" -ForegroundColor Gray

exit $LASTEXITCODE
