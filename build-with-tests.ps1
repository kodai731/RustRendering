# ビルドとテストを順次実行するスクリプト

param(
    [switch]$SkipTests,
    [switch]$Release
)

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
