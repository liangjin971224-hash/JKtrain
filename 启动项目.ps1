#Requires -Version 7.0
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:OMP_NUM_THREADS = '8'
$env:MKL_NUM_THREADS = '8'

Push-Location $PSScriptRoot
try {
    try {
        $health = Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 2
    }
    catch {
        $health = $null
    }
    if ($health -and $health.ready_ports.Count -ge 8) {
        Write-Host '项目已运行：http://127.0.0.1:8000/'
        return
    }
    docker compose up -d
    if ($LASTEXITCODE -ne 0) { throw 'Milvus 启动失败，请检查 Docker Desktop。' }
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-WebRequest 'http://127.0.0.1:9091/healthz' -TimeoutSec 2
            $ready = $response.StatusCode -eq 200
        }
        catch {
            $ready = $false
        }
        if ($ready) { break }
        Start-Sleep -Seconds 2
    }
    if (-not $ready) { throw 'Milvus 未就绪，请运行 docker compose logs milvus 排查。' }
    $python = Join-Path $PSScriptRoot '项目代码\.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw '请先创建项目虚拟环境。' }
    Set-Location (Join-Path $PSScriptRoot '项目代码')
    & $python serve.py
    if ($LASTEXITCODE -ne 0) { throw '应用启动失败，请检查上方日志。' }
}
finally {
    Pop-Location
}
