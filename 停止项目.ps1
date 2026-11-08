#Requires -Version 7.0
$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot '项目代码\deployment-server.pid'
$expectedPython = Join-Path $PSScriptRoot '项目代码\.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host '未找到本次后台服务的进程记录。前台服务请在对应终端按 Ctrl+C。'
    return
}
$launcherId = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$launcher = Get-CimInstance Win32_Process -Filter "ProcessId = $launcherId"
if (-not $launcher) {
    Write-Host '记录的后台服务已经停止。前台服务请在对应终端按 Ctrl+C。'
    return
}
if ($launcher.ExecutablePath -ne $expectedPython -or $launcher.CommandLine -notmatch 'serve\.py') {
    throw '进程记录已失效，未停止任何进程。'
}
# Windows venv 的 python.exe 是启动器，实际应用在其子进程中。
$children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $launcherId"
foreach ($child in $children) {
    if ($child.Name -eq 'python.exe' -and $child.CommandLine -match 'serve\.py') {
        Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
    }
}
Stop-Process -Id $launcherId -ErrorAction SilentlyContinue
Write-Host '本项目的后台应用已停止。MySQL、Neo4j 和 Milvus 保持运行。'
