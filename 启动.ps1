param([ValidateSet('train','live')][string]$Page = 'train', [switch]$AppWindow)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$runtimeConfig = Join-Path $taskRoot 'runtime.json'
. (Join-Path $taskRoot 'runtime.ps1')
$taskRuntimeSettings=Get-CemivoRuntime -Root $taskRoot
$taskRuntimeSettings = @{}
$pythonOptions = @(
    (Join-Path $taskRoot '.venv\Scripts\python.exe'),
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
)
if (Test-Path -LiteralPath $runtimeConfig) {
    $taskRuntimeSettings = Get-Content -LiteralPath $runtimeConfig -Raw -Encoding UTF8 | ConvertFrom-Json
    $pythonOptions = @($taskRuntimeSettings.python) + $pythonOptions
}
$pythonExe = $null
foreach ($candidate in $pythonOptions) {
    if ($candidate -and (Test-Path -LiteralPath $candidate)) {
        & $candidate -c "import numpy; from PIL import Image" 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonExe = $candidate; break }
    }
}
if (-not $pythonExe) {
    $systemPython = Get-Command python -ErrorAction SilentlyContinue
    if ($systemPython -and $systemPython.Source -notlike '*WindowsApps*') {
        & $systemPython.Source -c "import numpy; from PIL import Image" 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonExe = $systemPython.Source }
    }
}
if (-not $pythonExe) { throw '未找到带有 NumPy 和 Pillow 的 Python。请参考 README.md 安装依赖。' }
$taskPort = 7861
$taskReady = $false
while ($taskPort -lt 7911) {
    try {
        $status = Invoke-RestMethod -Uri "http://127.0.0.1:$taskPort/api/status" -TimeoutSec 1
        if ($status.app_id -eq 'cemivoAI.neural' -and $status.version -eq '2.11.4-experimental') { $taskReady = $true; break }
        $taskPort++
    } catch {
        $listener = Get-NetTCPConnection -LocalPort $taskPort -State Listen -ErrorAction SilentlyContinue
        if ($listener) { $taskPort++ } else { break }
    }
}
if ($taskPort -ge 7911) { throw '本机端口忙，请关闭占用端口的程序后重试。' }
if (-not $taskReady) {
    $taskLogs = Join-Path $taskRoot 'work'
    New-Item -ItemType Directory -Force -Path $taskLogs | Out-Null
    $env:PYTHONUTF8 = '1'
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    $taskServerArgs = @('-u', ('"' + (Join-Path $taskRoot 'server.py') + '"'), '--port', "$taskPort")
    if ($taskRuntimeSettings.live_snapshot) { $taskServerArgs += @('--snapshot', ('"' + $taskRuntimeSettings.live_snapshot + '"')) }
    $taskProcess = Start-Process -FilePath $pythonExe -ArgumentList $taskServerArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskLogs "server-$taskPort.log") -RedirectStandardError (Join-Path $taskLogs "server-$taskPort-error.log") -PassThru
    $taskProcess.Id | Set-Content -LiteralPath (Join-Path $taskLogs 'server.pid') -Encoding ASCII
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Milliseconds 250
        try {
            $status = Invoke-RestMethod -Uri "http://127.0.0.1:$taskPort/api/status" -TimeoutSec 1
            if ($status.app_id -eq 'cemivoAI.neural') { $taskReady = $true; break }
        } catch {}
        if ($taskProcess.HasExited) { break }
    }
}
if (-not $taskReady) { throw '启动未成功，请查看 work\server-error.log。' }
$taskUrl = "http://127.0.0.1:$taskPort/$Page"
$taskEdge = Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'
if ($AppWindow -and (Test-Path -LiteralPath $taskEdge)) {
    Start-Process -FilePath $taskEdge -ArgumentList @("--app=$taskUrl")
} else {
    Start-Process $taskUrl
}
