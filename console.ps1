$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'runtime.ps1')
$taskRuntime=Get-CemivoRuntime -Root $PSScriptRoot
& $taskRuntime.python -u (Join-Path $PSScriptRoot 'console_monitor.py') --port 7861
