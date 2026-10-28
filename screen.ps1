$ErrorActionPreference = 'Stop'
$taskRoot=$PSScriptRoot
. (Join-Path $taskRoot 'runtime.ps1')
$taskConfig=Get-CemivoRuntime -Root $taskRoot
$taskUrl='http://127.0.0.1:7861/screen'
try { $taskStatus=Invoke-RestMethod 'http://127.0.0.1:7861/api/status' -TimeoutSec 3 } catch {
    if(Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue){throw 'Port 7861 is occupied by an unavailable service.'}
    $env:PYTHONUTF8='1'
    $taskArgs=@('-u',('"'+(Join-Path $taskRoot 'server.py')+'"'),'--port','7861','--snapshot',('"'+$taskConfig.live_snapshot+'"'))
    $taskServer=Start-Process -FilePath $taskConfig.python -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'work\screen-server.log') -RedirectStandardError (Join-Path $taskRoot 'work\screen-server-error.log') -PassThru
    $taskStatus=$null
    for($taskAttempt=0;$taskAttempt -lt 50;$taskAttempt++){
        Start-Sleep -Milliseconds 200
        try{$taskStatus=Invoke-RestMethod 'http://127.0.0.1:7861/api/status' -TimeoutSec 1;break}catch{}
        if($taskServer.HasExited){break}
    }
    if(-not $taskStatus){throw 'cemivoAI did not start; inspect work/screen-server-error.log.'}
}
if($taskStatus.app_id -ne 'cemivoAI.neural'){throw '7861 端口不是 cemivoAI APP。'}
$taskUrl += '?v=' + [Uri]::EscapeDataString($taskStatus.version)
$taskEdge=Join-Path ${env:ProgramFiles(x86)} 'Microsoft\Edge\Application\msedge.exe'
if(-not(Test-Path -LiteralPath $taskEdge)){throw '未找到 Edge，请在浏览器打开 http://127.0.0.1:7861/screen。'}
Start-Process -FilePath $taskEdge -ArgumentList @("--app=$taskUrl",'--window-size=540,580')
