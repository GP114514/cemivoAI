$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskRoot=(Resolve-Path -LiteralPath $PSScriptRoot).Path
try {
    $taskTraining=Get-Content -LiteralPath (Join-Path $taskRoot 'neural\checkpoints\status.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskTrainingPid=$taskTraining.pid
    $taskListeners=@(Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue)
    if($taskListeners.Count -gt 1){throw 'Multiple listeners on port 7861; no process was stopped.'}
    if($taskListeners.Count -eq 1){
        $taskLive=Invoke-RestMethod 'http://127.0.0.1:7861/api/live/status' -TimeoutSec 5
        if($taskLive.runtime.busy){throw 'APP is processing a request. Wait for it to finish, then retry.'}
        if($taskLive.planner.awaiting_result){throw 'DSH is still awaiting an execution result. Check that task before restarting.'}
        $taskAppPid=$taskListeners[0].OwningProcess
        $taskProcess=Get-CimInstance Win32_Process -Filter ('ProcessId='+$taskAppPid)
        if($taskAppPid -eq $taskTrainingPid -or -not $taskProcess.CommandLine.Contains($taskRoot+'\server.py')){
            throw 'The listener is not the verified project APP server. No process was stopped.'
        }
        Write-Host ('Restarting APP server PID '+$taskAppPid+'. Training PID '+$taskTrainingPid+' is preserved.')
        Stop-Process -Id $taskAppPid
        for($taskTry=0;$taskTry -lt 20;$taskTry++){
            if(-not(Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue)){break}
            Start-Sleep -Milliseconds 200
        }
    }
    & (Join-Path $taskRoot 'screen.ps1')
    $taskStatus=Invoke-RestMethod 'http://127.0.0.1:7861/api/status' -TimeoutSec 10
    if($taskStatus.dsh_graphics_protocol -ne 1){throw 'The running service did not load the new DSH graphics protocol.'}
    $taskDsh=Invoke-RestMethod 'http://127.0.0.1:7861/api/live/config' -Method Post -ContentType 'application/json' -Body '{"backend":"dsh"}' -TimeoutSec 90
    if(-not $taskDsh.configured -or -not $taskDsh.verified){throw 'DSH connection has not passed verification.'}
    Write-Host ('Ready: '+$taskStatus.version+'; DSH connected. Enter the screen in the APP window.') -ForegroundColor Green
    if($taskTrainingPid){
        Write-Host ('Original training process still alive: '+[bool](Get-Process -Id $taskTrainingPid -ErrorAction SilentlyContinue))
    }
} catch {
    Write-Host ('APP restart did not complete: '+$_.Exception.Message) -ForegroundColor Red
    Write-Host 'Training was not stopped by this script. Check work/screen-server-error.log if startup failed.'
    Read-Host 'Press Enter to close'
    exit 1
}
