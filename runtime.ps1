function Get-CemivoRuntime {
    param([Parameter(Mandatory=$true)][string]$Root)
    $taskResolvedRoot=(Resolve-Path -LiteralPath $Root).Path
    $taskRuntimeFile=Join-Path $taskResolvedRoot 'runtime.json'
    $taskSettings=Get-Content -LiteralPath $taskRuntimeFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $taskPython=Join-Path $taskResolvedRoot '.venv\Scripts\python.exe'
    if(-not(Test-Path -LiteralPath $taskPython -PathType Leaf)){throw 'Python environment is missing from this app folder.'}
    $taskSettings.python=$taskPython
    $taskNeural=(Resolve-Path -LiteralPath (Join-Path $taskResolvedRoot 'neural')).Path
    foreach($taskField in @('live_snapshot','toggle_head')) {
        if(-not $taskSettings.$taskField){continue}
        $taskLeaf=Split-Path -Path $taskSettings.$taskField -Leaf
        if(-not $taskLeaf -or $taskLeaf -in @('.','..')){throw 'Invalid model directory in runtime.json.'}
        $taskModel=(Resolve-Path -LiteralPath (Join-Path $taskNeural $taskLeaf)).Path
        if(-not $taskModel.StartsWith($taskNeural+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Models must remain inside this app folder.'}
        if(-not(Test-Path -LiteralPath (Join-Path $taskModel 'inference.pt') -PathType Leaf)){throw 'Selected model weights are missing.'}
        if($taskField -eq 'live_snapshot' -and -not(Test-Path -LiteralPath (Join-Path $taskModel 'snapshot.json') -PathType Leaf)){throw 'Selected model snapshot metadata is missing.'}
        $taskSettings.$taskField=$taskModel
    }
    $taskNext=Join-Path $taskResolvedRoot ('runtime-'+[Guid]::NewGuid().ToString('N')+'.next')
    $taskSettings | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $taskNext -Encoding UTF8
    Move-Item -LiteralPath $taskNext -Destination $taskRuntimeFile -Force
    return $taskSettings
}
