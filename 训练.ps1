param([ValidateRange(0,1000000)][int]$Steps=0, [ValidateRange(0,16)][int]$BatchSize=0, [ValidateRange(0,32)][int]$Accum=0)
$ErrorActionPreference = 'Stop'
$taskRoot=$PSScriptRoot
$pythonExe=Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw '请先安装项目内 GPU 运行环境，参考 README.md。' }
& $pythonExe -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"
if ($LASTEXITCODE -ne 0) { throw 'GPU 运行环境尚未就绪。' }
Set-Location -LiteralPath $taskRoot
$taskState=@{}
$taskControl=@{}
$taskStateFile=Join-Path $taskRoot 'neural\checkpoints\status.json'
$taskControlFile=Join-Path $taskRoot 'work\training-control.json'
if(Test-Path -LiteralPath $taskStateFile){$taskState=Get-Content -LiteralPath $taskStateFile -Raw -Encoding UTF8 | ConvertFrom-Json}
if(Test-Path -LiteralPath $taskControlFile){$taskControl=Get-Content -LiteralPath $taskControlFile -Raw -Encoding UTF8 | ConvertFrom-Json}
if($Steps -eq 0){$Steps=if($taskState.total_steps){[int]$taskState.total_steps}else{10000}}
if($BatchSize -eq 0){$BatchSize=if($taskState.batch_size){[int]$taskState.batch_size}else{4}}
if($Accum -eq 0){$Accum=if($taskState.gradient_accumulation){[int]$taskState.gradient_accumulation}else{2}}
if($Steps -le [int]$taskState.step){throw '目标累计步数必须大于已经保存的训练步数。'}
$taskDataPath=if($taskState.data_path){[string]$taskState.data_path}elseif($taskControl.data_path){[string]$taskControl.data_path}else{Join-Path $taskRoot 'neural\data'}
$taskDataPath=[System.IO.Path]::GetFullPath($taskDataPath)
$taskAllowedData=@('neural\data','neural\data-v2','neural\data-native512') | ForEach-Object {[System.IO.Path]::GetFullPath((Join-Path $taskRoot $_))}
if($taskDataPath -notin $taskAllowedData){throw '训练数据须为项目内已知数据集。'}
$taskManifest=Get-Content -LiteralPath (Join-Path $taskDataPath 'manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskResolution=[int]$taskManifest.size
if($taskResolution -notin @(256,512)){throw '训练数据原生尺寸须为256或512。'}
if($taskResolution -eq 512 -and $taskManifest.render_native_size -ne 512){throw '512训练需要原生绘制的数据。'}
& $pythonExe -c 'from neural.locking import model_access; guard=model_access("neural/checkpoints"); guard.__enter__(); guard.__exit__(None,None,None)'
if($LASTEXITCODE -ne 0){throw '模型已经在使用中，无需重复启动训练。'}
$stopFile=Join-Path $taskRoot 'neural\checkpoints\STOP'
if (Test-Path -LiteralPath $stopFile) { Remove-Item -LiteralPath $stopFile }
$trainArgs=@('-u','-m','neural.train','--data',$taskDataPath,'--steps',"$Steps",'--batch-size',"$BatchSize",'--accum',"$Accum",'--save-every','500','--sample-every','500','--warmup-steps','0','--resolution',"$taskResolution",'--preview-steps','16')
if($null -ne $taskControl.edge_weight){$trainArgs+=@('--edge-weight',"$($taskControl.edge_weight)")}
$latest=Join-Path $taskRoot 'neural\checkpoints\latest.pt'
if (Test-Path -LiteralPath $latest) { $trainArgs += @('--resume',$latest) }
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='1'
& $pythonExe @trainArgs
if ($LASTEXITCODE -ne 0) { throw '训练中断，请查看屏幕错误信息。已有检查点仍保存在 neural\checkpoints。' }
