$ErrorActionPreference='Stop'
$taskCompiler=Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if(-not(Test-Path -LiteralPath $taskCompiler)){throw 'Windows .NET Framework C# compiler is required.'}
& $taskCompiler /nologo /target:winexe /reference:System.Windows.Forms.dll ('/out:'+(Join-Path $PSScriptRoot 'cemivoAI.exe')) (Join-Path $PSScriptRoot 'CemivoLauncher.cs')
if($LASTEXITCODE -ne 0){throw 'Launcher build failed.'}
