# Starts Phantom Claw as one application: the server, the listener that waits for
# "Phantom Claw, start", the record process and the avatars. Everything runs in the background.
# The avatars stay off screen until the wake phrase or Ctrl+Shift.
#
#   .\start.ps1             pretend arm (safe anywhere)
#   .\start.ps1 -RealArm    drives the real SO-101 on COM3
#   .\start.ps1 -Show       avatars on screen straight away, without the wake phrase
#
# Stop everything with .\stop.ps1

param([switch]$RealArm, [switch]$Show)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = "C:\Users\sudar\.conda\envs\lerobot\python.exe"
$logs = Join-Path $root "backend\cache\logs"
New-Item -ItemType Directory -Force $logs | Out-Null

# one copy only
& (Join-Path $root "stop.ps1") -Quiet

if ($RealArm) { $env:PHANTOM_ARM = "real" } else { Remove-Item Env:PHANTOM_ARM -ErrorAction SilentlyContinue }

function Start-Part($name, $file, $arguments, $dir) {
    Start-Process -FilePath $file -ArgumentList $arguments -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs "$name.log") -RedirectStandardError (Join-Path $logs "$name.err.log")
}

Start-Part "server" $py "-m backend.server" $root

# wait until the server answers before starting the parts that talk to it
$up = $false
foreach ($i in 1..40) {
    try { Invoke-RestMethod "http://127.0.0.1:8765/api/status" -TimeoutSec 2 | Out-Null; $up = $true; break } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $up) { Write-Host "The server did not come up. See backend\cache\logs\server.err.log"; exit 1 }

Start-Part "listen" $py "-m backend.listen" $root
if (Test-Path (Join-Path $root "backend\record\__main__.py")) { Start-Part "record" $py "-m backend.record" $root }

$overlayArgs = if ($Show) { "/c npm start -- --show" } else { "/c npm start" }
Start-Part "overlay" "cmd.exe" $overlayArgs (Join-Path $root "overlay")

$arm = if ($RealArm) { "the REAL arm" } else { "the pretend arm" }
Write-Host "Phantom Claw is running on $arm. Say 'Phantom Claw, start' or press Ctrl+Shift."

# checks before anything moves: cameras, motors, the app, the model services
Start-Sleep 12
& $py (Join-Path $root "setup\tools\preflight.py")
