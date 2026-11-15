# Stops every part of Phantom Claw that start.ps1 launched.
# The arm is left as it is: if it was holding with torque on, it keeps holding.

param([switch]$Quiet)

$count = 0
Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq "python.exe" -and $_.CommandLine -match "backend\.(server|listen|record|brain)") -or
    ($_.Name -eq "electron.exe" -and $_.CommandLine -match "third-hand") -or
    ($_.Name -in "node.exe", "cmd.exe" -and $_.CommandLine -match "third-hand\\overlay|npm start")
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    $count++
}
if (-not $Quiet) { Write-Host "Stopped $count Phantom Claw processes." }
