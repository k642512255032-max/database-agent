<#
.SYNOPSIS
  Tune the Ollama server on Windows for this agent (more prompt-cache slots, long keep-alive, flash attention,
  q8_0 KV cache) and optionally try the integrated GPU. Restarts the Ollama tray app so the settings take effect.

.DESCRIPTION
  Ollama on Windows reads its settings from USER environment variables. This script sets them, restarts Ollama,
  waits for the API, and prints the "server config" / "inference compute" lines of the server log so you can see
  what actually took effect. Measure with `python bench_agent.py` before and after.

    deploy\windows\ollama_env.ps1              # CPU: 4 cache slots, keep models 30 min, flash attention, q8_0 KV
    deploy\windows\ollama_env.ps1 -Igpu        # + try the integrated GPU through Vulkan (OLLAMA_IGPU_ENABLE=1)
    deploy\windows\ollama_env.ps1 -Rocm        # + try ROCm with HSA_OVERRIDE_GFX_VERSION=11.5.1 (gfx1152 -> gfx1151)
    deploy\windows\ollama_env.ps1 -Reset       # remove every variable this script sets and restart Ollama

  Why 4 slots: each slot keeps its own prompt cache, and the pipeline sends several different prompt families to
  the same model (expert plan, expert assessment, answer, charts, memory). One slot = the 2 000-token briefing +
  schema prefix is re-evaluated on almost every call (~50 s on CPU); four slots = it is usually cached (<1 s).
#>
param(
    [switch]$Igpu,
    [switch]$Rocm,
    [int]$Parallel = 4,
    [string]$KeepAlive = "30m",
    [switch]$Reset
)

$ErrorActionPreference = "Stop"
$vars = [ordered]@{
    OLLAMA_NUM_PARALLEL      = "$Parallel"
    OLLAMA_KEEP_ALIVE        = $KeepAlive
    OLLAMA_FLASH_ATTENTION   = "1"
    OLLAMA_KV_CACHE_TYPE     = "q8_0"          # needs flash attention; halves the KV cache memory
    OLLAMA_IGPU_ENABLE       = $(if ($Igpu) { "1" } else { $null })       # Vulkan on the integrated GPU
    HSA_OVERRIDE_GFX_VERSION = $(if ($Rocm) { "11.5.1" } else { $null })  # gfx1152 -> supported gfx1151 kernels
}

foreach ($k in $vars.Keys) {
    $value = if ($Reset) { $null } else { $vars[$k] }
    [Environment]::SetEnvironmentVariable($k, $value, "User")
    if ($null -eq $value) { Write-Host ("  {0,-26} (removed)" -f $k) } else { Write-Host ("  {0,-26} = {1}" -f $k, $value) }
}

Write-Host "Restarting Ollama..."
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2
$app = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama app.exe"
if (-not (Test-Path $app)) { throw "Ollama not found at $app - install it from https://ollama.com/download" }
Start-Process $app

$up = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 1
    try {
        $v = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2
        Write-Host "Ollama $($v.version) is up."
        $up = $true
        break
    } catch { }
}
if (-not $up) { throw "Ollama did not answer on http://127.0.0.1:11434 within 60 s - check $env:LOCALAPPDATA\Ollama\server.log" }

Start-Sleep -Seconds 8   # GPU discovery is logged a few seconds after the API answers
$log = Join-Path $env:LOCALAPPDATA "Ollama\server.log"
if (Test-Path $log) {
    Write-Host "`nserver config (only the variables this script touches):"
    $cfg = (Select-String -Path $log -Pattern "server config" | Select-Object -Last 1).Line
    foreach ($k in $vars.Keys) {
        if ($cfg -match "$k`:([^ \]]*)") { Write-Host ("  {0,-26} {1}" -f $k, $Matches[1]) }
    }
    Write-Host "`ninference compute:"
    Select-String -Path $log -Pattern "inference compute|dropping integrated GPU|dropping ROCm" |
        Select-Object -Last 3 | ForEach-Object { "  " + $_.Line.Substring([Math]::Min(60, $_.Line.Length)) }
}
Write-Host "`nNow measure: python bench_agent.py --json .claude\PRPs\bench\after.json"
