# MOSAIC Windows Launcher
# Usage: .\mosaic.ps1 [command] [args]
# Commands: start, stop, status, benchmark, demo, dashboard, classify, test, help
#
# Run from the mosaic\ directory:
#   cd C:\Users\ASUS\Downloads\mosaic_final\mosaic
#   .\mosaic.ps1 demo
#   .\mosaic.ps1 benchmark burst 8 60
#   .\mosaic.ps1 dashboard

param(
    [string]$Command = "help",
    [string]$Arg1 = "",
    [string]$Arg2 = "",
    [string]$Arg3 = ""
)

$MOSAIC_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$DATA_DIR    = Join-Path $MOSAIC_ROOT "data"
$PID_FILE    = Join-Path $DATA_DIR "mosaic.pid"
$LOG_FILE    = Join-Path $DATA_DIR "scheduler.log"
$SOCK_FILE   = Join-Path $DATA_DIR "mosaic.sock"
$PORT_FILE   = Join-Path $DATA_DIR "mosaic.port"

# Colours
function Write-OK   { param($msg) Write-Host "  [OK]  $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "        $msg" -ForegroundColor Cyan }
function Write-Warn { param($msg) Write-Host "  [!]   $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "  [ERR] $msg" -ForegroundColor Red }

function Show-Banner {
    Write-Host ""
    Write-Host "  MOSAIC -- Multi-Objective Scheduler" -ForegroundColor Cyan
    Write-Host "  Disaster-Response Edge Computing" -ForegroundColor DarkCyan
    Write-Host ""
}

function Test-PythonVersion {
    try {
        $ver = python --version 2>&1
        if ($ver -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]; $minor = [int]$Matches[2]
            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
                Write-Err "Python 3.9+ required (found $ver)"
                Write-Info "Download: https://www.python.org/downloads/"
                exit 1
            }
            Write-OK "$ver"
            return $true
        }
    } catch {
        Write-Err "Python not found. Install from https://www.python.org/downloads/"
        exit 1
    }
    return $false
}

function Initialize-DB {
    Write-Info "Initialising fingerprint database..."
    Set-Location $MOSAIC_ROOT
    python profiler\build_db.py --matrix 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-OK "Database ready" }
    else { Write-Warn "DB init issue -- will retry on start" }
}

function Start-Scheduler {
    param([string]$PowerCap = "85")

    if (Test-Path $PID_FILE) {
        $existingPid = Get-Content $PID_FILE -ErrorAction SilentlyContinue
        if ($existingPid -and (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
            Write-Warn "MOSAIC already running (PID=$existingPid)"
            return
        }
        Remove-Item $PID_FILE -ErrorAction SilentlyContinue
    }

    New-Item -ItemType Directory -Force -Path $DATA_DIR | Out-Null
    Write-Info "Starting MOSAIC scheduler (power_cap=${PowerCap}W)..."

    $proc = Start-Process python `
        -ArgumentList "scheduler\scheduler.py --power-cap $PowerCap" `
        -WorkingDirectory $MOSAIC_ROOT `
        -RedirectStandardOutput $LOG_FILE `
        -RedirectStandardError  "$LOG_FILE.err" `
        -PassThru `
        -WindowStyle Hidden

    $proc.Id | Set-Content $PID_FILE

    # Wait for TCP port file (Windows uses TCP, not Unix socket)
    $waited = 0
    while ($waited -lt 10) {
        if (Test-Path $PORT_FILE) { break }
        Start-Sleep -Milliseconds 400
        $waited += 0.4
    }

    if (Test-Path $PORT_FILE) {
        Write-OK "Scheduler started (PID=$($proc.Id))"
        Write-Info "Logs:      Get-Content $LOG_FILE -Wait"
        Write-Info "Dashboard: .\mosaic.ps1 dashboard"
        Write-Info "Status:    .\mosaic.ps1 status"
    } else {
        Write-Err "Scheduler failed to start. Check: $LOG_FILE"
    }
}

function Stop-Scheduler {
    if (-not (Test-Path $PID_FILE)) { Write-Warn "MOSAIC not running"; return }
    $pid = Get-Content $PID_FILE
    try {
        Stop-Process -Id $pid -Force -ErrorAction Stop
        Write-OK "MOSAIC stopped (PID=$pid)"
    } catch {
        Write-Warn "Process not found -- cleaning up"
    }
    Remove-Item $PID_FILE  -ErrorAction SilentlyContinue
    Remove-Item $PORT_FILE -ErrorAction SilentlyContinue
    Remove-Item $SOCK_FILE -ErrorAction SilentlyContinue
}

function Show-Status {
    Set-Location $MOSAIC_ROOT
    python scheduler\client.py 2>&1
}

function Run-Benchmark {
    param([string]$Pattern="burst", [string]$Rate="6", [string]$Duration="60")
    Write-Info "Benchmark: pattern=$Pattern  rate=$Rate/s  duration=${Duration}s"
    Set-Location $MOSAIC_ROOT
    python run_experiment.py --compare all `
        --pattern $Pattern --rate $Rate --duration $Duration --no-charts
}

function Run-Demo {
    Show-Banner
    Write-Info "Starting full demo..."
    Initialize-DB
    Start-Scheduler 85
    Start-Sleep -Seconds 2

    Write-Info "Running 30s disaster workload..."
    Set-Location $MOSAIC_ROOT
    python workload-gen\workload_gen.py `
        --pattern disaster --rate 4 --duration 30 `
        --scheduler --quiet

    Write-Info "Final status:"
    Show-Status
    Write-OK "Demo complete! Run '.\mosaic.ps1 benchmark' for full comparison."
}

function Start-Dashboard {
    param([string]$Port = "7777")
    Write-Info "Starting dashboard on http://localhost:$Port ..."
    Set-Location $MOSAIC_ROOT
    Start-Process python -ArgumentList "dashboard\server.py --port $Port" `
        -WorkingDirectory $MOSAIC_ROOT -WindowStyle Normal
    Start-Sleep -Seconds 2
    Start-Process "http://localhost:$Port"
    Write-OK "Dashboard opened at http://localhost:$Port"
}

function Run-Classify {
    param([string]$IPC="2.1", [string]$LLC="0.18", [string]$BW="8.5", [string]$BR="0.06")
    Set-Location $MOSAIC_ROOT
    python run_experiment.py --classify --ipc $IPC --llc $LLC --bw $BW --br $BR
}

function Run-Tests {
    Set-Location $MOSAIC_ROOT
    python tests\test_all.py
}

function Show-Help {
    Show-Banner
    Write-Host "  Usage: .\mosaic.ps1 [command] [args]" -ForegroundColor White
    Write-Host ""
    Write-Host "  Commands:" -ForegroundColor White
    Write-Host "    start [power_watts]              Start scheduler daemon (default: 85W)"
    Write-Host "    stop                             Stop scheduler daemon"
    Write-Host "    status                           Show live metrics"
    Write-Host "    benchmark [pattern] [rate] [sec] Compare all 5 schedulers"
    Write-Host "    demo                             Full demo: init + start + workload + status"
    Write-Host "    dashboard [port]                 Open live dashboard (default port: 7777)"
    Write-Host "    classify [ipc] [llc] [bw] [br]  ML-classify a workload fingerprint"
    Write-Host "    test                             Run 63-test suite"
    Write-Host "    help                             Show this message"
    Write-Host ""
    Write-Host "  Quick start:" -ForegroundColor Cyan
    Write-Host "    .\mosaic.ps1 demo" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Examples:" -ForegroundColor White
    Write-Host "    .\mosaic.ps1 start 90"
    Write-Host "    .\mosaic.ps1 benchmark disaster 8 120"
    Write-Host "    .\mosaic.ps1 classify 1.8 0.42 28.0 0.08"
    Write-Host "    .\mosaic.ps1 dashboard 8080"
    Write-Host ""
}

# -- Main dispatch --------------------------------------------------------------
Set-Location $MOSAIC_ROOT

switch ($Command.ToLower()) {
    "start"     { Test-PythonVersion | Out-Null; Initialize-DB; Start-Scheduler $Arg1 }
    "stop"      { Stop-Scheduler }
    "status"    { Show-Status }
    "benchmark" { Run-Benchmark $Arg1 $Arg2 $Arg3 }
    "demo"      { Test-PythonVersion | Out-Null; Run-Demo }
    "dashboard" { Start-Dashboard $Arg1 }
    "classify"  { Run-Classify $Arg1 $Arg2 $Arg3 }
    "test"      { Run-Tests }
    "help"      { Show-Help }
    default     { Write-Err "Unknown command: $Command"; Show-Help }
}
