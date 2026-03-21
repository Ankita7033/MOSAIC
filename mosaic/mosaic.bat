@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

set "MOSAIC_ROOT=%~dp0"
set "CMD=%~1"
if "!CMD!"=="" set "CMD=help"

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERR] Python not found. Download: https://www.python.org/downloads/
    pause & exit /b 1
)

if /i "!CMD!"=="help"      goto :help
if /i "!CMD!"=="demo"      goto :demo
if /i "!CMD!"=="start"     goto :start
if /i "!CMD!"=="stop"      goto :stop
if /i "!CMD!"=="status"    goto :status
if /i "!CMD!"=="benchmark" goto :benchmark
if /i "!CMD!"=="dashboard" goto :dashboard
if /i "!CMD!"=="run"       goto :run
if /i "!CMD!"=="classify"  goto :classify
if /i "!CMD!"=="test"      goto :test
if /i "!CMD!"=="logs"      goto :logs
echo [ERR] Unknown command: !CMD!
goto :help

:help
echo.
echo   MOSAIC - Multi-Objective Scheduler
echo   ------------------------------------
echo   .\mosaic.bat start         Start scheduler daemon
echo   .\mosaic.bat dashboard     Open live dashboard at localhost:7777
echo   .\mosaic.bat run           Send live workload (makes charts move)
echo   .\mosaic.bat benchmark     Compare all 5 schedulers
echo   .\mosaic.bat status        Show live metrics
echo   .\mosaic.bat logs          Show scheduler log
echo   .\mosaic.bat stop          Stop scheduler
echo   .\mosaic.bat classify 1.8 0.42 28 0.08   ML classify
echo   .\mosaic.bat test          Run 63 tests
echo   .\mosaic.bat demo          Full demo
echo.
echo   Live dashboard sequence (3 terminals):
echo     Terminal 1:  .\mosaic.bat start
echo     Terminal 2:  .\mosaic.bat dashboard
echo     Terminal 3:  .\mosaic.bat run
echo     Browser:     http://localhost:7777
echo.
goto :end

:start
cd /d "!MOSAIC_ROOT!"
set "POWER=%~2"
if "!POWER!"=="" set "POWER=85"
python profiler\build_db.py --matrix >nul 2>&1
echo   Starting MOSAIC scheduler (power_cap=!POWER!W)...
start /b python scheduler\scheduler.py --power-cap !POWER! >data\scheduler.log 2>&1
echo   Waiting for scheduler...
timeout /t 6 /nobreak >nul
if exist data\mosaic.port (
    echo   [OK] Scheduler started - port: 
    type data\mosaic.port
) else (
    echo   [!] Check log: .\mosaic.bat logs
)
goto :end

:stop
echo   Stopping MOSAIC...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":47777"') do taskkill /pid %%p /f >nul 2>&1
del /f "!MOSAIC_ROOT!data\mosaic.port" >nul 2>&1
del /f "!MOSAIC_ROOT!data\mosaic.sock" >nul 2>&1
echo   [OK] Stopped
goto :end

:status
cd /d "!MOSAIC_ROOT!"
python scheduler\client.py
goto :end

:dashboard
cd /d "!MOSAIC_ROOT!"
set "PORT=%~2"
if "!PORT!"=="" set "PORT=7777"
echo   Starting dashboard at http://localhost:!PORT!
start "MOSAIC Dashboard" python dashboard\server.py --port !PORT!
timeout /t 3 /nobreak >nul
start "" "http://localhost:!PORT!"
echo   [OK] Dashboard open - browser should open automatically
echo   If not, go to: http://localhost:!PORT!
goto :end

:run
cd /d "!MOSAIC_ROOT!"
set "DUR=%~2"
if "!DUR!"=="" set "DUR=120"
echo   Sending disaster workload to scheduler for !DUR! seconds...
echo   Watch dashboard at http://localhost:7777
python workload-gen\workload_gen.py --pattern disaster --rate 4 --duration !DUR! --scheduler
echo   [OK] Workload complete
goto :end

:benchmark
cd /d "!MOSAIC_ROOT!"
set "PATTERN=%~2"
set "RATE=%~3"
set "DUR=%~4"
if "!PATTERN!"=="" set "PATTERN=burst"
if "!RATE!"==""    set "RATE=6"
if "!DUR!"==""     set "DUR=60"
echo   Benchmark: pattern=!PATTERN! rate=!RATE!/s duration=!DUR!s
python run_experiment.py --compare all --pattern !PATTERN! --rate !RATE! --duration !DUR! --no-charts
goto :end

:classify
cd /d "!MOSAIC_ROOT!"
set "IPC=%~2"
set "LLC=%~3"
set "BW=%~4"
set "BR=%~5"
if "!IPC!"=="" set "IPC=2.1"
if "!LLC!"=="" set "LLC=0.18"
if "!BW!"==""  set "BW=8.5"
if "!BR!"==""  set "BR=0.06"
python run_experiment.py --classify --ipc !IPC! --llc !LLC! --bw !BW! --br !BR!
goto :end

:test
cd /d "!MOSAIC_ROOT!"
echo   Running 63 tests...
python tests\test_all.py
goto :end

:logs
cd /d "!MOSAIC_ROOT!"
echo   === Scheduler Log ===
if exist data\scheduler.log (type data\scheduler.log) else (echo   No log yet)
goto :end

:demo
cd /d "!MOSAIC_ROOT!"
echo.
echo   MOSAIC Demo
echo.
python profiler\build_db.py --matrix >nul 2>&1
echo   [OK] Database ready
start /b python scheduler\scheduler.py --power-cap 85 >data\scheduler.log 2>&1
timeout /t 4 /nobreak >nul
python workload-gen\workload_gen.py --pattern disaster --rate 4 --duration 25 --dry-run --quiet
echo.
echo   [OK] Demo complete
goto :end

:end
endlocal