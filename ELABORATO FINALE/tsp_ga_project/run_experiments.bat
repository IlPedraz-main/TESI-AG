@echo off
echo ========================================
echo TSP GA THESIS EXPERIMENTS
echo ========================================
echo.

:menu
echo Select an option:
echo 1. Install dependencies
echo 2. Download TSPLIB instances
echo 3. Run quick test (3 runs, 100 generations)
echo 4. Run standard experiments (30 runs each)
echo 5. Run sensitivity analyses
echo 6. Generate all figures
echo 7. Run COMPLETE experiment suite (3-5 hours)
echo 8. Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto download
if "%choice%"=="3" goto test
if "%choice%"=="4" goto standard
if "%choice%"=="5" goto sensitivity
if "%choice%"=="6" goto visualize
if "%choice%"=="7" goto all
if "%choice%"=="8" goto end

echo Invalid choice. Please try again.
echo.
goto menu

:install
echo.
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Installation complete!
echo.
pause
goto menu

:download
echo.
echo Downloading TSPLIB instances...
python main.py --download
echo.
pause
goto menu

:test
echo.
echo Running quick test...
python main.py --test
echo.
echo Test complete! Check results/ directory.
echo.
pause
goto menu

:standard
echo.
echo Running standard experiments (this may take 30-60 minutes)...
python main.py --standard
echo.
echo Standard experiments complete!
echo.
pause
goto menu

:sensitivity
echo.
echo Running sensitivity analyses (this may take 2-3 hours)...
python main.py --sensitivity
echo.
echo Sensitivity analyses complete!
echo.
pause
goto menu

:visualize
echo.
echo Generating figures...
python main.py --visualize
echo.
echo Figures generated! Check plots/ directory.
echo.
pause
goto menu

:all
echo.
echo WARNING: This will run ALL experiments and may take 3-5 hours!
echo.
set /p confirm="Are you sure? (Y/N): "
if /i "%confirm%"=="Y" (
    echo.
    echo Starting complete experiment suite...
    python main.py --all
    echo.
    echo All experiments complete!
) else (
    echo Cancelled.
)
echo.
pause
goto menu

:end
echo.
echo Goodbye!
exit