@echo off
REM Install required Python packages from requirements.txt
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Requirements installation finished.
pause
