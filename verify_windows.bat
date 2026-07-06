@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
REM =====================================================================
REM  verify_windows.bat  -  one-click reproducibility check (Windows)
REM  Repository: stable-architecture-rotating-hubs
REM
REM  Run this FROM THE REPO ROOT in "Anaconda Prompt". It:
REM    1) builds a pinned conda env (python 3.10 + verified package versions)
REM    2) fetches Git LFS data and normalizes line endings so checksums match
REM    3) runs the author's verify.py
REM
REM  PREREQUISITES: Anaconda/Miniconda, Git for Windows (with Git LFS).
REM  (Recommended clone:  git lfs install  &&  git clone -c core.autocrlf=false <repo-url> )
REM =====================================================================
set "ENVNAME=sarh_verify"
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo [1/5] Checking conda...
where conda >nul 2>nul || ( echo   ERROR: conda not found. Open "Anaconda Prompt" or install Miniconda. & pause & exit /b 1 )

echo.
echo [2/5] Creating conda env "%ENVNAME%" (python 3.10) if missing...
call conda env list | findstr /I /R /C:"[\\/]%ENVNAME%$" >nul || call conda create -y -n %ENVNAME% python=3.10
for /f "delims=" %%i in ('conda run -n %ENVNAME% python -c "import sys;print(sys.executable)" 2^>nul') do set "PY=%%i"
if not defined PY ( echo   ERROR: env python not found. & pause & exit /b 1 )
echo   Interpreter: !PY!
"!PY!" -m pip install --upgrade pip >nul

echo.
echo [3/5] Installing pinned dependencies...
if exist "%ROOT%requirements-lock.txt" (
  echo   Using requirements-lock.txt (exact versions)...
  "!PY!" -m pip install -r "%ROOT%requirements-lock.txt"
) else (
  echo   No lock file - installing requirements.txt then overlaying verified pins...
  if exist "%ROOT%requirements.txt" "!PY!" -m pip install -r "%ROOT%requirements.txt"
  "!PY!" -m pip install linearmodels libpysal esda spreg pymannkendall statsmodels ^
    scikit-learn networkx python-louvain pyarrow pydynpd "prettytable==3.18.0"
  "!PY!" -m pip install --force-reinstall --no-cache-dir numpy==1.26.4 scipy==1.13.1
  "!PY!" -m pip install --force-reinstall --no-cache-dir "wcwidth>=0.8.2"
)

echo.
echo [4/5] Fetching Git LFS data + normalizing line endings (so sha256 checksums match)...
where git >nul 2>nul && (
  git lfs install >nul 2>nul
  git lfs pull >nul 2>nul
  git config core.autocrlf false
  git checkout --force >nul 2>nul
)

echo.
echo [5/5] Verification.
echo    [1] Quick (~1-2 min)  : checksums + published values + LIVE FE re-estimation
echo    [2] Full  (30-60 min) : the above + RE-RUN Track A and Track B
echo    [N] Skip
set "GO="
set /p GO="Select [1/2/N]: "
set "KMP_DUPLICATE_LIB_OK=TRUE"
if "!GO!"=="1" ( echo. & echo === python verify.py === & "!PY!" -u verify.py & goto report )
if "!GO!"=="2" ( echo. & echo === python verify.py --full === & "!PY!" -u verify.py --full & goto report )
goto skip

:report
if errorlevel 1 (
  echo.
  echo ============================================================
  echo   VERIFICATION FAILED - read the FAIL lines above.
  echo   If ONLY "sha256 ..." lines failed, that is a byte-integrity
  echo   issue (usually line endings or missing Git LFS data), NOT an
  echo   analysis error: the canonical-value and live-re-estimation
  echo   checks passing means the numbers themselves are correct.
  echo ============================================================
) else (
  echo.
  echo ============================================================
  echo   ALL CHECKS PASSED - results match the published values.
  echo ============================================================
)
goto done

:skip
echo.
echo Skipped. To verify later:
echo    conda activate %ENVNAME%
echo    python verify.py            (quick)
echo    python verify.py --full     (full re-run)

:done
echo.
pause
