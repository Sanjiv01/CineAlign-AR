@echo off
REM ============================================================
REM Download Condensed Movies videos LOCALLY (Windows)
REM ============================================================
REM SCC blocks YouTube, so download videos on your local machine
REM and then transfer them to SCC via scp.
REM
REM Prerequisites:
REM   - Python 3.10+ with pandas, yt-dlp installed
REM   - ffmpeg on PATH (for trimming)
REM   - Metadata already downloaded (clips.csv, durations.csv)
REM     If not, this script will download metadata first.
REM
REM Usage:
REM   scripts\download_local.bat                   Full download
REM   scripts\download_local.bat --metadata_only   Metadata only
REM
REM After download, transfer to SCC:
REM   scp -r data\condensed_movies\videos sanjiv@scc1.bu.edu:/projectnb/cs585/students/sanjiv/CineAlign-AR/data/condensed_movies/
REM ============================================================

cd /d "%~dp0\.."

echo ============================================================
echo CineAlign-AR: Local Video Download
echo ============================================================
echo Project root: %CD%
echo.

REM Check yt-dlp is available
where yt-dlp >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] yt-dlp not found. Installing...
    pip install yt-dlp
)

REM Check ffmpeg is available
where ffmpeg >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARN] ffmpeg not found on PATH. Trimming will fail.
    echo   Install from: https://ffmpeg.org/download.html
    echo.
)

REM Run the download script
python dataset_generation\data_prep\download.py ^
    --data_dir data\condensed_movies ^
    --metadata_dir data\metadata ^
    --batch %*

echo.
echo ============================================================
echo Download complete!
echo ============================================================
echo.
echo Next: Transfer videos to SCC:
echo   scp -r data\condensed_movies\videos sanjiv@scc1.bu.edu:/projectnb/cs585/students/sanjiv/CineAlign-AR/data/condensed_movies/
echo.
echo Or use rsync for resumable transfer:
echo   rsync -avP --rsh=ssh data/condensed_movies/videos/ sanjiv@scc1.bu.edu:/projectnb/cs585/students/sanjiv/CineAlign-AR/data/condensed_movies/videos/

pause
