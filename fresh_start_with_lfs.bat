@echo off
echo ============================================================
echo Fresh Start with Git LFS - Clean Upload
echo ============================================================
echo.
echo This will:
echo 1. Clean up unnecessary files
echo 2. Remove old Git history
echo 3. Setup Git LFS for model files
echo 4. Commit everything cleanly
echo 5. Push to GitHub
echo.
echo WARNING: This will overwrite your GitHub repository!
echo.
pause

echo.
echo Step 1: Cleaning up unnecessary files...
del /q CLEAN_AND_USE_LFS.md 2>nul
del /q FIX_GITHUB_PUSH.md 2>nul
del /q FIX_NOW.md 2>nul
del /q QUICK_FIX.md 2>nul
del /q fix_git.bat 2>nul
del /q download_models.py 2>nul
del /q cleanup_unnecessary_files.bat 2>nul
rmdir /s /q __pycache__ 2>nul
echo Done!

echo.
echo Step 2: Removing old .git folder...
if exist .git (
    rmdir /s /q .git
    echo Done!
) else (
    echo No .git folder found, skipping...
)

echo.
echo Step 3: Initializing fresh Git repository...
git init

echo.
echo Step 4: Setting up Git LFS...
git lfs install
git lfs track "*.pth"

echo.
echo Step 5: Adding all files...
git add .

echo.
echo Step 6: Committing...
git commit -m "Competition submission - Offroad Segmentation (0.73 IoU)"

echo.
echo Step 7: Adding remote...
git remote add origin https://github.com/Satadru-575/Hack-For-Green-Bharat-Submission-Offroad-Semantic-Scene-Segmentation-.git

echo.
echo Step 8: Pushing to GitHub with Git LFS...
echo This may take 5-10 minutes for large model files...
echo.
git push -f origin master

echo.
echo ============================================================
echo Done!
echo ============================================================
echo.
echo Your repository is now live with Git LFS!
echo Check: https://github.com/Satadru-575/Hack-For-Green-Bharat-Submission-Offroad-Semantic-Scene-Segmentation-
echo.
echo Model files should show "Stored with Git LFS" badge.
echo.
pause
