@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem LLM ModelBench Git sync helper for Windows.
rem Usage:
rem   scripts\llmb-git-sync.bat pull
rem   scripts\llmb-git-sync.bat push
rem   scripts\llmb-git-sync.bat sync
rem   scripts\llmb-git-sync.bat status
rem
rem Safety:
rem - Pull/rebase refuses to run with a dirty worktree.
rem - Push first fetches origin/main and refuses if local main is behind.
rem - Uses SSH key auth only; never switches to HTTPS.

set "MODE=%~1"
if "%MODE%"=="" set "MODE=sync"

set "REMOTE=origin"
set "BRANCH=main"
set "REMOTE_URL=git@github.com:chrsdme/llm-modelbench.git"
set "SSH_KEY=%USERPROFILE%\.ssh\id_ed25519"

where git >nul 2>nul
if errorlevel 1 (
  echo ERROR: git is not on PATH.
  exit /b 1
)

git rev-parse --show-toplevel >nul 2>nul
if errorlevel 1 (
  echo ERROR: not inside a git repository.
  exit /b 1
)

for /f "delims=" %%R in ('git rev-parse --show-toplevel') do set "ROOT=%%R"
cd /d "%ROOT%"

echo Repo: %ROOT%
echo Mode: %MODE%
echo.

call :ensure_ssh || exit /b 1

if /I "%MODE%"=="status" (
  call :show_status
  exit /b %ERRORLEVEL%
)

if /I "%MODE%"=="pull" (
  call :pull_latest
  exit /b %ERRORLEVEL%
)

if /I "%MODE%"=="push" (
  call :push_current
  exit /b %ERRORLEVEL%
)

if /I "%MODE%"=="sync" (
  call :pull_latest || exit /b 1
  call :push_current
  exit /b %ERRORLEVEL%
)

echo ERROR: unknown mode "%MODE%".
echo Usage: scripts\llmb-git-sync.bat [pull^|push^|sync^|status]
exit /b 1

:ensure_ssh
echo Configuring SSH remote and repo-local SSH command...
git remote set-url %REMOTE% %REMOTE_URL%
git remote set-url --push %REMOTE% %REMOTE_URL%
git config core.sshCommand "ssh -i %SSH_KEY:\=/% -o IdentitiesOnly=yes"

if not exist "%SSH_KEY%" (
  echo WARNING: SSH key not found: %SSH_KEY%
  echo Push may fail until this key exists and its public key is registered with GitHub.
)
exit /b 0

:require_clean
for /f %%S in ('git status --porcelain ^| find /c /v ""') do set "DIRTY_COUNT=%%S"
if not "%DIRTY_COUNT%"=="0" (
  echo ERROR: worktree is not clean. Commit/stash/discard local changes before pulling/rebasing.
  git status --short
  exit /b 1
)
exit /b 0

:pull_latest
echo Fetching %REMOTE%...
git fetch %REMOTE% --tags --prune || exit /b 1

call :require_clean || exit /b 1

for /f "delims=" %%B in ('git branch --show-current') do set "CURRENT_BRANCH=%%B"
if not "%CURRENT_BRANCH%"=="%BRANCH%" (
  echo ERROR: current branch is "%CURRENT_BRANCH%", expected "%BRANCH%".
  exit /b 1
)

echo Rebasing %BRANCH% onto %REMOTE%/%BRANCH%...
git rebase %REMOTE%/%BRANCH% || (
  echo ERROR: rebase failed. Resolve conflicts, then run:
  echo   git rebase --continue
  echo or:
  echo   git rebase --abort
  exit /b 1
)

echo Pull sync complete.
call :show_status
exit /b 0

:push_current
echo Fetching %REMOTE% before push...
git fetch %REMOTE% --tags --prune || exit /b 1

for /f "delims=" %%B in ('git branch --show-current') do set "CURRENT_BRANCH=%%B"
if not "%CURRENT_BRANCH%"=="%BRANCH%" (
  echo ERROR: current branch is "%CURRENT_BRANCH%", expected "%BRANCH%".
  exit /b 1
)

git merge-base --is-ancestor %REMOTE%/%BRANCH% HEAD
if errorlevel 1 (
  echo ERROR: local %BRANCH% is not based on %REMOTE%/%BRANCH%.
  echo Run:
  echo   scripts\llmb-git-sync.bat pull
  echo then resolve any rebase conflicts before pushing.
  exit /b 1
)

echo Pushing %BRANCH%...
git push %REMOTE% %BRANCH% || exit /b 1

echo Pushing tags...
git push %REMOTE% --tags || exit /b 1

echo Push sync complete.
call :show_status
exit /b 0

:show_status
echo.
echo Remote:
git remote -v
echo.
echo HEAD:
git show --no-patch --format="%%h %%D %%s" HEAD
echo.
echo Local status:
git status --short
echo.
echo Recent tags:
git tag --sort=-creatordate | more +0
exit /b 0
