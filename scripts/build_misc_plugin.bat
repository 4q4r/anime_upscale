@echo off
REM ===========================================================================
REM Build the VapourSynth `misc` plugin (SCDetect) into the project .venv.
REM
REM Windows mirror of scripts/build_misc_plugin.sh (the Linux canonical).
REM Why: vsrife's scene-detection path calls misc.SCDetect; the plugin ships
REM with the official VS installers but not with the PyPI vapoursynth wheel,
REM and no PyPI package provides it. This script compiles it from the
REM canonical source (vapoursynth/vs-miscfilters-obsolete, LGPL-2.1) against
REM the VS headers that ship inside the wheel. Re-run after `uv sync`
REM wipes/recreates .venv.
REM
REM Supply-chain pins (identical to the .sh): miscfilters.cpp is fetched by
REM COMMIT; the VS headers come from the R81 release tag. The plugin directory
REM is derived from the RUNNING project interpreter -- no hardcoded python
REM version. Requires curl.exe (ships with Windows 10 1803+). If no C++
REM compiler is present, the script bootstraps one itself, unattended, via
REM winget (installs winget from the official winget-cli release first if
REM needed): WinLibs portable g++ (BrechtSanders.WinLibs.POSIX.UCRT), then
REM MSYS2 + pacman mingw-w64-x86_64-gcc as the fallback. CRLF line endings.
REM ===========================================================================
chcp 65001 >nul
setlocal

set "MISC_PIN=07e0589a381f7deb3bf533bb459a94482bccc5c7"
set "VS_TAG=R81"

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"
set "REPO_ROOT=%CD%"
set "PY=%REPO_ROOT%\.venv\Scripts\python.exe"

echo [СКРИПТ] Каталог проекта: %REPO_ROOT%
echo [СКРИПТ] Интерпретатор: %PY%

if not exist "%PY%" (
    echo [ОШИБКА] Не найден интерпретатор проекта: %PY%
    echo Сначала uv sync
    goto :fail
)

set "WORK=%TEMP%\misc_build_%RANDOM%"
mkdir "%WORK%"

set "PYTHONUTF8=1"
set "VS_PKG_DIR="
"%PY%" -c "import vapoursynth, os; print(os.path.dirname(vapoursynth.__file__))" > "%WORK%\vsdir.txt" 2>nul
if errorlevel 1 goto :vs_fail
set /p VS_PKG_DIR=<"%WORK%\vsdir.txt"
if not defined VS_PKG_DIR goto :vs_fail
if not exist "%VS_PKG_DIR%" goto :vs_missing
set "PLUGIN_DIR=%VS_PKG_DIR%\plugins\misc"

set "BASE=https://raw.githubusercontent.com/vapoursynth"
curl.exe -fsSL --connect-timeout 15 --max-time 120 "%BASE%/vs-miscfilters-obsolete/%MISC_PIN%/src/miscfilters.cpp" -o "%WORK%\miscfilters.cpp" || goto :dl_fail
curl.exe -fsSL --connect-timeout 15 --max-time 120 "%BASE%/vapoursynth/%VS_TAG%/include/VSHelper4.h" -o "%WORK%\VSHelper4.h" || goto :dl_fail
curl.exe -fsSL --connect-timeout 15 --max-time 120 "%BASE%/vapoursynth/%VS_TAG%/include/VSConstants4.h" -o "%WORK%\VSConstants4.h" || goto :dl_fail
curl.exe -fsSL --connect-timeout 15 --max-time 120 "%BASE%/vapoursynth/%VS_TAG%/include/VapourSynth4.h" -o "%WORK%\VapourSynth4.h" || goto :dl_fail

if not exist "%PLUGIN_DIR%" mkdir "%PLUGIN_DIR%"

where g++ >nul 2>nul || goto :try_cl
g++ -std=c++17 -O2 -shared -static-libgcc -static-libstdc++ -I"%WORK%" -o "%PLUGIN_DIR%\misc.dll" "%WORK%\miscfilters.cpp" || goto :gxx_fail
goto :built

:try_cl
where cl >nul 2>nul || goto :bootstrap_compiler
pushd "%WORK%"
cl /nologo /std:c++17 /O2 /LD /I"%WORK%" miscfilters.cpp /Fe:"%PLUGIN_DIR%\misc.dll" || goto :cl_fail
popd
goto :built

:no_compiler
echo [ОШИБКА] Автоустановка компилятора не удалась.
echo Итог попыток:
echo   1) WinLibs (portable g++) через winget — установлена, но g++ не найден;
echo   2) MSYS2: %MSYS2_NOTE%
echo Вывод winget/pacman — выше по тексту.
echo Последний резерв вручную (если автоматизация недоступна):
echo   (а) скачать и распаковать 64-битный WinLibs (https://winlibs.com) и добавить
echo       его mingw64\bin в PATH, затем перезапустить этот скрипт;
echo   (б) взять misc.dll из официального установщика VapourSynth версии R80 или
echo       старее и скопировать его в %APPDATA%\VapourSynth\plugins64\
goto :fail

:bootstrap_compiler
echo [ЭТАП] Компилятор не найден — устанавливаю через winget…
winget --version >nul 2>nul || goto :install_winget
goto :have_winget

:install_winget
echo [ЭТАП] winget не найден — устанавливаю App Installer с официального релиза winget-cli…
set "WINGET_URL="
powershell -NoProfile -Command "(Invoke-RestMethod -Uri 'https://api.github.com/repos/microsoft/winget-cli/releases/latest' -TimeoutSec 120).assets | Where-Object { $_.name -eq 'Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle' } | Select-Object -First 1 -ExpandProperty browser_download_url" > "%WORK%\winget_url.txt"
set /p WINGET_URL=<"%WORK%\winget_url.txt"
if not defined WINGET_URL goto :winget_fail
curl.exe -fsSL --connect-timeout 15 --max-time 300 -o "%WORK%\winget.msixbundle" "%WINGET_URL%" || goto :winget_fail
powershell -NoProfile -Command "Add-AppxPackage -Path '%WORK%\winget.msixbundle'" || goto :winget_fail
winget --version >nul 2>nul || goto :winget_fail

:have_winget
echo [ЭТАП] winget доступен — проверяю WinLibs g++ (portable, 64-битный)…
REM Probe-first: winget exits NON-zero on "already installed", so locate
REM BEFORE installing and re-locate after any exit code (normalize).
call :locate_gxx
if defined GXX_BIN goto :have_gxx
echo [ЭТАП] WinLibs g++ не найден — устанавливаю через winget…
winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e --silent --accept-source-agreements --accept-package-agreements --disable-interactivity
call :locate_gxx
if defined GXX_BIN goto :have_gxx
goto :msys2_route

:have_gxx
set "PATH=%GXX_BIN%;%PATH%"
where g++ >nul 2>nul || goto :msys2_route
echo [ЭТАП] g++ доступен: %GXX_BIN%
goto :compile_gxx

:msys2_route
REM MSYS2.MSYS2 is ElevationRequirement: elevationRequired — from a
REM non-elevated console its UAC prompt would wait on a human. Probe first.
set "MSYS2_NOTE=установка MSYS2 не удалась"
net session >nul 2>&1 || goto :msys2_no_elevation
if exist "C:\msys64\mingw64\bin\g++.exe" goto :have_msys2_gxx
echo [ЭТАП] MSYS2 g++ не найден — устанавливаю MSYS2 через winget…
winget install --id MSYS2.MSYS2 -e --silent --accept-source-agreements --accept-package-agreements --disable-interactivity
if not exist "C:\msys64\usr\bin\bash.exe" goto :no_compiler
echo [ЭТАП] Устанавливаю mingw-w64-x86_64-gcc через pacman…
"C:\msys64\usr\bin\bash.exe" -lc "pacman -S --noconfirm --needed mingw-w64-x86_64-gcc" || goto :no_compiler
if not exist "C:\msys64\mingw64\bin\g++.exe" goto :no_compiler

:have_msys2_gxx
set "GXX_BIN=C:\msys64\mingw64\bin"
set "PATH=%GXX_BIN%;%PATH%"
echo [ЭТАП] g++ доступен: %GXX_BIN%
goto :compile_gxx

:msys2_no_elevation
echo [ОШИБКА] MSYS2 требует прав администратора (UAC) — пропущен в неинтерактивном режиме.
set "MSYS2_NOTE=пропущен: требует прав администратора (UAC)"
goto :no_compiler

:compile_gxx
echo [ЭТАП] Собираю misc.dll…
g++ -std=c++17 -O2 -shared -static-libgcc -static-libstdc++ -I"%WORK%" -o "%PLUGIN_DIR%\misc.dll" "%WORK%\miscfilters.cpp" || goto :gxx_fail
goto :built

:gxx_fail
echo [ОШИБКА] g++ завершился с ошибкой при сборке misc.dll.
echo Подсказка: g++ должен быть 64-битным (MinGW-W64 x86_64 — WinLibs/MSYS2);
echo 32-битный компилятор не слинкуется с 64-битным Python.
goto :fail

:cl_fail
popd
echo [ОШИБКА] cl.exe завершился с ошибкой при сборке misc.dll.
echo Подсказка: cl работает только из окружения Visual Studio —
echo запускайте скрипт из "x64 Native Tools Command Prompt for VS" (Developer Command Prompt),
echo либо используйте вариант (а) с MinGW-W64 g++.
goto :fail

:dl_fail
echo [ОШИБКА] Не удалось скачать закреплённые исходники с raw.githubusercontent.com.
echo Подсказка: curl.exe встроен в Windows 10 начиная с версии 1803; проверьте сеть
echo и доступность https://raw.githubusercontent.com
goto :fail

:winget_fail
echo [ОШИБКА] Не удалось установить winget (App Installer) автоматически.
echo Подсказка: на Windows Server без предпосылок App Installer (VCLibs, UI.Xaml)
echo установка msixbundle может завершаться ошибкой — инструкция по ручной установке:
echo https://learn.microsoft.com/windows/package-manager/winget/
echo Последний резерв вручную:
echo   (а) скачать и распаковать 64-битный WinLibs (https://winlibs.com) и добавить
echo       его mingw64\bin в PATH, затем перезапустить этот скрипт;
echo   (б) взять misc.dll из официального установщика VapourSynth версии R80 или
echo       старее и скопировать его в %APPDATA%\VapourSynth\plugins64\
goto :fail

:vs_fail
echo [ОШИБКА] Пакет vapoursynth не импортируется из окружения проекта.
echo Сначала uv sync
goto :fail

:vs_missing
echo [ОШИБКА] Каталог пакета vapoursynth не найден: %VS_PKG_DIR%
echo Сначала uv sync
goto :fail

:built
echo [ГОТОВО] Собран: %PLUGIN_DIR%\misc.dll
"%PY%" -c "import vapoursynth as vs; assert hasattr(vs.core, 'misc'), 'misc namespace not visible after build'; print('verified: core.misc is available')" || goto :verify_fail
echo [ПРОВЕРЕНО] core.misc доступен — плагин misc работает.
rd /s /q "%WORK%"
popd
exit /b 0

:verify_fail
echo [ОШИБКА] Проверка не прошла: vapoursynth загрузился, но namespace core.misc не виден.
echo Подсказка: скорее всего misc.dll собран не под 64-битный Python (проверьте разрядность
echo компилятора) либо DLL не экспортирует VapourSynthPluginInit2; логи компилятора выше.
goto :fail

:locate_gxx
REM WinLibs manifest: InstallerType zip + NestedInstallerType portable +
REM ArchiveBinariesDependOnPath true -> per the winget installer schema the
REM install location is added directly to the (registry) PATH. The session
REM PATH is NOT refreshed either way, so locate and prepend manually:
REM Links shims first, then a Packages walk as belt-and-suspenders.
set "GXX_BIN="
if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\g++.exe" set "GXX_BIN=%LOCALAPPDATA%\Microsoft\WinGet\Links"
if defined GXX_BIN exit /b 0
set "GXX_ANY="
for /r "%LOCALAPPDATA%\Microsoft\WinGet\Packages" %%G in (g++.exe) do if not defined GXX_ANY set "GXX_ANY=%%~dpG"
set "GXX_BIN=%GXX_ANY%"
exit /b 0

:fail
popd
if exist "%WORK%" rd /s /q "%WORK%"
exit /b 1
