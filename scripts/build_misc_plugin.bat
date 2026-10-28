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
REM version. Requires curl.exe and tar-free tooling only (both ship with
REM Windows 10 1803+; tar is not needed here). CRLF line endings.
REM ===========================================================================
chcp 65001 >nul
setlocal

set "MISC_PIN=07e0589a381f7deb3bf533bb459a94482bccc5c7"
set "VS_TAG=R81"

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
set "PY=%REPO_ROOT%\.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ОШИБКА] Не найден интерпретатор проекта: %PY%
    echo Сначала uv sync
    goto :fail
)

set "PYTHONUTF8=1"
for /f "delims=" %%P in ('"%PY%" -c "import vapoursynth, os; print(os.path.dirname(vapoursynth.__file__))"') do set "VS_PKG_DIR=%%P"
if not defined VS_PKG_DIR (
    echo [ОШИБКА] Пакет vapoursynth не импортируется из окружения проекта.
    echo Сначала uv sync
    goto :fail
)
set "PLUGIN_DIR=%VS_PKG_DIR%\plugins\misc"

set "WORK=%TEMP%\misc_build_%RANDOM%"
mkdir "%WORK%"

set "BASE=https://raw.githubusercontent.com/vapoursynth"
curl.exe -fsSL "%BASE%/vs-miscfilters-obsolete/%MISC_PIN%/src/miscfilters.cpp" -o "%WORK%\miscfilters.cpp" || goto :dl_fail
curl.exe -fsSL "%BASE%/vapoursynth/%VS_TAG%/include/VSHelper4.h" -o "%WORK%\VSHelper4.h" || goto :dl_fail
curl.exe -fsSL "%BASE%/vapoursynth/%VS_TAG%/include/VSConstants4.h" -o "%WORK%\VSConstants4.h" || goto :dl_fail
curl.exe -fsSL "%BASE%/vapoursynth/%VS_TAG%/include/VapourSynth4.h" -o "%WORK%\VapourSynth4.h" || goto :dl_fail

if not exist "%PLUGIN_DIR%" mkdir "%PLUGIN_DIR%"

where g++ >nul 2>nul || goto :try_cl
g++ -std=c++17 -O2 -shared -static-libgcc -static-libstdc++ -I"%WORK%" -o "%PLUGIN_DIR%\misc.dll" "%WORK%\miscfilters.cpp" || goto :gxx_fail
goto :built

:try_cl
where cl >nul 2>nul || goto :no_compiler
pushd "%WORK%"
cl /nologo /std:c++17 /O2 /LD /I"%WORK%" miscfilters.cpp /Fe:"%PLUGIN_DIR%\misc.dll" || goto :cl_fail
popd
goto :built

:no_compiler
echo [ОШИБКА] Не найден компилятор C++: нет ни g++, ни cl.exe в PATH.
echo Варианты исправления:
echo   (а) установить MinGW-W64 g++ — standalone WinLibs (https://winlibs.com) или MSYS2
echo       (пакет mingw-w64-x86_64-gcc, ОБЯЗАТЕЛЬНО 64-битный) — и перезапустить этот скрипт;
echo   (б) вручную: взять misc.dll из официального установщика VapourSynth версии R80 или
echo       старее и скопировать его в %APPDATA%\VapourSynth\plugins64\
goto :fail

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

:built
echo [ГОТОВО] Собран: %PLUGIN_DIR%\misc.dll
"%PY%" -c "import vapoursynth as vs; assert hasattr(vs.core, 'misc'), 'misc namespace not visible after build'; print('verified: core.misc is available')" || goto :verify_fail
echo [ПРОВЕРЕНО] core.misc доступен — плагин misc работает.
rd /s /q "%WORK%"
exit /b 0

:verify_fail
echo [ОШИБКА] Проверка не прошла: vapoursynth загрузился, но namespace core.misc не виден.
echo Подсказка: скорее всего misc.dll собран не под 64-битный Python (проверьте разрядность
echo компилятора) либо DLL не экспортирует VapourSynthPluginInit2; логи компилятора выше.
goto :fail

:fail
cd /d "%REPO_ROOT%" >nul 2>nul
if exist "%WORK%" rd /s /q "%WORK%"
exit /b 1
