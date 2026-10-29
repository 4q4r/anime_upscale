@echo off
chcp 65001 >nul
rem =============================================================================
rem Супервизор sshd — цикл перезапуска.
rem
rem Режим `sshd.exe -d` (без службы, без прав администратора) обслуживает ОДНО
rem подключение и завершается — это официально задокументированное поведение
rem (https://github.com/PowerShell/Win32-OpenSSH/wiki/Troubleshooting-Steps).
rem Поэтому вокруг sshd стоит простой цикл: завершился -> пауза 2 с -> снова
rem слушает. На стороне Linux-агента включается ControlMaster (одна транспортная
rem сессия мультиплексирует все exec- и SFTP-каналы) — см. agent-linux/.
rem
rem Файл копируется в корень портативного каталога скриптом 1-install.ps1.
rem =============================================================================
set "ROOT=%~dp0"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"

rem --- Защита от двойного запуска (ревью PR #12, M-1): второй супервизор
rem --- крутился бы вечно на провале bind 127.0.0.1:2222, раздувая sshd.log.
rem --- Детекция по командной строке в пределах пути канала (паттерн
rem --- согласован с uninstall.ps1); wmic не используем — удален из современных
rem --- Windows. Исключаем СОБСТВЕННЫЙ cmd по PID родителя ($me), иначе первый
rem --- легитимный запуск находил бы сам себя и тихо завершался.
set "GUARDPAT=%ROOT%start_sshd.cmd"
powershell -NoProfile -Command "$me = (Get-CimInstance Win32_Process -Filter \"ProcessId = $PID\").ParentProcessId; if (Get-CimInstance Win32_Process -Filter \"Name = 'cmd.exe'\" | Where-Object { $_.ProcessId -ne $me -and $_.CommandLine -and $_.CommandLine -like ('*' + $env:GUARDPAT + '*') }) { Write-Host '[sshd] супервизор уже запущен - повторный запуск пропущен'; exit 0 } else { exit 1 }"
if not errorlevel 1 exit /b 0

:loop
cd /d "%ROOT%ssh"
sshd.exe -d -f "%ROOT%ssh\sshd_config" >> "%ROOT%logs\sshd.log" 2>&1
timeout /t 2 /nobreak >nul
goto loop
