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
:loop
cd /d "%ROOT%ssh"
sshd.exe -d -f "%ROOT%ssh\sshd_config" >> "%ROOT%logs\sshd.log" 2>&1
timeout /t 2 /nobreak >nul
goto loop
