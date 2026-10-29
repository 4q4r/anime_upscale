@echo off
chcp 65001 >nul
rem =============================================================================
rem Комбинированный запуск канала: tailscaled (userspace) + супервизор sshd.
rem
rem Вызывается назначенным заданием \AppOps\Maintenance при входе пользователя
rem (install-task.ps1) либо вручную. Прав администратора не требует:
rem  - tailscaled в userspace-режиме (--tun=userspace-networking) не создаёт
rem    сетевой интерфейс и не грузит драйвер;
rem  - --windows-mode=dev даёт именованный канал CLI \\.\pipe\tsnet-<пользователь>,
rem    доступный только текущему пользователю (tsconst/windowsmode.go:
rem    per-user named pipe, "possibly without administrator rights");
rem  - --statedir держит состояние узла внутри портативного каталога
rem    (в %ProgramData% ничего не пишется);
rem  - sshd в режиме -d работает под текущим пользователем
rem    (см. start_sshd.cmd и sshd_config.template).
rem
rem Файл копируется в корень портативного каталога скриптом 1-install.ps1.
rem =============================================================================
set "ROOT=%~dp0"
if not exist "%ROOT%ts\state" mkdir "%ROOT%ts\state"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"

start "tsnet-tailscaled" /min "%ROOT%ts\tailscaled.exe" --windows-mode=dev --tun=userspace-networking --statedir="%ROOT%ts\state" --socket="\\.\pipe\tsnet-%USERNAME%"
start "tsnet-sshd" /min "%ROOT%start_sshd.cmd"
