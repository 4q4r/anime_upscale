# =============================================================================
# 2-start.ps1 — первый подъем канала и регистрация узла в tailnet.
#
# Запускает (каждый — только если еще не запущен):
#   - tailscaled в userspace-режиме (без интерфейса, без драйвера, без админа);
#   - супервизор sshd (start_sshd.cmd);
# затем регистрирует узел на координаторе одноразовым preauth-ключом.
#
# Регистрация нужна ОДИН раз: состояние узла (--statedir) переживает перезагрузки,
# дальше start_all.cmd из назначенного задания поднимает канал автоматически.
#
# ЗАПУСК:  powershell -NoProfile -ExecutionPolicy Bypass -File .\2-start.ps1
# =============================================================================

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# ======================= НАСТРОЙКИ (заполнить!) ==============================
# Адрес координатора — тот же, что в 1-install.ps1 / на сервере.
$LoginServer = 'https://hs.example.com'

# Одноразовый preauth-ключ для ЭТОЙ машины. Создается на VPS:
#   docker compose exec headscale headscale preauthkeys create --user owner
# (живает 1 час, одноразовый — создавай прямо перед этим запуском).
$AuthKey = 'ВСТАВЬ_КЛЮЧ_ЗДЕСЬ'

# Должен совпадать с $Root из 1-install.ps1.
$Root = Join-Path $env:USERPROFILE 'tsnet'
# ===================== /НАСТРОЙКИ ============================================

if ($AuthKey -match 'ВСТАВЬ|CHANGE' -or [string]::IsNullOrWhiteSpace($AuthKey)) {
    throw 'Не заполнен $AuthKey — создай preauth-ключ на VPS и впиши его в блок НАСТРОЙКИ.'
}

$tsDir     = Join-Path $Root 'ts'
$socket    = "\\.\pipe\tsnet-$env:USERNAME"   # именованный канал CLI; режим dev разрешает любой путь вне защищенного префикса

# --- tailscaled ----------------------------------------------------------------
# Ищем наш процесс ТОЧНО по пути (чужой tailscaled не трогаем).
$running = Get-CimInstance Win32_Process -Filter "Name = 'tailscaled.exe'" |
    Where-Object { $_.ExecutablePath -like "$tsDir*" }
if ($running) {
    Write-Host '[tailscaled] уже запущен, пропускаю'
} else {
    Write-Host "[tailscaled] запускаю (userspace, состояние: $tsDir\state)"
    # ВНИМАНИЕ: Start-Process в Windows PowerShell 5.1 НЕ экранирует аргументы
    # с пробелами сам — кавычки внутри значений ставим явно (профиль/имя
    # пользователя могут содержать пробелы).
    Start-Process -FilePath (Join-Path $tsDir 'tailscaled.exe') -WindowStyle Hidden -ArgumentList @(
        '--windows-mode=dev'
        '--tun=userspace-networking'
        "`"--statedir=$(Join-Path $tsDir 'state')`""
        "`"--socket=$socket`""
    )
    Start-Sleep -Seconds 3   # даем демону подняться перед CLI-командами
}

# --- Регистрация на координаторе ------------------------------------------------
# С флагами --login-server + --authkey поток неинтерактивный: браузерный
# fallback не срабатывает никогда. Сокет указываем явно — детерминированно.
Write-Host "[tailscale] регистрирую узел на $LoginServer"
& (Join-Path $tsDir 'tailscale.exe') --socket $socket up --login-server "$LoginServer" --authkey "$AuthKey"
if ($LASTEXITCODE -ne 0) { throw "tailscale up завершился с кодом $LASTEXITCODE (проверь ключ/доступность сервера)." }

Write-Host '[tailscale] статус:'
& (Join-Path $tsDir 'tailscale.exe') --socket $socket status
if ($LASTEXITCODE -ne 0) { throw 'tailscale status не отработал.' }

# --- Супервизор sshd -------------------------------------------------------------
$sshdRunning = Get-CimInstance Win32_Process -Filter "Name = 'cmd.exe'" |
    Where-Object { $_.CommandLine -like "*`"$Root\start_sshd.cmd`"*" -or $_.CommandLine -like "*$Root\start_sshd.cmd*" }
if ($sshdRunning) {
    Write-Host '[sshd] супервизор уже запущен'
} else {
    Write-Host '[sshd] запускаю супервизор'
    Start-Process -FilePath (Join-Path $Root 'start_sshd.cmd') -WindowStyle Hidden
}

Write-Host ''
Write-Host 'Канал поднят. 100.x-адрес узла — в статусе выше.'
Write-Host 'Проверка с Linux-агента: ssh -p 2222 <пользователь>@<100.x-адрес>'
Write-Host 'Автозапуск при входе: install-task.ps1'
