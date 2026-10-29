# =============================================================================
# uninstall.ps1 — полный и аккуратный снос канала.
#
# 1) удаляет назначенное задание \AppOps\Maintenance;
# 2) останавливает ТОЛЬКО процессы, запущенные из портативного каталога
#    (отбор по полному пути/командной строке, завершение по точным PID —
#    никаких массовых kill по имени);
# 3) удаляет сам портативный каталог целиком;
# 4) проверяет, что следов не осталось.
#
# Права администратора не нужны: задание создано от этого пользователя,
# процессы запущены от этого пользователя.
#
# ЗАПУСК:  powershell -NoProfile -ExecutionPolicy Bypass -File .\uninstall.ps1
# =============================================================================

[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

# ======================= НАСТРОЙКИ ===========================================
# Должен совпадать с $Root из 1-install.ps1.
$Root = Join-Path $env:USERPROFILE 'tsnet'
$TaskPath = '\AppOps\'
$TaskName = 'Maintenance'
# ===================== /НАСТРОЙКИ ============================================

# --- 1. Задача ----------------------------------------------------------------
Write-Host "[1/4] Удаляю задачу $TaskPath$TaskName"
Unregister-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# --- 2. Процессы ---------------------------------------------------------------
Write-Host '[2/4] Останавливаю процессы портативного каталога'
# tailscaled/sshd/sshd-session/... — по полному пути исполняемого файла;
# cmd-супервизор — по командной строке (у cmd.exe путь системный, путь к
# каталогу виден только в аргументах).
$victims = Get-CimInstance Win32_Process | Where-Object {
    ($_.ExecutablePath -ne $null -and $_.ExecutablePath -like "$Root*") -or
    ($_.CommandLine -ne $null -and $_.CommandLine -like "*$Root*")
}
foreach ($p in $victims) {
    Write-Host ("  PID {0}: {1}" -f $p.ProcessId, $p.Name)
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue   # только точный PID
}
Start-Sleep -Seconds 1

# --- 3. Файлы ------------------------------------------------------------------
Write-Host "[3/4] Удаляю каталог $Root"
if (Test-Path $Root) { Remove-Item -LiteralPath $Root -Recurse -Force }

# --- 4. Проверка чистоты --------------------------------------------------------
Write-Host '[4/4] Проверка'
$leftTask = Get-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName -ErrorAction SilentlyContinue
$leftProc = Get-CimInstance Win32_Process | Where-Object {
    ($_.ExecutablePath -ne $null -and $_.ExecutablePath -like "$Root*") -or
    ($_.CommandLine -ne $null -and $_.CommandLine -like "*$Root*")
}
if (-not (Test-Path $Root) -and -not $leftTask -and -not $leftProc) {
    Write-Host 'Удалено полностью: задачи нет, процессов нет, файлов нет.'
} else {
    Write-Warning 'Что-то осталось — проверь руками:'
    if ($leftTask) { Write-Warning "  задача: $TaskPath$TaskName" }
    if ($leftProc) { $leftProc | ForEach-Object { Write-Warning ("  процесс: PID {0} {1}" -f $_.ProcessId, $_.Name) } }
    if (Test-Path $Root) { Write-Warning "  каталог: $Root" }
}
