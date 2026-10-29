# =============================================================================
# install-task.ps1 — назначенное задание: автозапуск канала при входе пользователя.
#
# Параметры задачи (см. дизайн-документ кита, раздел «Persistence profile»):
#   - путь \AppOps\ — отдельная папка в планировщике (не корень);
#   - имя Maintenance — внешне «бытовое», назначение не выдает;
#   - триггер ВХОД ТЕКУЩЕГО пользователя, контекст Interactive, RunLevel Limited:
#     без сохраненных паролей, без SYSTEM, без прав администратора.
#
# ПОЧЕМУ Register-ScheduledTask, а НЕ `schtasks /SC ONLOGON` (отклонение от
# эскиза в дизайн-документе, задокументировано): по официальной документации
# schtasks ONLOGON — триггер «whenever a user (ANY user) logs on», т.е.
# общесистемный, и его создание требует элевации
# (https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/schtasks-create).
# Register-ScheduledTask с триггером -AtLogOn -User <текущий> и Principal
# Interactive создает точно пер-пользовательскую задачу в контексте текущего
# пользователя без прав администратора
# (https://learn.microsoft.com/en-us/powershell/module/scheduledtasks/register-scheduledtask).
#
# ЗАПУСК:  powershell -NoProfile -ExecutionPolicy Bypass -File .\install-task.ps1
# Удаление: uninstall.ps1 (или вручную: Unregister-ScheduledTask \AppOps\Maintenance)
# =============================================================================

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# ======================= НАСТРОЙКИ ===========================================
# Должен совпадать с $Root из 1-install.ps1.
$Root = Join-Path $env:USERPROFILE 'tsnet'
# Папка и имя задачи в планировщике (бытовые, назначение не раскрывают).
$TaskPath = '\AppOps\'
$TaskName = 'Maintenance'
# ===================== /НАСТРОЙКИ ============================================

$launcher = Join-Path $Root 'start_all.cmd'
if (-not (Test-Path $launcher)) { throw "Не найден $launcher — сначала выполни 1-install.ps1." }

$action    = New-ScheduledTaskAction -Execute $launcher -WorkingDirectory $Root
$trigger   = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$settings  = New-ScheduledTaskSettingsSet `
                -AllowStartIfOnBatteries `
                -DontStopIfGoingOnBatteries `
                -MultipleInstances IgnoreNew `
                -ExecutionTimeLimit ([TimeSpan]::Zero)   # 0 = не убивать по таймауту (супервизор живет вечно)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskPath $TaskPath -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description 'Пользовательские фоновые службы' -Force | Out-Null

Write-Host "Задача создана: $TaskPath$TaskName (вход пользователя $env:USERNAME -> $launcher)"
Write-Host 'Проверить: schtasks /Query /TN "\AppOps\Maintenance" /V'
