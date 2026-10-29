# =============================================================================
# 1-install.ps1 — установка портативного канала на Windows (БЕЗ прав администратора).
#
# Что делает:
#   1) создаёт портативный каталог $Root (по умолчанию %USERPROFILE%\tsnet);
#   2) скачивает MSI Tailscale и извлекает его БЕЗ установки
#      (msiexec /a — административный образ = просто распаковка файлов,
#      в «Установленные программы» ничего не появляется, службы не создаются);
#   3) скачивает переносимый архив Win32-OpenSSH и распаковывает в $Root\ssh;
#   4) генерирует хост-ключ sshd (ed25519) без элевации;
#   5) материализует sshd_config из шаблона с абсолютными портативными путями
#      (легальность абсолютного AuthorizedKeysFile: ветка WINDOWS в
#      expand_authorized_keys() возвращает путь «как есть» —
#      https://github.com/PowerShell/openssh-portable/blob/latestw_all/auth.c);
#   6) создаёт пустой authorized_keys (в него нужно вставить публичный ключ
#      агента — см. README в корне кита) и нормализует права (icacls);
#   7) копирует start_all.cmd и start_sshd.cmd в корень портативного каталога.
#
# ЗАПУСК:  powershell -NoProfile -ExecutionPolicy Bypass -File .\1-install.ps1
# После установки: 2-start.ps1 (первый подъем и регистрация в tailnet),
# затем install-task.ps1 (автозапуск при входе).
# =============================================================================

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # Invoke-WebRequest без полосы прогресса (в разы быстрее)

# ======================= НАСТРОЙКИ (заполнить!) ==============================
# Адрес координатора (Headscale на VPS) — тот же, что server_url на сервере.
$LoginServer = 'https://hs.example.com'

# Версия стабильного MSI Tailscale. Актуальную смотри на
# https://pkgs.tailscale.com/stable/ (файл вида tailscale-setup-<версия>-amd64.msi).
# Проверено 2026-10-30: текущая стабильная = 1.104.4.
$TailscaleVersion = '1.104.4'

# Версия Win32-OpenSSH (тег релиза). Последний релиз проверен 2026-10-30:
# 10.4.0.0p1-Preview — https://github.com/PowerShell/Win32-OpenSSH/releases
$OpenSshVersion = '10.4.0.0p1-Preview'

# Корень портативного каталога: ВСЁ живет здесь, наружу — только назначенное
# задание (install-task.ps1). Меняешь — меняй и переустановку задания.
$Root = Join-Path $env:USERPROFILE 'tsnet'

# Порт sshd (должен совпадать с ожиданиями агента; доставки на этот порт из
# tailnet на 127.0.0.1 делает сам tailscaled — см. README кита, раздел «Схема»).
$SshPort = 2222
# ===================== /НАСТРОЙКИ ============================================

if ($LoginServer -match 'example\.com' -or [string]::IsNullOrWhiteSpace($LoginServer)) {
    throw 'Не заполнен $LoginServer — впиши адрес своего Headscale-сервера в блоке НАСТРОЙКИ.'
}

$TailscaleMsiUrl = "https://pkgs.tailscale.com/stable/tailscale-setup-$TailscaleVersion-amd64.msi"
$OpenSshZipUrl   = "https://github.com/PowerShell/Win32-OpenSSH/releases/download/v$OpenSshVersion/OpenSSH-Win64.zip"

# --- 1. Каркас каталогов ------------------------------------------------------
$dirTs   = Join-Path $Root 'ts'
$dirSsh  = Join-Path $Root 'ssh'
$dirLogs = Join-Path $Root 'logs'
$dirDl   = Join-Path $Root 'dl'
foreach ($d in @($dirTs, (Join-Path $dirTs 'state'), $dirSsh, $dirLogs, $dirDl)) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}
Write-Host "[1/7] Каталоги созданы: $Root"

# --- 2. Tailscale: скачать MSI и извлечь без установки ------------------------
$msiPath = Join-Path $dirDl "tailscale-setup-$TailscaleVersion-amd64.msi"
if (-not (Test-Path $msiPath)) {
    Write-Host "[2/7] Скачиваю $TailscaleMsiUrl"
    Invoke-WebRequest -Uri $TailscaleMsiUrl -OutFile $msiPath -UseBasicParsing
}
# msiexec /a = создание административного образа (распаковка). Ничего не
# регистрируется: ни записи в реестре установок, ни служб, ни драйверов.
$msiExtract = Join-Path $dirDl 'msi-extract'
if (Test-Path $msiExtract) { Remove-Item $msiExtract -Recurse -Force }
Start-Process -FilePath 'msiexec.exe' -ArgumentList "/a `"$msiPath`" /qn TARGETDIR=`"$msiExtract`"" -Wait

# Извлечённые бинарники ищем в глубине распаковки (раскладка MSI зависит от версии).
$tscaledExe = Get-ChildItem -Path $msiExtract -Recurse -Filter 'tailscaled.exe' | Select-Object -First 1
if ($null -eq $tscaledExe) { throw "tailscaled.exe не найден в $msiExtract — распаковка MSI прошла нештатно." }
$tscaleExe = Join-Path $tscaledExe.DirectoryName 'tailscale.exe'
if (-not (Test-Path $tscaleExe)) { throw 'tailscale.exe не найден рядом с tailscaled.exe.' }
Copy-Item $tscaledExe.FullName (Join-Path $dirTs 'tailscaled.exe') -Force
Copy-Item $tscaleExe        (Join-Path $dirTs 'tailscale.exe')    -Force
Write-Host "[2/7] Tailscale $TailscaleVersion извлечен в $dirTs (без установки)"

# --- 3. Win32-OpenSSH: скачать и распаковать ----------------------------------
$zipPath = Join-Path $dirDl 'OpenSSH-Win64.zip'
if (-not (Test-Path $zipPath)) {
    Write-Host "[3/7] Скачиваю $OpenSshZipUrl"
    Invoke-WebRequest -Uri $OpenSshZipUrl -OutFile $zipPath -UseBasicParsing
}
$zipExtract = Join-Path $dirDl 'ssh-zip'
if (Test-Path $zipExtract) { Remove-Item $zipExtract -Recurse -Force }
Expand-Archive -Path $zipPath -DestinationPath $zipExtract -Force
# Архив содержит верхнюю папку OpenSSH-Win64\ — поднимаем содержимое в $Root\ssh.
$sshSrc = Join-Path $zipExtract 'OpenSSH-Win64'
if (-not (Test-Path $sshSrc)) { $sshSrc = $zipExtract }
Copy-Item (Join-Path $sshSrc '*') $dirSsh -Recurse -Force
Write-Host "[3/7] Win32-OpenSSH $OpenSshVersion распакован в $dirSsh"

# --- 4. Хост-ключ sshd (генерация от пользователя, без элевации) --------------
$hostKey = Join-Path $dirSsh 'ssh_host_ed25519_key'
if (-not (Test-Path $hostKey)) {
    Write-Host '[4/7] Генерирую хост-ключ ed25519'
    & (Join-Path $dirSsh 'ssh-keygen.exe') -t ed25519 -f $hostKey -N '""' | Out-Null
    if (-not (Test-Path $hostKey)) {
        # Страховка на случай, когда PowerShell съел пустой аргумент passphrase.
        & (Join-Path $dirSsh 'ssh-keygen.exe') -t ed25519 -f $hostKey -N '' | Out-Null
    }
    if (-not (Test-Path $hostKey)) { throw 'Хост-ключ не создан — проверь запуск ssh-keygen вручную.' }
}
Write-Host "[4/7] Хост-ключ: $hostKey"

# --- 5. sshd_config из шаблона ------------------------------------------------
$templatePath = Join-Path $PSScriptRoot 'sshd_config.template'
if (-not (Test-Path $templatePath)) { throw "Не найден шаблон: $templatePath (запускай скрипт из каталога windows/)." }
$forwardRoot = $Root -replace '\\', '/'
$template = Get-Content -LiteralPath $templatePath -Raw -Encoding UTF8
$config = $template.Replace('__TSNET__', $forwardRoot).Replace('__PORT__', [string]$SshPort)
$configPath = Join-Path $dirSsh 'sshd_config'
# UTF8 без BOM: sshd_config парсится побайтово, BOM сломал бы первую строку.
[System.IO.File]::WriteAllText($configPath, $config, (New-Object System.Text.UTF8Encoding($false)))
Write-Host "[5/7] sshd_config: $configPath (Port $SshPort, ListenAddress 127.0.0.1)"

# --- 6. authorized_keys + права -----------------------------------------------
$authKeys = Join-Path $dirSsh 'authorized_keys'
if (-not (Test-Path $authKeys)) {
    # ВСТАВЬ сюда ОДНОЙ строкой публичный ключ агента (содержимое
    # ~/.ssh/agent_ed25519.pub на Linux-машине).
    Set-Content -LiteralPath $authKeys -Value '' -Encoding ASCII
}
# Нормализация ACL: владелец — текущий пользователь, чужая запись запрещена.
# Строгая проверка sshd (w32-sshfileperm.c) требует: владелец = пользователь/
# Администраторы/SYSTEM и запись только им; наследуемые ACL обычно проходят,
# icacls делает это детерминированным.
icacls $authKeys /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
icacls $hostKey /inheritance:r /grant:r "$($env:USERNAME):F" | Out-Null
Write-Host "[6/7] authorized_keys: $authKeys — ВСТАВЬ публичный ключ агента одной строкой!"

# --- 7. Лаунчеры в корень портативного каталога --------------------------------
foreach ($f in @('start_all.cmd', 'start_sshd.cmd')) {
    Copy-Item (Join-Path $PSScriptRoot $f) (Join-Path $Root $f) -Force
}
Write-Host "[7/7] Лаунчеры скопированы в $Root"

Write-Host ''
Write-Host 'Готово. Дальше:'
Write-Host "  1) вставь публичный ключ агента в $authKeys"
Write-Host '  2) заполни $LoginServer/$AuthKey в 2-start.ps1 и запусти его (первый подъем + регистрация)'
Write-Host '  3) запусти install-task.ps1 — автозапуск при входе в систему'
