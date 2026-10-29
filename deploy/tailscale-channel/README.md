# Канал управления: Headscale + userspace-Tailscale + переносимый sshd

Кит развертывания скрытого канала управления Windows-машиной с Linux-агента.
Базируется на проверенном дизайн-документе (исследование 2026-10-29/30,
primary sources: исходники tailscale/headscale/Win32-OpenSSH + официальные
доки; полная версия — в архиве проекта, ключевые цитаты вынесены в файлы кита).

Граница приемлемости (зафиксирована владельцем): БЕЗ техник руткитов.
Процессы в диспетчере задач видны — это принято; запрещено только появление
в «Установленных программах», службах и типовых местах автозапуска. Для
ежедневной работы права администратора не нужны нигде.

## Схема

```
Linux-агент ──WireGuard-UDP (прямой)──┐
                                      ├──> VPS Headscale (координатор, TLS 443)
Windows ──userspace tailscaled────────┘         │
  └─ sshd 127.0.0.1:2222 <── tailscaled netstack <┘ (вход 100.x:2222 → 127.0.0.1:2222)
```

Ключевой факт схемы: в userspace-режиме tailscaled пересылает ВХОДЯЩИЕ
подключения с адреса 100.x на **тот же порт на 127.0.0.1** автоматически
(`wgengine/netstack/netstack.go`, acceptTCP) — переносимый sshd на loopback
доступен извне tailnet без файрвол-правил, интерфейсов и прав администратора.
Когда прямое WireGuard-UDP не пробивается (NAT/ТСПУ-особенности), трафик
идет через DERP-релей — TLS поверх 443, это обычный HTTPS для промежуточных
узлов; релей — наш собственный VPS.

## Состав кита

| Путь                          | Назначение                                             |
| :---------------------------- | :----------------------------------------------------- |
| `vps/docker-compose.yml`      | Headscale v0.29.6 + встроенный DERP (compose)          |
| `vps/config.yaml`             | Конфиг сервера, комментарии на русском                 |
| `vps/README-vps.md`           | Шаг A: DNS → запуск → пользователь → preauth-ключ      |
| `windows/1-install.ps1`       | Шаг B-1: скачивание/распаковка бинарей, ключи, конфиги |
| `windows/2-start.ps1`         | Шаг B-2: первый подъем + регистрация в tailnet         |
| `windows/install-task.ps1`    | Шаг B-3: задание на автозапуск при входе               |
| `windows/uninstall.ps1`       | Полный чистый снос (задача + процессы + файлы)         |
| `windows/start_all.cmd`       | Лаунчер обоих процессов (цель задания)                 |
| `windows/start_sshd.cmd`      | Супервизор sshd (цикл перезапуска)                     |
| `windows/sshd_config.template`| Шаблон sshd_config (портативные абсолютные пути)       |
| `agent-linux/README-agent.md` | Шаг C: установка, режимы, проверка, troubleshooting    |
| `agent-linux/join_tailnet.sh` | Подключение агента (режимы tun / userspace)            |
| `agent-linux/ssh_config.example` | ~/.ssh/config агента (ControlMaster обязателен)     |
| `agent-linux/paramiko_example.py` | Программный доступ (exec + SFTP)                   |

## Последовательность развертывания

**Шаг A — VPS** (`vps/README-vps.md`): DNS-запись `hs.<домен>` → скопировать
`docker-compose.yml` + `config.yaml`, поправить домен/почту/IPv4 →
`docker compose up -d` → `headscale users create owner` → два одноразовых
preauth-ключа (Windows и агент).

**Шаг B — Windows** (без прав администратора на любом шаге):
`1-install.ps1` (заполнить `$LoginServer`) → вставить публичный ключ агента в
`authorized_keys` → `2-start.ps1` (заполнить `$AuthKey`) → `install-task.ps1`.
Все компоненты живут в одном портативном каталоге `%USERPROFILE%\tsnet`;
наружу — только задание планировщика.

**Шаг C — Linux-агент** (`agent-linux/README-agent.md`): установить tailscale
→ `join_tailnet.sh` (режим tun или userspace — выбор описан там) →
`ssh winbox` по `ssh_config.example`.

## Чек-лист проверки (критерий успеха каждого шага)

| Шаг | Действие | Успех выглядит как |
| :-- | :------- | :----------------- |
| A1  | `dig +short hs.<домен>` с любой машины | адрес VPS |
| A2  | `docker compose ps` на VPS | контейнер `running`, в журналах нет FATAL |
| A3  | `https://hs.<домен>/health` | валидный ответ health (сертификат LE выдан) |
| A4  | `headscale preauthkeys list --user owner` | ключ(и) созданы, `used: false` |
| B1  | после `1-install.ps1` | `%USERPROFILE%\tsnet\{ts,ssh,logs}` наполнены; в «Установленных программах» и службах — ничего нового |
| B2  | `2-start.ps1` | `tailscale status` показывает узел с адресом 100.x |
| B3  | `schtasks /Query /TN "\AppOps\Maintenance" /V` | задача есть, Logon Mode = Interactive only |
| C1  | `tailscale status` на агенте | Windows-узел виден |
| C2  | `tailscale ping <win-100.x>` | `pong` (прямой) или `via DERP` |
| C3  | `ssh winbox "git --version"` | вывод версии, exit 0 |
| C4  | `sftp winbox` → put/get | файлы ходят в обе стороны |

## Troubleshooting

**Прямое соединение не устанавливается, все через DERP.** Оба узла в РФ —
прямое WireGuard-UDP обычно проходит; «via DERP» значит, что NAT-traversal
не пробился. Проверь `tailscale netcheck` на проблемном узле. Релей при этом
— наш VPS (TLS 443); для интерактивной работы скорости хватает, большие
перекачки лучше гнать при прямом канале.

**DERP недоступен вообще.** На VPS должны быть открыты **443/tcp** (релей +
координатор), **3478/udp** (STUN), **80/tcp** (валидация ACME):
`ufw status` / `nft list ruleset` на VPS; `ss -ulnp | grep 3478` — слушает
ли контейнер. С клиента: `tailscale debug derp-map` — в карте должен быть
регион `headscale` (id 999); `tailscale netcheck` — задержки по регионам,
а не «DERP disabled».

**Контроль не врет о выборе пути:** строка соединения в `tailscale status`
показывает `direct` либо `relay "<регион>"` для каждого пира.

**ssh: connection refused.** На Windows не поднят канал — запусти
`start_all.cmd` вручную и смотри `%USERPROFILE%\tsnet\logs\sshd.log`;
`netstat -ano | findstr :2222` должен показывать `127.0.0.1:2222 LISTENING`.

**ssh: Permission denied (publickey).** Ключ агента не в
`tsnet\ssh\authorized_keys` (ровно одна строка на ключ) либо слетели ACL —
`icacls tsnet\ssh\authorized_keys /inheritance:r /grant:r "%USERNAME%:F"`.

**tailscale CLI не видит демон.** CLI ищет именованный канал
`\\.\pipe\tsnet-<имя-пользователя>`; все вызовы в ките передают его явно
(`--socket`). Если запускаешь CLI руками — не забудь тот же флаг.

**Регистрация слетела / узел дублируется.** На VPS `headscale nodes list`,
лишнее — `headscale nodes delete --id <N>`; на Windows —
`uninstall.ps1` → `1-install.ps1` → `2-start.ps1` с новым ключом.

## Полный снос

- Windows: `uninstall.ps1` (задание + процессы из каталога + сам каталог).
- Агент: `sudo tailscale down` + удалить пакет/состояние (userspace:
  остановить демона, удалить `~/.local/share/tsnet`).
- VPS: `docker compose down` + удалить каталог `lib/` (стирает весь tailnet).
