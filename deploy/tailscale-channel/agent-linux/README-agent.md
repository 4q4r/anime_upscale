# Linux-агент: подключение к Windows-машине через tailnet

Часть кита `deploy/tailscale-channel/`. Шаг **C (агент)** — твоя Linux-машина.
Прав администратора на Windows-стороне нет и не нужны; здесь — твоя машина,
решение о режиме принимаешь сам (см. ниже).

## 0. Предпосылки

- Headscale на VPS развернут (шаг A, `../vps/README-vps.md`);
- Windows-узел зарегистрирован (шаг B, `../README.md`); его 100.x-адрес
  известен (вывод `tailscale status` на Windows);
- создан одноразовый preauth-ключ для агента:
  `docker compose exec headscale headscale preauthkeys create --user owner`.

## 1. Установка tailscale (если еще нет)

Поставь пакетом дистрибутива (на Arch: `sudo pacman -S tailscale`, на
Debian/Ubuntu — официальный репозиторий pkgs.tailscale.com). После установки
на системах с systemd: `sudo systemctl enable --now tailscaled`.

## 2. Режим подключения — выбор владельца

**Обычный режим (TUN, рекомендован):** нужен root для `tailscaled` (системная
служба), зато дальше все работает «как обычно» — `ssh`, `rsync`, `sftp`
напрямую по 100.x-адресам. Используй `--accept-dns=false`, чтобы служба
не трогала `/etc/resolv.conf`:

```bash
sudo tailscale up --login-server https://hs.example.com --authkey <КЛЮЧ> --accept-dns=false
```

**Userspace-режим (без root вообще):** демон в пространстве пользователя,
наружу торчит SOCKS5-прокси; в сеть tailnet попадает только трафик через него
(ssh — через `ProxyCommand` с `nc -x`). Выбирай, если root недоступен:

```bash
tailscaled --tun=userspace-networking --socks5-server=127.0.0.1:1055 \
    --statedir="$HOME/.local/share/tsnet" --socket="$HOME/.local/share/tsnet/tailscaled.sock" &
tailscale --socket="$HOME/.local/share/tsnet/tailscaled.sock" \
    up --login-server https://hs.example.com --authkey <КЛЮЧ>
```

Либо просто запусти `./join_tailnet.sh` — он умеет оба режима
(`MODE=tun` по умолчанию, `MODE=userspace`) и сам проверяет переменные.

**Критерий успеха:** `tailscale status` показывает Windows-узел и его
100.x-адрес; `tailscale ping <windows-100.x>` отвечает (прямой `pong`
или `via DERP`).

## 3. Ключи и ssh-конфиг агента

```bash
ssh-keygen -t ed25519 -f ~/.ssh/agent_ed25519 -N ''
```

Публичный ключ (`~/.ssh/agent_ed25519.pub`) одной строкой вставь на
Windows-машине в `<корень>\ssh\authorized_keys` (если канала под рукой нет —
любым доступным способом: общий диск, буфер через RDP и т.п.).

Шаблон `~/.ssh/config` — в `ssh_config.example` (замени адрес и пользователя).
`ControlMaster auto` + `ControlPersist 10m` обязательны: sshd на Windows в
режиме `-d` обслуживает одно транспортное соединение, а мультиплексирование
пускает все exec/SFTP-каналы через него одним куском.

**Критерий успеха канала целиком:**

```bash
ssh winbox "git --version"      # exec-канал
sftp winbox                     # файловый канал: put/get любого файла
```

## 4. Программное использование

Скетч на paramiko — `paramiko_example.py` (exec + SFTP, оба режима сети).
Для userspace-режима дополнительно нужен PySocks (`uv run --with pysocks`).

## Troubleshooting (агент)

- **`tailscale ping` идет через DERP, медленно.** Оба узла в РФ — прямое
  WireGuard-UDP обычно устанавливается; «via DERP» = прямой канал не
  пробился (NAT/TSPU-особенность). Скорость релея упирается в VPS и TLS —
  для интерактивного ssh этого достаточно, для больших SFTP-перекачек —
  проверь `tailscale netcheck` и старайся держать прямой канал.
- **`ssh` висит / connection refused.** На Windows-машине поднят ли канал
  (`start_all.cmd` / задание), слушается ли порт: там же
  `netstat -ano | findstr :2222` должен показывать `127.0.0.1:2222 LISTENING`.
- **Permission denied (publickey).** Ключ не попал в `authorized_keys`
  (ровно одна строка, полный файл), либо слетели ACL на файле — на Windows:
  `icacls <корень>\ssh\authorized_keys /inheritance:r /grant:r "%USERNAME%:F"`.
- **После перезагрузки Windows узел офлайн.** Задание `\AppOps\Maintenance`
  не сработало — проверь `schtasks /Query /TN "\AppOps\Maintenance" /V` и
  журнал `%USERPROFILE%\tsnet\logs\sshd.log`; подними вручную `start_all.cmd`.
- **Узел «забыл» регистрацию.** Сноси состояние на Windows и повтори шаг B
  (см. uninstall/1-install), затем на VPS `headscale nodes list` — старую
  запись можно удалить: `headscale nodes delete --id <N>`.
