#!/usr/bin/env bash
# =============================================================================
# join_tailnet.sh — подключение Linux-агента к tailnet (Headscale).
#
# Режимы (переменная MODE):
#   tun        — системный tailscaled (TUN-интерфейс); требует установленный
#                пакет tailscale и root для `tailscale up`. Рекомендован:
#                дальше ssh/rsync/sftp работают напрямую по 100.x.
#   userspace  — демон в пространстве пользователя БЕЗ root; сеть tailnet
#                доступна только через локальный SOCKS5-прокси (ssh — через
#                `nc -x`, paramiko — через PySocks, см. paramiko_example.py).
#
# Примеры:
#   MODE=tun       LOGIN_SERVER=https://hs.example.com AUTH_KEY=xxxx ./join_tailnet.sh
#   MODE=userspace LOGIN_SERVER=https://hs.example.com AUTH_KEY=xxxx ./join_tailnet.sh
# =============================================================================

set -euo pipefail

# ---------------------- ПАРАМЕТРЫ (можно переопределить окружением) ----------
LOGIN_SERVER="${LOGIN_SERVER:-https://hs.example.com}"   # адрес Headscale (VPS)
AUTH_KEY="${AUTH_KEY:-}"                                 # одноразовый preauth-ключ
MODE="${MODE:-tun}"                                      # tun | userspace
TS_STATE="${HOME}/.local/share/tsnet"                    # состояние userspace-режима
TS_SOCK="${TS_STATE}/tailscaled.sock"                    # сокет CLI userspace-режима
SOCKS_ADDR="127.0.0.1:1055"                              # SOCKS5 userspace-режима
WIN_HOST="${WIN_HOST:-}"                                 # 100.x Windows-машины (только для проверки)
WIN_PORT="${WIN_PORT:-2222}"
# ---------------------- /ПАРАМЕТРЫ -------------------------------------------

err() { printf 'ОШИБКА: %s\n' "$1" >&2; exit 1; }

[ -n "$AUTH_KEY" ] || err 'задай AUTH_KEY=<одноразовый preauth-ключ> (создать на VPS: docker compose exec headscale headscale preauthkeys create --user owner)'

command -v tailscale  >/dev/null 2>&1 || err 'tailscale не найден — установи пакет дистрибутива (см. README-agent.md, раздел 1)'
command -v tailscaled >/dev/null 2>&1 || err 'tailscaled не найден — установи пакет tailscale целиком'

join_args=(up --login-server "$LOGIN_SERVER" --authkey "$AUTH_KEY")

case "$MODE" in
    tun)
        # Системная служба уже должна работать (systemctl enable --now tailscaled).
        # --accept-dns=false: не трогать /etc/resolv.conf на этой машине.
        echo "Режим tun: системный tailscaled, регистрация на ${LOGIN_SERVER}"
        sudo tailscale "${join_args[@]}" --accept-dns=false
        tailscale status
        ;;
    userspace)
        echo "Режим userspace: демон без root, состояние в ${TS_STATE}"
        mkdir -p "$TS_STATE"
        if ! pgrep -f "tailscaled.*${TS_STATE}" >/dev/null 2>&1; then
            nohup tailscaled \
                --tun=userspace-networking \
                --socks5-server="$SOCKS_ADDR" \
                --statedir="$TS_STATE" \
                --socket="$TS_SOCK" >/dev/null 2>&1 &
            sleep 2
        fi
        tailscale --socket="$TS_SOCK" "${join_args[@]}"
        tailscale --socket="$TS_SOCK" status
        echo "SOCKS5-прокси tailnet: ${SOCKS_ADDR} (ssh -o ProxyCommand='nc -x ${SOCKS_ADDR} %h %p' ...)"
        ;;
    *)
        err "MODE=${MODE} не поддерживается; используй tun или userspace"
        ;;
esac

# Финальная проверка, если адрес Windows-машины передан.
if [ -n "$WIN_HOST" ]; then
    echo "--- проверка канала до ${WIN_HOST}:${WIN_PORT} ---"
    if [ "$MODE" = "tun" ]; then
        ssh -o BatchMode=yes -p "$WIN_PORT" "$WIN_HOST" true \
            && echo 'Канал SSH поднят (tun: прямое подключение).' \
            || echo 'SSH не прошел — смотри Troubleshooting в README-agent.md'
    else
        ssh -o BatchMode=yes -o "ProxyCommand=nc -x ${SOCKS_ADDR} %h %p" \
            -p "$WIN_PORT" "$WIN_HOST" true \
            && echo 'Канал SSH поднят (userspace: через SOCKS5).' \
            || echo 'SSH не прошел — смотри Troubleshooting в README-agent.md'
    fi
fi
