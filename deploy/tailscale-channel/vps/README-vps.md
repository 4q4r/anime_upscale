# VPS: развёртывание Headscale (координатор tailnet + DERP-релей)

Часть кита `deploy/tailscale-channel/`. Общая последовательность — в
[../README.md](../README.md). Этот файл — шаг **A (VPS)**.

## Выбор схемы TLS (документированное решение)

Выбран **ACME на самом headscale** (встроенный клиент Let's Encrypt,
`tls_letsencrypt_hostname`) — на одну движущуюся часть меньше: не нужен
реверс-прокси, headscale сам терминирует TLS на `listen_addr` (0.0.0.0:443).
Условие: порты **80/tcp** (валидация HTTP-01), **443/tcp** (control + DERP)
и **3478/udp** (STUN) свободны на VPS.

Если на этом же VPS уже стоит nginx/caddy, занявший 443, — вариант
«реверс-прокси спереди»: в `config.yaml` выключи ACME
(`tls_letsencrypt_hostname: ""`), верни `listen_addr: 127.0.0.1:8080`,
убери публикацию портов 80/443 у контейнера и поставь перед ним прокси.
Для Caddy достаточно:

```
hs.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

(сертификат LE тогда получает Caddy; `server_url` в config.yaml не меняется —
это внешний адрес клиентов). Не используй оба способа одновременно.

## Порядок развёртывания

### 1. DNS

Создай на своем DNS-провайдере запись для поддомена:

| Тип  | Имя              | Значение            |
| :--- | :--------------- | :------------------ |
| A    | `hs.example.com` | публичный IPv4 VPS  |
| AAAA | `hs.example.com` | IPv6 VPS (если есть) |

**Критерий успеха:** с локальной машины `dig +short hs.example.com`
возвращает адрес VPS.

### 2. Файлы и права

Положи на VPS файлы этого каталога (`docker-compose.yml`, `config.yaml`) в
одну папку, например `/opt/headscale/`, и поправь в `config.yaml`:

- `server_url: https://hs.<твой-домен>` (обязательно https — без него не
  включится встроенный DERP);
- `acme_email` — твоя почта;
- `tls_letsencrypt_hostname: hs.<твой-домен>`;
- `derp.server.ipv4` — публичный IPv4 VPS.

Создай пустые каталоги состояния: `mkdir -p lib run`.

### 3. Запуск

```bash
cd /opt/headscale
docker compose up -d
docker compose logs --tail 50 headscale
```

**Критерий успеха:** в журнале — строка о старте serve без `FATAL`/`error`;
`docker compose ps` показывает `running`. Регистрация ACME-сертификата видна
в тех же журналах при первом запросе от клиента.

Проверка конфигурации (без запуска, полезно после правок):

```bash
docker compose run --rm headscale configtest
```

### 4. Пользователь tailnet

```bash
docker compose exec headscale headscale users create owner
```

**Критерий успеха:** вывод `User created` (или `already exists` при повторе).

### 5. Preauth-ключ для каждого узла

Windows-машина:

```bash
docker compose exec headscale headscale preauthkeys create --user owner
```

Linux-агент (отдельный, одноразовый):

```bash
docker compose exec headscale headscale preauthkeys create --user owner
```

Ключ по умолчанию **одноразовый и живет 1 час** — создавай прямо перед
подключением узла. Для пакетной регистрации нескольких узлов за один раз:
`--reusable --expiration 24h`.

Список активных ключей: `... preauthkeys list --user owner`.
Список узлов: `... nodes list`.

**Критерий успеха:** `preauthkeys list` показывает ключ с `used: false`.

## Проверка после подключения клиентов

- `https://hs.<домен>/health` из браузера — `{ "status": "pass" }`-ответ;
- с любого клиента: `tailscale status` — узлы видят друг друга;
- `tailscale ping <100.x-адрес-второго-узла>` — в выводе видно
  `via DERP(headscale)` либо прямой `pong` (см. troubleshooting ниже).

## Troubleshooting (VPS)

**Клиент не регистрируется.** Проверь `docker compose logs headscale`;
частая причина — `server_url` не совпадает с реальным именем/схемой или
сертификат не выпустился (порт 80 закрыт, см. ниже).

**ACME не выдает сертификат.** HTTP-01 требует, чтобы Let's Encrypt достучался
до 80/tcp этого адреса. Проверь файрвол VPS:
`ufw status` / `nft list ruleset` — наружу должны быть открыты
**80/tcp, 443/tcp, 3478/udp**. Проверь, что порт 80 не занят другим сервисом:
`ss -tlnp | grep -E ':80|:443'`.

**DERP недоступен / клиенты не могут соединиться через релей.**
- STUN: `ss -ulnp | grep 3478` — слушает контейнер;
  снаружи: `nc -u -v <vps> 3478` (пакет уходит — порт открыт).
- С клиента: `tailscale debug derp-map` — в карте должен быть регион
  `headscale` (999); `tailscale netcheck` — строка про DERP-регионы с
  задержками, а не `DERP disabled`.
- Убедись, что `server_url` — https и сертификат валиден:
  `curl -v https://hs.<домен>/health`.

**Сертификат истек / ротация.** headscale продлевает ACME сам; кэш —
`lib/`. Если сломал домен — исправь `tls_letsencrypt_hostname` и
перезапусти `docker compose restart headscale`.

**Полный снос** (осторожно — стирает tailnet): `docker compose down` +
удалить каталог `lib/`.
