#!/usr/bin/env bash
# Docker smoke test for the anime-upscale image (docs/docker.md).
#
# Sequence: build → compose config → CLI help → models verify (sha256) →
# GPU doctor (nvidia-smi -L) → synthetic 5 s clip through the pipeline →
# assert the output landed in the mounted output/ → container layer hygiene
# (docker diff: no writes outside the bind mounts; /tmp scratch tolerated).
#
# GPU-dependent steps fail loudly on a GPU-less host. Set SMOKE_SKIP_GPU=1 to
# skip them explicitly (the skip is always printed — never silent).
#
# Runs from the repo root (resolved from the script location); leaves no
# artifacts: input/output smoke files and the named container are removed in
# the EXIT trap even after a failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SERVICE="anime-pipeline"
SMOKE_NAME="anime-upscale-smoke"
SMOKE_SRC="smoke_src.mkv"
COMPOSE="docker compose"

log() { printf '%s\n' "$*"; }
ok()  { log "[OK]   $*"; }
fail() { log "[FAIL] $*"; exit 1; }

cleanup() {
  docker rm -f "$SMOKE_NAME" >/dev/null 2>&1 || true
  rm -f "input/$SMOKE_SRC" "output/${SMOKE_SRC%.mkv}.mkv"
}
trap cleanup EXIT

log "=== Смоук Docker-образа anime-upscale ==="
log "Репозиторий: $REPO_ROOT"

# ---- Шаг 1/8: сборка -------------------------------------------------------
log "[ШАГ 1/8] Сборка образа ($COMPOSE build)…"
if $COMPOSE build; then
  ok "образ собран"
else
  fail "сборка образа не удалась (см. вывод выше)"
fi

# ---- Шаг 2/8: валидность compose-конфига -----------------------------------
log "[ШАГ 2/8] Проверка compose-конфига…"
if $COMPOSE config -q; then
  ok "compose.yaml валиден"
else
  fail "compose.yaml невалиден"
fi

# ---- Шаг 3/8: CLI отвечает внутри контейнера -------------------------------
log "[ШАГ 3/8] CLI внутри образа (--help)…"
if $COMPOSE run --rm "$SERVICE" --help >/dev/null 2>&1; then
  ok "CLI отвечает"
else
  fail "CLI не отвечает внутри контейнера"
fi

# ---- Шаг 4/8: веса моделей, sha256-пины ------------------------------------
log "[ШАГ 4/8] Проверка весов (models verify)…"
if $COMPOSE run --rm "$SERVICE" models verify; then
  ok "sha256-пины моделей сошлись"
else
  fail "models verify не прошёл — веса в образе повреждены"
fi

# ---- Шаг 5/8: GPU-доктор ----------------------------------------------------
log "[ШАГ 5/8] GPU-доктор (nvidia-smi -L)…"
if [ "${SMOKE_SKIP_GPU:-0}" = "1" ]; then
  log "[SKIP]  GPU-шаги пропущены явно (SMOKE_SKIP_GPU=1)"
else
  if gpu_list="$($COMPOSE run --rm --entrypoint nvidia-smi "$SERVICE" -L 2>/dev/null)" && [ -n "$gpu_list" ]; then
    ok "GPU виден из контейнера:"
    log "$gpu_list"
  else
    fail "GPU недоступен из контейнера. Чините хост: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker (подробности: docs/docker.md). Либо перезапустите с SMOKE_SKIP_GPU=1."
  fi
fi

# ---- Шаг 6/8: синтетический клип во входную маунт-папку ---------------------
if [ "${SMOKE_SKIP_GPU:-0}" = "1" ]; then
  log "[SKIP]  шаги 6–8 (клип, прогон, результат, diff) пропущены явно (SMOKE_SKIP_GPU=1)"
else
  log "[ШАГ 6/8] Генерация синтетического клипа 5 с (внутри контейнера)…"
  if $COMPOSE run --rm --entrypoint ffmpeg "$SERVICE" \
      -hide_banner -loglevel error \
      -f lavfi -i "testsrc=duration=5:size=320x240:rate=24" \
      -c:v libx264 -preset ultrafast -pix_fmt yuv420p -y "/app/input/$SMOKE_SRC"; then
    ok "клип создан: input/$SMOKE_SRC"
  else
    fail "ffmpeg внутри контейнера не сработал"
  fi

# ---- Шаг 7/8: сквозной прогон пайплайна ------------------------------------
  log "[ШАГ 7/8] Сквозной прогон (interp-only, контейнер без --rm для docker diff)…"
  if $COMPOSE run --name "$SMOKE_NAME" "$SERVICE" --profile interp-only -i "$SMOKE_SRC"; then
    ok "пайплайн завершился с нулевым кодом"
  else
    fail "пайплайн упал (см. вывод выше)"
  fi
fi

# ---- Шаг 8/8: выходной файл в маунте + гигиена слоя -------------------------
if [ "${SMOKE_SKIP_GPU:-0}" = "1" ]; then
  log "[SKIP]  Шаг 8/8: assert выходного файла пропущен явно (SMOKE_SKIP_GPU=1)"
  log "[SKIP]  Шаг 8/8: docker diff пропущен явно (SMOKE_SKIP_GPU=1)"
  log ""
  log "=== СМОУК ЗАВЕРШЁН — SKIPPED (GPU-часть 5–8 пропущена явно; полный прогон на GPU-хосте) ==="
  exit 0
fi

log "[ШАГ 8/8] Проверка результата и гигиены слоя…"
if [ -f "output/$SMOKE_SRC" ]; then
  ok "выходной файл появился на хосте: output/$SMOKE_SRC"
else
  fail "выходного файла нет в output/ — маунт-контракт нарушен"
fi

# docker diff shows every path changed in the container layer. Allowed:
# the bind mounts (/app/input, /app/output, /app/config.toml), $HOME scratch
# (bestsource index cache lives in ~/bsindex by design), /tmp scratch, and
# the bare mount-parent directory nodes (/app, /home) whose inodes are
# touched by mount activity. Anything else pollutes the image layer — fail.
# docker diff itself must succeed (loud failure: the smoke container vanished);
# only the allowlist filtering is non-failing (awk exits 0 on empty input).
if ! diff_output="$(docker diff "$SMOKE_NAME")"; then
  fail "docker diff не выполнился — контейнер смока $SMOKE_NAME не найден"
fi
dirty="$(printf '%s\n' "$diff_output" | awk '
  { path = $2 }
  path ~ /^\/app\/input/   { next }
  path ~ /^\/app\/output/  { next }
  path == "/app/config.toml" || path == "/app" { next }
  path ~ /^\/home\/app/    { next }
  path == "/home"          { next }
  path ~ /^\/tmp/          { next }
  { print path }
')"
if [ -z "$dirty" ]; then
  ok "docker diff чист вне маунтов (кроме /tmp-скретча)"
else
  log "$dirty"
  fail "контейнерный слой загрязнён вне маунтов — см. список выше"
fi

log ""
log "=== СМОУК ПРОЙДЕН ✓ ==="
