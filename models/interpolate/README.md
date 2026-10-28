# models/interpolate/ — веса интерполяции RIFE

Каталог пуст by design: веса RIFE 4.25 живут в кэше обёртки **vsrife 5.7.0** —
`<site-packages>/vsrife/models/flownet_v4.25.pkl`. Файл кладётся заранее
командой `anime-pipeline models fetch` (запись `[models.rife_v4_25]` в
`models/manifest.toml`, sha256-пин) и проверяется перед каждым батчем со
стадией `interpolate`; собственный auto-download обёртки с github.com остаётся
лишь аварийным путём — на сетях с перехватом TLS он ненадёжен.

Кэш TensorRT-движков (сборка двигателя под конкретный GPU/размер кадра) живёт
в каталоге `[stages.interpolate] trt_cache_dir` (по умолчанию `~/.cache/vsrife`
на Linux, `%USERPROFILE%\.cache\vsrife` на Windows, если не переопределён).

Ничего из этого не коммитится — провенанс зафиксирован записью
`models/manifest.toml` (`[models.rife_v4_25]`).
