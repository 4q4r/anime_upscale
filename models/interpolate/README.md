# models/interpolate/ — веса интерполяции RIFE

Каталог пуст by design: веса RIFE 4.25 управляются обёрткой **vsrife 5.7.0**
(модель скачивается автоматически при первом запуске стадии `interpolate`
из релизов HolyWu/vs-rife; параметр `auto_download=True` в коде стадии).

Кэш TensorRT-движков (сборка двигателя под конкретный GPU/размер кадра) живёт
в каталоге `[stages.interpolate] trt_cache_dir` (по умолчанию `~/.cache/vsrife`
на Linux, `%USERPROFILE%\.cache\vsrife` на Windows, если не переопределён).

Ничего из этого не закрепляется в манифесте и не коммитится — провенанс
зафиксирован документационной записью `models/manifest.toml` (`[models.rife_4_25]`).
