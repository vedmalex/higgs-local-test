# Issue #132 — черновик описи TTS-моделей на MLX (в процессе)

Статус: сбор фактов, не публиковать как есть.

## Источник 1: mlx-audio 0.5.0, `.venv-tts`, `mlx_audio/tts/models/`

Полный список подпапок (37 моделей, без base.py/interpolate.py/__init__.py):
arktts, bailingmm, bark, chatterbox, chatterbox_turbo, confucius4, dense, dia,
dramabox, echo_tts, fish_qwen3_omni, higgs_audio, higgs_audio_v3, indextts,
irodori_tts, kitten_tts, kokoro, kugelaudio, llama, longcat_audiodit, melotts,
moss_tts, moss_tts_delay, moss_tts_local, moss_tts_nano, omnivoice, outetts,
pocket_tts, qwen3, qwen3_tts, sesame, soprano, spark, tada, vibevoice, voxcpm,
voxcpm2, voxtral_tts, zonos2

**ВАЖНО: в mlx-audio реально есть папка `omnivoice`!** Владелец не ошибся с названием
(или ошибка совпала с реальным именем). Нужно детально разобрать.

## Рабочий журнал

ЗАВЕРШЕНО. Итоговый документ: docs/research/mlx-tts-survey-issue-132.md
Отписано в issue: https://github.com/vedmalex/higgs-local-test/issues/132#issuecomment-5411500048
Весов не качалось (только WebFetch на карточки HF — метаданные, не snapshot_download).
Сверено с notebooks/model_catalog.json — пересечений нет, статусы не менялись.
