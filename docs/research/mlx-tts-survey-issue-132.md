# Опись TTS-моделей на MLX (issue #132)

Снимок: 2026-08-25. Источники: `mlx-audio` 0.5.0 (`.venv-tts`, реализация на диске —
`mlx_audio/tts/models/*/model.py` и `README.md` рядом), карточки моделей на
Hugging Face (`mlx-community`, апстрим-репозитории), апстрим GitHub issues.
Веса не скачивались, тяжёлая генерация не запускалась — все выводы по коду и
документации. Там, где первичный источник не даёт ответа, честно написано
«не выяснено» — вместо того чтобы выдавать догадку за факт.

Машина: Apple M1, 16 ГБ unified memory. Модель, не влезающая в это в
принципе (по заявленному размеру весов), в таблицу всё равно включена —
но с пометкой «не проходит» и без глубокого разбора остальных возможностей
(проверять их дальше не имеет смысла, см. правило issue: «Проверяй это раньше»).

## Как собиралась опись

`mlx_audio/tts/models/` в установленном пакете содержит 37 реализаций моделей
(список получен `ls`). У каждой либо есть свой `README.md` с примерами
Python/CLI API, либо только код. Русский язык проверялся по разделу
«Languages»/README и, где он не был явным, — по карточке модели на HF
(`WebFetch`) или через `WebSearch` по апстрим-репозиторию. Модели без
задокументированного русского в опись не включались вовсе (правило issue —
«Модель без русского не рассматривай вовсе»); из 37 реализаций русский
подтверждён для: Higgs Audio v3, Qwen3-TTS (все 3 чекпоинта), **OmniVoice**,
VoxCPM2, ZONOS2, MOSS-TTS, KugelAudio. Остальные 30 — английский/китайский/
ограниченный набор языков без ru (arktts, confucius4, longcat_audiodit, tada,
voxtral_tts, dia, kitten_tts, kokoro, sesame, spark, bark, chatterbox,
chatterbox_turbo, dense, bailingmm, dramabox, echo_tts, irodori_tts, llama,
melotts, outetts, pocket_tts, soprano, indextts, fish_qwen3_omni,
moss_tts_delay/_local/_nano — не проверялись отдельно, семейство MOSS уже
покрыто через `moss_tts`, higgs_audio v1/v2 — тот же язык что и v3, но не
приоритет: проект уже на v3).

## Опись

Пометки: **есть** / **нет** / *не выяснено*. Столбец «16 ГБ» — проходит ли
модель ограничение по памяти на заявленном размере весов (bf16/fp16, если не
указана более лёгкая квантованная версия на `mlx-community`).

| Модель | Размер (вес) | 16 ГБ | Лицензия | Русский | Клонирование (+батч) | Дизайн голоса (закрепление?) | Ударения RU | DSL разметки |
|---|---|---|---|---|---|---|---|---|
| **Higgs Audio v3** (`bosonai/higgs-tts-3-4b`) | 4B | ✅ | research/non-commercial (Boson) | есть, эмпирически | **есть**, батч совместим | **нет** | **есть** (апостроф после гласной) | **есть**, 43 тега |
| **Qwen3-TTS Base** (0.6B/1.7B) | 0.6–1.7B | ✅ | Apache-2.0 | есть (карточка) | **есть** (`generate_voice_clone`) | нет | ломается (артефакты) | *не выяснено* |
| **Qwen3-TTS CustomVoice** (0.6B/1.7B) | 0.6–1.7B | ✅ | Apache-2.0 | есть | нет | **есть выбор** (9 тембров + `instruct=`); закрепление между вызовами *не выяснено* | ломается | *не выяснено* |
| **Qwen3-TTS VoiceDesign** (1.7B) | 1.7B | ✅ | Apache-2.0 | есть | нет | **есть выбор** (описанием); закрепление между вызовами — по внешним данным НЕТ (сэмплирует заново) | ломается | *не выяснено* |
| **OmniVoice** (`mlx-community/OmniVoice-bfloat16`, based on `k2-fsa/OmniVoice`) | 0.6B backbone, ~2.0 ГБ вес | ✅ | Apache-2.0 | есть, явно (`ru` в списке языков) | **есть**, и батч на уровне API (`generate_batch`, свой `ref_tokens` на элемент) | **есть выбор** (`instruct=` текстом: пол/возраст/питч/стиль/акцент), но обучен только на zh/en — для ru нестабильно по апстриму; закрепление между вызовами *не выяснено* | **нет** — открытые issue в апстриме просят поддержку акута U+0301, пока нет способа управлять ударением | небольшой: 13 невербальных тегов (`[laughter]`, `[sigh]`, `[question-*]`, `[surprise-*]`, `[dissatisfaction-hnn]`) + один `instruct`-тег стиля; пауз/просодии нет |
| **VoxCPM2** (`mlx-community/VoxCPM2-*`, based on `openbmb/VoxCPM2`) | 2B, 2.3–5.0 ГБ (4bit/8bit/bf16) | ✅ | Apache-2.0 | есть, явно (в списке 30 языков) | **есть** (`ref_audio=`) | **есть выбор** (`instruct=` текстом), НО апстрим-карточка прямо предупреждает: «Voice Design and Style Control results may vary between runs» — закрепление не гарантировано | *не выяснено* | нет тегов эмоций/пауз в реализации; только `instruct` для дизайна и `prompt_text`/`prompt_audio` для continuation |
| **ZONOS2** (`mlx-community/Zyphra-ZONOS2`) | ~15.4 ГБ (bf16, квантованных версий на mlx-community не найдено) | ⚠️ впритык/не проходит на 16 ГБ (веса сами почти всю память) | Apache-2.0 | есть (карточка, Tier 2) | **есть**, с явным батчем на элемент (`ref_audios=`, `speaker_embeddings=`) | **нет** (только `speaker_embedding` из аудио, текстового описания голоса нет) | *не выяснено* | *не выяснено* (в README нет тегов; не проверялось глубже — модель не проходит по памяти) |
| **MOSS-TTS v1.5** (`OpenMOSS-Team/MOSS-TTS-v1.5`) | 8B bf16, квантованных весов на mlx-community не найдено | ⚠️ не проходит на 16 ГБ | Apache-2.0 | есть (карточка, 20→31 язык, включая ru) | есть (`ref_audio=`), но **батч explicitly НЕ реализован** в MLX-порте | нет описания текстом (только клонирование); апстрим упоминает «voice/character design» на GitHub, но это не видно в mlx-audio README/API | *не выяснено* | **есть**: `[pause 3.2s]` — маркер паузы проходит в промпт как есть; упоминаются Pinyin/IPA-коррекция произношения |
| KugelAudio (`kugelaudio-0-open`) | 7B, README прямо пишет «~17 ГБ unified memory» | **нет**, явно заявлено в README | MIT | есть (карточка, 24 языка) | нет — README: «Pre-encoded voice presets are not yet available... generates with a default voice» | не выяснено (не проверялось — отсеяно по памяти) | не выяснено | не выяснено |

Цитаты/пути к находкам:

- OmniVoice батч + клонирование по элементу: `.venv-tts/lib/python3.12/site-packages/mlx_audio/tts/models/omnivoice/omnivoice.py:293` (`def generate_batch`), логика per-item `ref_tokens_list[i] = create_voice_clone_prompt(...)` в теле того же метода.
- OmniVoice невербальные теги: `omnivoice.py:14-17` (`_NONVERBAL_PATTERN`), `omnivoice.py:124-131` (`_tokenize_with_nonverbal_tags`).
- OmniVoice `instruct` как тег дизайна: `omnivoice.py:181-192` (`_tokenize_style_and_text`, `<|instruct_start|>...<|instruct_end|>`), сигнатуры `generate`/`generate_batch` на `omnivoice.py:293` и `:483`.
- OmniVoice: обучен на дизайн голоса только для zh/en, клонирование — «most stable mode»: GitHub `k2-fsa/OmniVoice` README (WebFetch, 2026-08-25).
- OmniVoice: ударения RU не поддерживаются — открытые issues в апстриме: `k2-fsa/OmniVoice#65` («Support Unicode stress marks (U+0301) for Russian text»), `#52` («Russian language: stress control»), `#37` («Indicate stress in words»).
- OmniVoice лицензия и поддерживаемые языки: `mlx_audio/tts/models/omnivoice/README.md` (в дистрибутиве `mlx-audio`), карточка `mlx-community/OmniVoice-bfloat16` (Apache-2.0, 2.04 ГБ).
- VoxCPM2 voice design + предупреждение о разбросе: README `mlx_audio/tts/models/voxcpm2/README.md` (`## Voice Design`, параметр `instruct`), карточка `openbmb/VoxCPM2` (WebFetch): «Voice Design and Style Control results may vary between runs; generating 1–3 times is recommended».
- VoxCPM2 нет батча: в `mlx_audio/tts/models/voxcpm2/voxcpm2.py` метода `generate_batch`/`batch_generate` нет (проверено `grep`).
- ZONOS2 батч + клонирование: `mlx_audio/tts/models/zonos2/README.md`, разделы `## Voice Cloning` и `## Batch Generation` (`ref_audios=`, `speaker_embeddings=` — «per-row speaker conditioning»).
- ZONOS2 размер: карточка `mlx-community/Zyphra-ZONOS2` (WebFetch) — 4 файла `.safetensors`, суммарно ≈15.4 ГБ, только bf16.
- MOSS-TTS батч не реализован: `mlx_audio/tts/models/moss_tts/moss_tts.py:746,927,1079` — `raise NotImplementedError("MOSS-TTS batch generation is not implemented.")`.
- MOSS-TTS пауза-тег: `mlx_audio/tts/models/moss_tts/README.md` — «Inline pause markers such as `[pause 3.2s]` are passed through to the v1.5 prompt.»
- KugelAudio память и отсутствие клонирования: `mlx_audio/tts/models/kugelaudio/README.md` — «Requires approximately 17GB of unified memory... Tested on M4 Max 36GB» и «Pre-encoded voice presets are not yet available in the upstream model; the model generates with a default voice.»
- Higgs Audio v3 и Qwen3-TTS — как в постановке issue и `docs/research/qwen3-tts-notes.md`, `docs/research/audiobook/qwen-voicedesign-for-higgs-reference.md`, `docs/guides/tag_reference.md`; лицензия Higgs — `docs/research/higgs-current-apis.md:108-112` (research/non-commercial, Boson).

## Про OmniVoice отдельно

Модель с таким названием **действительно существует** и реализована в
`mlx-audio` 0.5.0: `mlx_audio/tts/models/omnivoice/` (684 строки в
`omnivoice.py`), апстрим — `k2-fsa/OmniVoice`, MLX-чекпоинт —
`mlx-community/OmniVoice-bfloat16` (2.04 ГБ, Apache-2.0). Это не путаница с
похожим именем — сама модель называется именно так, и она портирована на MLX,
а не только существует как обещание в PyTorch. Архитектура нетривиальная:
двунаправленный Qwen3-0.6B бэкбон + итеративное маскированное диффузионное
декодирование (не авторегрессивная генерация, в отличие от Higgs/Qwen3-TTS) с
акустическим токенайзером HiggsAudioV2.

По четырём пунктам:

1. **Клонирование** — есть, `ref_audio=` (до 10с эталона), совместимо с батчем
   на уровне `generate_batch()`, где у каждого элемента батча свой независимый
   `ref_tokens` (в отличие от Higgs, где эталон общий на весь батч — здесь
   допускается разный диктор на каждую строку батча, что даже гибче).
2. **Дизайн голоса** — есть параметр `instruct=` (текстовое описание: пол,
   возраст, питч, стиль, акцент), но апстрим прямо пишет, что дизайн-режим
   обучен только на китайских и английских данных — для русского текста
   ожидаемо нестабилен. Закрепляет ли одно и то же `instruct` один и тот же
   голос между независимыми вызовами на разном тексте — **не выяснено**,
   в документации об этом ничего нет, а генерация не запускалась (правило
   issue — не гонять тяжёлую генерацию). Учитывая нестандартную
   (недиффузионно-авторегрессивную) архитектуру, аналогия с «AR-модель сужает
   область, но не фиксирует точку» на OmniVoice механически не переносится —
   нужен отдельный эмпирический тест, если модель когда-нибудь понадобится
   всерьёз.
3. **Ударения RU** — нет. В апстрим-репозитории есть минимум три открытых
   issue именно об этом (#37, #52, #65), последний прямо просит поддержку
   Unicode-акута U+0301 после гласной — то есть той же нотации, что уже
   работает в Higgs. Раз issue открыт, значит на момент снимка (2026-08-25)
   такой поддержки в апстриме нет.
4. **DSL** — минимальный: 13 невербальных тегов вида `[laughter]`, `[sigh]`,
   `[question-en]`/`[question-ah]`/... , `[surprise-*]`, `[dissatisfaction-hnn]`
   плюс один тег `instruct` для дизайна голоса. Тегов паузы, громкости,
   темпа, явной эмоции (кроме нескольких зашитых межличностных реакций) нет —
   ничего похожего на 43 тега Higgs.

Гипотеза о том, с чем могли спутать OmniVoice, оказалась не нужна — модель не
вымышленная и не опечатка, но стоит явно предупредить владельца: OmniVoice —
это k2-fsa/OmniVoice, а не что-то похожее по названию вроде "Voicebox",
"VoiceCraft" или "OmniAudio" (эти существуют в экосистеме TTS отдельно и с
OmniVoice не связаны; не проверялись подробно, так как владелец не просил).

## Прямые ответы

**Есть ли модель, закрывающая все четыре пункта сразу?** Нет. Ближе всех —
**OmniVoice**: клонирование с батчем есть, дизайн голоса текстом есть (пусть
и с оговоркой по качеству для ru), но ударения не поддерживаются вовсе (три
открытых апстрим-issue это подтверждают) и DSL почти нет. Higgs Audio v3
закрывает ударения и DSL, но не дизайн голоса. Qwen3-TTS CustomVoice/
VoiceDesign закрывают выбор голоса, но ломают ударения и не клонируют.
Ни одна из семи проверенных с русским моделей не закрывает все четыре разом.

**Если нет — какая пара закрывает их вместе, и стоит ли она усложнения?**
Уже рассматриваемая связка **Qwen3-TTS VoiceDesign/CustomVoice → Higgs Audio
v3** (первая рождает дизайн-эталон, вторая клонирует его и озвучивает с
разметкой сентимента) остаётся лучшей из просмотренных: она даёт все четыре
пункта одновременно (выбор голоса через Qwen, закрепление + ударения + DSL
через Higgs-клонирование), ценой одного дополнительного прохода перед
основной генерацией. Единственная найденная альтернативная пара —
**OmniVoice (дизайн) → Higgs Audio v3 (клонирование + DSL)** — устроена так
же (сначала дизайн голоса, потом клонирование с разметкой), но не даёт
преимуществ перед связкой Qwen→Higgs и добавляет риск: дизайн-режим OmniVoice
не обучен на русском тексте (только zh/en), тогда как VoiceDesign у Qwen
заявлен для 10 языков включая ru. Дополнительной пользы от OmniVoice в этой
роли не просматривается — усложнение того не стоит.

**Что нашлось про OmniVoice, коротко.** Реальная модель, реализована в
`mlx-audio`, MLX-чекпоинт есть, Apache-2.0, 2 ГБ, русский заявлен. Закрывает
клонирование (с батчем per-item) и текстовый дизайн голоса, но ударения не
поддерживает (три открытых issue в апстриме) и разметки почти нет.
Для роли «Qwen порождает эталон» может рассматриваться как второй кандидат
после Qwen3-TTS VoiceDesign — если понадобится второе мнение или Qwen
почему-то не подойдёт — но по документированному охвату языков дизайн-режима
Qwen для русского выглядит надёжнее.

## Что предлагается выкачать (только предложение, не выполнено)

Правило владельца — веса только через `notebooks/model_prefetch_to_drive.ipynb`
и Colab. Ничего не скачивалось. Если стоит попробовать эмпирически, кандидаты
по приоритету:

1. `mlx-community/OmniVoice-bfloat16` (2 ГБ) — маленький, легко проверить
   клонирование+батч и качество RU без DSL/ударений.
2. `mlx-community/VoxCPM2-4bit` (2.3 ГБ) — единственный из новых с явным
   voice-design ПЛЮС клонированием в одной модели; сразу можно проверить,
   насколько «плывёт» голос между вызовами при одинаковом `instruct`.

ZONOS2, MOSS-TTS и KugelAudio выкачивать не имеет смысла — не проходят по
памяти на M1 16 ГБ на заявленных (единственно доступных) весах.

**Сверка с `notebooks/model_catalog.json`** (23 позиции на момент снимка):
пересечений с моделями из этой описи нет — ни OmniVoice, ни VoxCPM2, ни ZONOS2,
ни MOSS-TTS, ни KugelAudio в каталоге не значатся, никакой существующий статус
`rejected`/`candidate` эта опись не оспаривает. Обе рекомендации выше — это
предложение добавить новые записи, а не менять чужие; сам файл не редактировался
(правило координатора — во избежание столкновения правок с другим агентом).
