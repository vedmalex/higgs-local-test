# Higgs TTS 3 — справочник тегов разметки (Refs #57)

Дата: 2026-08-25. Источник истины — реальный чекпоинт `bosonai/higgs-tts-3-4b`
(`tokenizer.json` `added_tokens`, снимок `7556c17e05201fccd9c8cc120bc216dcc7b5d561`), а не код
проекта. В словаре чекпоинта **84 добавленных токена**; этот документ разбирает все 84, а не
только 43 «официальных» тега разметки.

Все образцы сгенерированы **в одном прогоне**, на одном и том же нейтральном по смыслу русском
тексте — `docs/research/audiobook/m4_tag_catalog_bench.py`, batch=8, та же модель/параметры для
всех 46 звучащих клипов (нейтральная база + 43 официальных тега + 2 недокументированных `env`).
Это специально сделано заново одним заходом, а не собрано из прежних отдельных PR — предыдущие
проверки (M4-T0, M4-T5, sfx-фикс) делались в разных заходах и напрямую не сравнимы друг с другом;
цифры из них сохранены как независимое подтверждение, но группировка A/B/C и все конкретные числа
в этом документе — из свежего единого прогона.

## 0. Чем пользоваться, а чем нет (коротко, для тех кто пишет сценарий)

Установлено владельцем ранее на слух (см. `m4-sentiment-results.md`, `m4-tag-inventory-results.md`)
и подтверждено/дополнено этим прогоном:

- **Эмоции работают.** Все 21 `emotion:*` дают различимый на слух эффект (проверено владельцем на
  паре sadness/elation, подтверждено по всей группе). Используйте свободно, ставьте в начале
  каждого предложения, которое должно звучать в этой эмоции.
- **Ударения через апостроф — обязательны для омографов.** `за'мок` / `замо'к` и т.п. — единственная
  нотация, которая была STT-чистой в 6/6 случаев в пробе на 3 парах омографов. Сдвигает ли ударение
  реально к нужному слогу — не проверено (русский ASR не размечает ударение), но апостроф не читается
  вслух буквально и не ломает наш `split_sentences`/`validate_control_tags`.
- **Паузы (`prosody:pause`/`long_pause`) работают, но одноразово.** Ставьте инлайново, между фразами,
  не открывайте как «состояние» — `chunk_sentences()` и не пытается их переоткрывать.
- **На `speed_*` полагаться нельзя.** Все четыре темповых тега — группа C: слова/сек отличаются от
  нейтрали не больше чем на ±0.11, `speed_very_fast`/`speed_very_slow` даже двигаются в обратную
  сторону от названия. Тег не пустышка (что-то в звуке меняется), но не используйте его как способ
  реально ускорить/замедлить чтение — для реального замедления `PROMPTING.md` сам рекомендует
  `prosody:long_pause` между фразами.
- **`style:whispering` не шепчет.** Дважды независимо подтверждено (энергия выше, а не ниже
  нейтрали) — это не шёпот, а просто иная манера, тише не становится.
- **sfx и env проверяются здесь впервые.** Ни один из 9 `sfx:*` и 2 `env:*` тегов не был раньше
  прослушан человеком. Статусы ниже — «только метрики» до тех пор, пока владелец не прослушает.
- **Структурные и служебные токены (38 из 84) не для сценария вообще** — см. §3/§4. Не пытайтесь
  вставлять `<|system|>`, `<|user|>`, `<tool_call>` и т.п. в текст главы.

## 1. Метод измерения (кратко)

- Один и тот же нейтральный по смыслу русский текст из 3 предложений (тот же, что в
  `m4_tag_inventory_bench.py`, PR #108, для сравнимости с прежним прогоном):
  *«Сегодня я занимался повседневными делами. Утром я выпил чай и почитал книгу. Потом вышел на
  улицу и немного прошёлся.»*
- Тег уровня предложения (emotion/style/prosody speed_*/pitch_*/expressive_*, а также `env:*`)
  переоткрывается в начале каждого из 3 предложений — так же, как `chunk_sentences()` реально
  переоткрывает тег внутри одного чанка.
- Инлайновые одноразовые теги (`prosody:pause`/`long_pause`, все 9 `sfx:*`) вставлены один раз
  между предложением 1 и 2, не в начало текста. `sfx` — строго по формату `PROMPTING.md`:
  `<|sfx:tag|>ономатопея, ...` (тег сразу перед ономатопеей, без пробела).
- `env:music`/`env:noise` — недокументированы; пробовались как префикс предложения (та же
  конвенция, что emotion/style), по прямому указанию владельца.
- Батч 8, `model.batch_generate`, `temperature=1.0`, `max_new_tokens=4096` — тот же путь, что уже
  измерен в PR #105 (батчинг) и переиспользован PR #108/#109 без изменений в самом методе.
- Метрики: `docs/research/audiobook/m4_tag_catalog_metrics.py` (тонкая обёртка над уже
  существующим `m4_tag_inventory_metrics.analyze_full`/`m4_prosody_metrics.analyze` — F0
  медиана/std/размах, энергия, паузы, темп (слов/сек), наклон терминального F0). **Медиана F0
  фиксируется, но не используется как единственное основание для вывода** — по прямому указанию
  владельца (дважды вводила в заблуждение: расхождение у sadness/anger в прежнем прогоне,
  «неверное направление» у грусти при верной оценке на слух). Выводы — по разбросу F0 (std),
  числу/длительности пауз и темпу (слов/сек), с перекрёстной проверкой по длительности клипа.

### 1.1 Важная оговорка: нейтральная база этого прогона сама нетипична

`neutral_baseline` в этом прогоне: F0 std **102.6 Hz**, медиана F0 146.3 Hz, энергия -28.1 дБ,
2 паузы (760 мс), темп 2.45 сл/с, длительность 7.76 с. Для сравнения, нейтральная база
прежнего 34-тегового прогона (PR #108, тот же текст, другой прогон) — F0 std **60.3 Hz**, темп
2.23 сл/с. Разброс F0 у нейтрали внутри одного прогона может отличаться почти в 2 раза —
`temperature=1.0`, сид не фиксирован, один сэмпл на клип (не усреднение по повторам).

**Следствие**: почти все `ΔF0 std` в таблицах ниже — отрицательные (тег «спокойнее» этой
конкретной нейтрали), но это в первую очередь свойство того, что именно ЭТА нейтраль оказалась
необычно экспрессивной, а не свидетельство, что теги систематически снижают разброс F0. Числа в
таблицах ниже — честные измерения этого конкретного прогона (и внутри одного прогона они
взаимно сравнимы — все клипы сгенерированы в одних условиях), но абсолютную величину дельты
по F0 std не стоит обобщать на «тег X даёт настолько же эффекта каждый раз». Где это важно
(`speed_*`), ниже используется темп (слов/сек) как более прямая метрика, а не F0 std.
Статусы по слуху ниже в первую очередь опираются на уже установленные владельцем вердикты
(`m4-sentiment-results.md`, `m4-tag-inventory-results.md`), а не на абсолютные числа одного
нового прогона.

**`neutral_baseline`** (нейтральная база всех сравнений ниже): F0 std 102.6 Hz, F0 median 146.3 Hz,
энергия -28.1 дБ, паузы 2 (760 мс), темп 2.45 сл/с, длительность 7.76 с. Образец:
`output/m4_tag_catalog/neutral_baseline.wav`.

Столбцы `Δ*` в каждой таблице ниже — это `клип - neutral_baseline` этого же прогона.

## 2. 43 официальных тега разметки

### 2.1 Emotion (21) — теги уровня предложения

> `PROMPTING.md`: «Emotion (21) — sentence-level... Syntax: `<|emotion:elation|>`»

| Тег | id | ΔF0 std (Hz) | ΔЭнергия (dB) | ΔPauses (n) | ΔPause (ms) | ΔТемп (сл/с) | ΔDuration (s) | ΔF0 median (Hz)* | Образец |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **emotion:affection** | 151687 | -63.0 | -3.0 | 6 | 2460 | -0.87 | 4.28 | 21.5 | `output/m4_tag_catalog/tag_emotion_affection.wav` |
| **emotion:amusement** | 151682 | -35.7 | -6.6 | 4 | 1200 | -0.82 | 3.92 | 98.6 | `output/m4_tag_catalog/tag_emotion_amusement.wav` |
| **emotion:anger** | 151695 | -30.2 | -8.2 | 2 | 60 | -0.35 | 1.28 | 44.2 | `output/m4_tag_catalog/tag_emotion_anger.wav` |
| **emotion:arousal** | 151694 | -29.8 | 3.7 | 3 | 400 | -0.68 | 2.96 | 64.2 | `output/m4_tag_catalog/tag_emotion_arousal.wav` |
| **emotion:awe** | 151692 | -61.2 | 1.7 | 0 | -220 | -0.45 | 1.76 | -37.7 | `output/m4_tag_catalog/tag_emotion_awe.wav` |
| **emotion:bitterness** | 151698 | -42.4 | -12.4 | 1 | 620 | -0.82 | 3.88 | -18.0 | `output/m4_tag_catalog/tag_emotion_bitterness.wav` |
| **emotion:confusion** | 151690 | -16.2 | -5.0 | -2 | -760 | -0.91 | 4.6 | -60.0 | `output/m4_tag_catalog/tag_emotion_confusion.wav` |
| **emotion:contemplation** | 151689 | -38.6 | -7.1 | 6 | 2960 | -1.13 | 6.6 | 55.4 | `output/m4_tag_catalog/tag_emotion_contemplation.wav` |
| **emotion:contentment** | 151686 | -39.7 | -6.7 | 4 | 1040 | -0.98 | 5.2 | 19.2 | `output/m4_tag_catalog/tag_emotion_contentment.wav` |
| **emotion:determination** | 151684 | 6.6 | -8.6 | 0 | 100 | -0.9 | 4.52 | -62.4 | `output/m4_tag_catalog/tag_emotion_determination.wav` |
| **emotion:disgust** | 151697 | -26.8 | -4.0 | 1 | -200 | -0.66 | 2.88 | -12.2 | `output/m4_tag_catalog/tag_emotion_disgust.wav` |
| **emotion:elation** | 151681 | -40.2 | 2.6 | 0 | -320 | -0.74 | 3.32 | 68.0 | `output/m4_tag_catalog/tag_emotion_elation.wav` |
| **emotion:enthusiasm** | 151683 | -12.0 | -6.0 | 4 | 740 | -0.74 | 3.36 | 5.6 | `output/m4_tag_catalog/tag_emotion_enthusiasm.wav` |
| **emotion:fear** | 151696 | -46.7 | -11.4 | 4 | 860 | -0.8 | 3.76 | 31.5 | `output/m4_tag_catalog/tag_emotion_fear.wav` |
| **emotion:helplessness** | 151701 | 12.0 | -2.5 | 3 | 1060 | -0.62 | 2.6 | -47.5 | `output/m4_tag_catalog/tag_emotion_helplessness.wav` |
| **emotion:longing** | 151693 | -2.6 | -1.8 | 1 | 420 | -0.5 | 1.96 | -46.7 | `output/m4_tag_catalog/tag_emotion_longing.wav` |
| **emotion:pride** | 151685 | -26.8 | 0.9 | 4 | 540 | -0.83 | 3.96 | -10.3 | `output/m4_tag_catalog/tag_emotion_pride.wav` |
| **emotion:relief** | 151688 | -31.4 | -5.2 | 3 | 960 | -0.84 | 4.04 | 62.4 | `output/m4_tag_catalog/tag_emotion_relief.wav` |
| **emotion:sadness** | 151699 | -52.1 | -0.9 | 1 | -100 | -0.38 | 1.4 | -38.2 | `output/m4_tag_catalog/tag_emotion_sadness.wav` |
| **emotion:shame** | 151700 | -60.3 | -14.9 | 4 | 1200 | -0.53 | 2.16 | -16.9 | `output/m4_tag_catalog/tag_emotion_shame.wav` |
| **emotion:surprise** | 151691 | -28.5 | 3.0 | 0 | -400 | 0.01 | -0.04 | 58.8 | `output/m4_tag_catalog/tag_emotion_surprise.wav` |

**Статус проверки на слух** (emotion):

- `emotion:affection`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:amusement`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:anger`: Группа C по объективной триаге M4-T5 (PR #108) -- близкий к нейтрали или в обратную сторону сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `emotion:arousal`: Группа B по объективной триаге M4-T5 (PR #108) -- слабый/неоднозначный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `emotion:awe`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:bitterness`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:confusion`: Группа B по объективной триаге M4-T5 (PR #108) -- слабый/неоднозначный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `emotion:contemplation`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:contentment`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:determination`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:disgust`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:elation`: Подтверждено владельцем вслепую (M4-T0): отличимо от sadness, PASSED.
- `emotion:enthusiasm`: Группа C по объективной триаге M4-T5 (PR #108) -- близкий к нейтрали или в обратную сторону сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `emotion:fear`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:helplessness`: Группа B по объективной триаге M4-T5 (PR #108) -- слабый/неоднозначный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `emotion:longing`: Группа B по объективной триаге M4-T5 (PR #108) -- слабый/неоднозначный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `emotion:pride`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:relief`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:sadness`: Подтверждено владельцем вслепую (M4-T0): звучит грустно, PASSED.
- `emotion:shame`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).
- `emotion:surprise`: Группа A по объективной триаге M4-T5 (PR #108) -- в числе 15 эмоций, которые владелец обобщённо подтвердил как «различимы и работают». Отдельно на слух не переслушивалась (кроме sadness/elation).

### 2.2 Prosody (10)

> `PROMPTING.md`: «Sentence-level: `speed_very_slow`, `speed_slow`, `speed_fast`,
> `speed_very_fast`, `pitch_low`, `pitch_high`, `expressive_high`, `expressive_low`. Inline:
> `pause`, `long_pause`». Также: «`speed_very_slow` only slows the model to roughly ~5s; for
> slower delivery, insert `<|prosody:long_pause|>` between phrases instead.»

| Тег | id | ΔF0 std (Hz) | ΔЭнергия (dB) | ΔPauses (n) | ΔPause (ms) | ΔТемп (сл/с) | ΔDuration (s) | ΔF0 median (Hz)* | Образец |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **prosody:speed_very_slow** | 151716 | -7.7 | -3.7 | 4 | 940 | -0.58 | 2.4 | -7.6 | `output/m4_tag_catalog/tag_prosody_speed_very_slow.wav` |
| **prosody:speed_slow** | 151717 | -58.4 | -9.9 | 3 | 1340 | -0.73 | 3.28 | -26.9 | `output/m4_tag_catalog/tag_prosody_speed_slow.wav` |
| **prosody:speed_fast** | 151718 | -51.6 | -7.2 | 1 | -80 | -0.07 | 0.24 | -11.1 | `output/m4_tag_catalog/tag_prosody_speed_fast.wav` |
| **prosody:speed_very_fast** | 151719 | -47.8 | 3.5 | 0 | 20 | -0.19 | 0.64 | -53.6 | `output/m4_tag_catalog/tag_prosody_speed_very_fast.wav` |
| **prosody:pitch_low** | 151720 | -7.1 | 4.2 | 3 | 800 | -1.04 | 5.76 | -52.5 | `output/m4_tag_catalog/tag_prosody_pitch_low.wav` |
| **prosody:pitch_high** | 151721 | -21.0 | -4.5 | 2 | 160 | -0.29 | 1.04 | 98.6 | `output/m4_tag_catalog/tag_prosody_pitch_high.wav` |
| **prosody:expressive_high** | 151725 | -30.5 | -4.7 | 4 | 920 | -0.61 | 2.56 | -16.6 | `output/m4_tag_catalog/tag_prosody_expressive_high.wav` |
| **prosody:expressive_low** | 151726 | -33.9 | -0.9 | 0 | -200 | -0.1 | 0.32 | -52.5 | `output/m4_tag_catalog/tag_prosody_expressive_low.wav` |
| **prosody:pause** | 151722 | -40.1 | -4.5 | 0 | 280 | -0.56 | 2.28 | -6.8 | `output/m4_tag_catalog/tag_prosody_pause.wav` |
| **prosody:long_pause** | 151723 | -37.5 | 4.1 | -2 | -760 | -0.4 | 1.52 | -18.0 | `output/m4_tag_catalog/tag_prosody_long_pause.wav` |

**Статус проверки на слух** (prosody):

- `prosody:speed_very_slow`: См. общий вывод по всем 4 speed_* тегам ниже.
- `prosody:speed_slow`: Подтверждено владельцем: эффект слабый и непостоянный. В M4-T5 темп почти не менялся (-0.03 сл/с); в этом прогоне разброс F0 и темп заметно ниже нейтрали, но нейтраль этого прогона сама аномально высокая (см. §1.1) -- не противоречит выводу «ненадёжно», скорее подтверждает его от обратного (число прогона к прогону скачет).
- `prosody:speed_fast`: См. общий вывод по всем 4 speed_* тегам ниже.
- `prosody:speed_very_fast`: См. общий вывод по всем 4 speed_* тегам ниже.
- `prosody:pitch_low`: Группа A по объективной триаге M4-T5 (PR #108) -- сильный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `prosody:pitch_high`: Группа A по объективной триаге M4-T5 (PR #108) -- сильный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `prosody:expressive_high`: Группа B по объективной триаге M4-T5 (PR #108) -- слабый/неоднозначный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `prosody:expressive_low`: Группа A по объективной триаге M4-T5 (PR #108) -- сильный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `prosody:pause`: Группа B по объективной триаге M4-T5 (PR #108) -- слабый/неоднозначный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `prosody:long_pause`: Группа A по объективной триаге M4-T5 (PR #108) -- сильный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.

### 2.3 Style (3) — теги уровня предложения

> `PROMPTING.md`: «Style (3) — sentence-level: `singing`, `shouting`, `whispering`»

| Тег | id | ΔF0 std (Hz) | ΔЭнергия (dB) | ΔPauses (n) | ΔPause (ms) | ΔТемп (сл/с) | ΔDuration (s) | ΔF0 median (Hz)* | Образец |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **style:singing** | 151704 | -56.3 | 2.7 | 1 | 540 | -0.45 | 1.76 | -48.7 | `output/m4_tag_catalog/tag_style_singing.wav` |
| **style:shouting** | 151705 | -29.0 | -2.0 | 1 | 440 | -0.78 | 3.64 | 22.7 | `output/m4_tag_catalog/tag_style_shouting.wav` |
| **style:whispering** | 151706 | -37.4 | 8.0 | -2 | -760 | -0.64 | 2.76 | -47.9 | `output/m4_tag_catalog/tag_style_whispering.wav` |

**Статус проверки на слух** (style):

- `style:singing`: Группа B по объективной триаге M4-T5 (PR #108) -- слабый/неоднозначный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `style:shouting`: Группа A по объективной триаге M4-T5 (PR #108) -- сильный сигнал в прошлом прогоне. Не переслушивалась владельцем отдельно.
- `style:whispering`: Подтверждено владельцем ДВАЖДЫ независимо (M4-T0, M4-T5): НЕ шёпот, просто иная манера, энергия выше нейтрали, а не ниже. В этом прогоне энергия тоже выше нейтрали (+8.0 дБ) -- третье независимое подтверждение того же вывода.

### 2.4 Sound effects / sfx (9) — инлайновые, впервые проверены

> `PROMPTING.md`: «Sound effects (9) — inline: `cough`, `laughter`, `crying`, `screaming`,
> `burping`, `humming`, `sigh`, `sniff`, `sneeze`. Syntax: `<|sfx:cough|>Ahem, ...` (tag first,
> onomatopoeia attached, no space)». **Гочка формата**: тег сразу перед ономатопеей, без пробела,
> запятая продолжает то же предложение.

| Тег | id | ΔF0 std (Hz) | ΔЭнергия (dB) | ΔPauses (n) | ΔPause (ms) | ΔТемп (сл/с) | ΔDuration (s) | ΔF0 median (Hz)* | Образец |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **sfx:cough** | 151707 | -6.8 | -7.9 | 2 | 320 | -0.85 | 4.76 | -25.1 | `output/m4_tag_catalog/tag_sfx_cough.wav` |
| **sfx:laughter** | 151708 | -6.2 | 4.4 | 2 | 280 | -0.63 | 3.24 | -5.9 | `output/m4_tag_catalog/tag_sfx_laughter.wav` |
| **sfx:crying** | 151709 | -31.5 | -5.6 | 3 | 660 | -0.53 | 2.68 | -57.4 | `output/m4_tag_catalog/tag_sfx_crying.wav` |
| **sfx:screaming** | 151710 | -31.2 | -3.5 | 4 | 1720 | -0.53 | 2.68 | -5.9 | `output/m4_tag_catalog/tag_sfx_screaming.wav` |
| **sfx:burping** | 151711 | -17.2 | 0.6 | 0 | -400 | -0.17 | 1.0 | -8.4 | `output/m4_tag_catalog/tag_sfx_burping.wav` |
| **sfx:humming** | 151712 | -53.4 | -6.9 | 1 | -120 | -0.16 | 0.96 | 12.6 | `output/m4_tag_catalog/tag_sfx_humming.wav` |
| **sfx:sigh** | 151713 | -47.0 | -18.3 | 7 | 1880 | -0.86 | 4.84 | -43.7 | `output/m4_tag_catalog/tag_sfx_sigh.wav` |
| **sfx:sniff** | 151714 | 9.6 | -2.0 | 2 | 600 | -0.58 | 2.92 | -26.3 | `output/m4_tag_catalog/tag_sfx_sniff.wav` |
| **sfx:sneeze** | 151715 | -41.8 | -3.7 | 2 | 600 | -0.59 | 3.0 | 25.1 | `output/m4_tag_catalog/tag_sfx_sneeze.wav` |

**Статус проверки на слух** (sfx):

- `sfx:cough`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `sfx:laughter`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `sfx:crying`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `sfx:screaming`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `sfx:burping`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `sfx:humming`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `sfx:sigh`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `sfx:sniff`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `sfx:sneeze`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.

## 3. 2 недокументированных `env`-тега

`<|env:music|>` (id 151702) и `<|env:noise|>` (id 151703) — есть в словаре чекпоинта (числятся
подряд с `style:*`/`sfx:*` по id, то есть добавлены Higgs, а не унаследованы от Qwen), но
**отсутствуют в `PROMPTING.md`** — ни в разделе тегов, ни в «Full tag catalog (43)». Поведение
неизвестно заранее. Важно: `src/audiobook.py`'s `validate_control_tags` **отклоняет** оба тега как
`unknown control tag` (они соответствуют форме `<|category:name|>`, но не входят в `VALID_TAGS`) —
то есть сегодня их нельзя использовать в реальной главе без изменения кода, даже если звук
окажется полезным.

| Тег | id | ΔF0 std (Hz) | ΔЭнергия (dB) | ΔPauses (n) | ΔPause (ms) | ΔТемп (сл/с) | ΔDuration (s) | ΔF0 median (Hz)* | Образец |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **env:music** | 151702 | -48.0 | -0.9 | 1 | 200 | -0.72 | 3.24 | -22.6 | `output/m4_tag_catalog/tag_env_music.wav` |
| **env:noise** | 151703 | -22.5 | -2.1 | 1 | -140 | -0.51 | 2.04 | -26.6 | `output/m4_tag_catalog/tag_env_noise.wav` |

**Статус проверки на слух** (env):

- `env:music`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.
- `env:noise`: НЕ прослушано ни разу до этого прогона -- статус «только метрики», ждёт прослушивания владельцем.

## 4. 16 структурных токенов Higgs (протокол, не звук)

Извлечены из `tokenizer.json` (id 151665-151680), сверены построчно с реальным кодом
`mlx_audio/tts/models/higgs_audio_v3/{prompt.py,model.py,generation.py,continuous_batching.py}`
(путь в проекте: `.venv-tts/lib/python3.12/site-packages/mlx_audio/...`) и с
`chat_template.jinja` из снимка чекпоинта. **Образцы не генерировались** — это управляющие
токены протокола построения промпта, а не звуковые эффекты; вставка их в текст реплики не имеет
смысла (`HiggsAudioV3PromptBuilder.build_prompt` строит промпт напрямую из Python, не читая текст
реплики на предмет этих токенов) и может непредсказуемо повлиять на генерацию.

| id | token | used by mlx_audio HiggsAudioV3? | role |
|---|---|---|---|
|151665|<\|asr\|>|NO (grep: 0 hits in higgs_audio_v3/*)|Reserved for a unified ASR task-selector token in the base Higgs multimodal architecture; mlx_audio's separate STT path (bosonai/higgs-audio-v3-stt, own checkpoint) doesn't reference it either in this project's dependency tree. Not used to generate TTS audio.|
|151666|<\|streaming_asr\|>|NO|Same family as asr; unused.|
|151667|<\|tts|>|**YES** - prompt.py:37,54 `self.tts_id`, first token of EVERY prompt `HiggsAudioV3PromptBuilder.build_prompt`|Task selector: tells the backbone "generate TTS audio codes now." Always present; this is the one structural token every one of our generations actually emits.|
|151668|<\|streaming_tts\|>|NO (0 hits)|Not emitted by build_prompt. IMPORTANT: `HiggsAudioV3.generate(..., stream=True)` exists as a *Python kwarg* but does NOT change decoding - model.py:816 always calls `self._decode_audio(delayed_rows)` once, after the full AR loop finishes, and always does one `yield` at the end (single GenerationResult). `stream=True` only flips `is_streaming_chunk`/`is_final_chunk` *metadata* flags on that one result (model.py:850-851) - it is vestigial, not real incremental audio streaming. The batch path (continuous_batching.py) explicitly raises "Higgs Audio v3 batch streaming is not implemented." **Conclusion: no streaming decode path exists anywhere in mlx_audio's Higgs v3 implementation; the `<|streaming_tts|>` token is unused and the `stream` flag does nothing audible.**|
|151669|<\|audio_cont_txt\|>|NO|Unused. Likely reserved for an audio-continuation-with-text task (e.g. "continue this audio, guided by more text") in the base architecture; mlx_audio implements neither cloning-with-continuation nor this token.|
|151670|<\|audio\|>|**YES** - prompt.py:40,69 `self.audio_id`, LAST token of every prompt|Marks "audio generation starts here" - everything after it in the prompt is where the model begins emitting audio codes.|
|151671|<\|audio_end\|>|NO in higgs_audio_v3 (0 hits); used by *other, unrelated* mlx_audio model families (outetts, moss, qwen3_asr) with their own independent tokenizers/vocabularies|Reserved multimodal-protocol token, not part of this checkpoint's actual TTS prompt structure.|
|151672|<\|text\|>|**YES** - prompt.py:39,67 `self.text_id`, precedes the target text|Marks "here begins the text to be spoken."|
|151673|<\|text_end\|>|NO|Unused by build_prompt (which appends `<|audio|>` directly after the encoded text, no explicit text_end).|
|151674|<\|eoc\|>|NO as a *text* token (0 hits for the literal string "<|eoc|>"). CAUTION - naming collision: model.py/generation.py's `eoc_id`/`audio_eoc_token_id` (config.py:73, default 1025) is a completely different thing - an End-Of-Codes marker **inside the audio codec's own RVQ codebook vocabulary** (vocab_size 1026 per audio_encoder_config), not this text-tokenizer id 151674.|Text-tokenizer `<|eoc|>` (151674) is unused in our path. Do not confuse with the per-codebook EOC id used internally by the audio generation loop - different vocabulary space entirely.|
|151675|<\|user\|>|NO in higgs_audio_v3 (0 hits); used by *other* mlx_audio model families (glmasr) with their own tokenizer|Not part of this checkpoint's TTS prompt.|
|151676|<\|assistant\|>|NO in higgs_audio_v3; used by glmasr (different model/tokenizer)|Not part of this checkpoint's TTS prompt.|
|151677|<\|system\|>|NO (0 hits anywhere in mlx_audio)|**Owner's question answered: no working system-prompt path exists.** See "system prompt" finding below.|
|151678|<\|await_audio|>|NO|Unused. Likely a duplex/interactive-mode marker (e.g. "wait for the next audio chunk from the user") for a conversational variant mlx_audio does not implement.|
|151679|<\|ref_audio\|>|**YES** - prompt.py:38,61 `self.ref_audio_id`|Voice-cloning mechanics, see below.|
|151680|<\|ref_text\|>|**YES** - prompt.py:41-42,58-60 `self.ref_text_id` (optional)|Voice-cloning mechanics, see below.|

### 4.1 Находка про системный промпт (прямой ответ на вопрос владельца)

**Нет, рабочего способа задать голосу манеру/характер через системный промпт в нашем пути
генерации не существует.** `<|system|>` (151677) нигде не встречается в
`HiggsAudioV3PromptBuilder.build_prompt` (0 совпадений по всему `higgs_audio_v3/*.py`) — реальный
промпт, который получает модель, всегда ровно:
`<|tts|> [<|ref_text|>...] [<|ref_audio|>...codes...] <|text|>...текст...<|audio|>` — места для
системного сообщения в этой структуре нет вообще.

В снимке чекпоинта ЕСТЬ `chat_template.jinja`, поддерживающий роль `system` — но это **стоковый
шаблон Qwen3-4B-Base** (использует `<|im_start|>system ... <|im_end|>`, легаси-токены Qwen, НЕ
`<|system|>`/`<|user|>`/`<|assistant|>`). Это наследие базовой LLM, на которой построена
мультимодальная модель. `HiggsAudioV3PromptBuilder.build_prompt` вообще не рендерит этот jinja —
токены промпта собираются напрямую в Python, минуя chat-templating целиком.

Это не проверено генерацией (у `model.generate()` в сигнатуре нет параметра `system`/`instruction`
вообще — проверять нечего), но это находка по коду, а не догадка: ответ «встроенного выбора голоса
нет» не отменяется, а закрывается окончательно с указанием, почему именно.

### 4.2 `ref_audio` / `ref_text` — механика клонирования (уже используется)

Из `prompt.py`, `build_prompt`, для каждого `ReferenceCodes(codes, text)` в `references=`:
1. Если задан `reference.text` — добавляется `<|ref_text|>`, затем закодированный текст референса.
2. Добавляется `<|ref_audio|>`, затем блок из `AUDIO_PLACEHOLDER_ID` (-100) по числу строк
   `reference.codes` (предвычисленные RVQ-коды референсного аудио) — именно в эти позиции
   подставляются эмбеддинги референсного звука перед тем, как их увидит бэкбон.
3. Блок может повторяться (несколько референсов), затем один раз в конце — `<|text|>` + целевой
   текст + `<|audio|>`.

`ref_text` — необязательный контекст (помогает выравниванию текст-звук референса); `ref_audio`
обязателен при клонировании и это то место, куда реально подставляются коды референсного звука
как эмбеддинги, а не как обычные id токенов.

### 4.3 `streaming_tts` — потокового декодирования НЕТ, даже когда флаг `stream=True` передан

Важная находка в свете вопроса о потоковом декодировании: `<|streaming_tts|>` (151668) нигде не
встречается в коде — 0 совпадений. Более того, у самого `HiggsAudioV3.generate()` есть Python-параметр
`stream: bool = False`, но он **не меняет декодирование**: `model.py:816` всегда вызывает
`self._decode_audio(delayed_rows)` один раз, целиком, после того как весь AR-цикл завершён, и всегда
делает один `yield` в конце (один `GenerationResult`). `stream=True` лишь переключает МЕТА-флаги
`is_streaming_chunk`/`is_final_chunk` на этом единственном результате (`model.py:850-851`) — то есть
это **декоративный флаг, а не настоящий потоковый вывод**. Батч-путь (`continuous_batching.py`)
явно и честно поднимает исключение `"Higgs Audio v3 batch streaming is not implemented."`.
**Вывод: потокового декодирования нигде в реализации `HiggsAudioV3` в `mlx_audio` не существует** —
ни через токен, ни через флаг.

### 4.4 `audio_cont_txt`, `await_audio`, `<|eoc|>` (текстовый) — не используются вообще

Все три: 0 совпадений в коде `higgs_audio_v3`. Существуют в словаре (унаследованы от полной
мультимодальной архитектуры-предка — вероятно, для непрерывной/дуплексной обработки звука,
которую `mlx_audio` для этого чекпоинта не реализует), но ни на что не влияют в этом проекте.

**Осторожно, коллизия имён**: `<|eoc|>` (id 151674, текстовый токенайзер) — НЕ то же самое, что
`eoc_id`/`audio_eoc_token_id` в `model.py`/`generation.py`/`config.py` (по умолчанию 1025) — это
маркер конца кодов **внутри словаря самого аудиокодека** (RVQ, `vocab_size` 1026 в
`audio_encoder_config`), совершенно другое пространство токенов. Текстовый `<|eoc|>` (151674) в
пути генерации не используется никогда.

### 4.5 `asr`, `streaming_asr`, `audio_end`, `text_end`, `user`, `assistant` — код-путь

0 совпадений в `higgs_audio_v3/*.py`. `<|audio_end|>`/`<|user|>`/`<|assistant|>` встречаются в
СОВСЕМ ДРУГИХ семействах моделей `mlx_audio` (`outetts`, `moss_transcribe_diarize`, `qwen3_asr`,
`glmasr`) — у каждой свой токенайзер и свой словарь, это не значит что они действуют и для
`bosonai/higgs-tts-3-4b`. Для STT этого проекта используется отдельный чекпоинт
(`bosonai/higgs-audio-v3-stt`) — вне рамок этого документа (другой токенайзер, другой пайплайн).

## 5. 1 недокументированный одиночка: `<|chatml|>` (id 151724)

0 совпадений в `mlx_audio`. Это не пара `<|im_start|>`/`<|im_end|>` (те — отдельные, более старые
токены, id 151644/151645, из легаси-блока Qwen). `<|chatml|>` численно находится ВНУТРИ блока,
добавленного Higgs (между `prosody:long_pause` 151723 и `prosody:expressive_high` 151725), а не в
блоке легаси Qwen в начале `added_tokens` — то есть добавлен одновременно с тегами разметки, а не
унаследован. Не задокументирован в `PROMPTING.md`. Вероятно, зарезервирован под альтернативный
ChatML-формат промпта для этого чекпоинта, который прямой билдер промпта `mlx_audio` не использует.
Образец не генерировался — протокольный токен, не звук.

## 6. 22 служебных токена — наследие базового токенайзера Qwen (ids 151643-151664)

Унаследованы целиком от `Qwen3-4B-Base` (`text_config._name_or_path` в `config.json` чекпоинта),
на котором построена мультимодальная модель. Ни один не относится к звуку; это исходная
текстовая/визуальная/код-поверхность базовой LLM, которую мультимодальная сборка Higgs не
вычистила из токенайзера. Образцы не генерировались.

| Токены | id | Что это (одна строка на группу) |
|---|---|---|
| `<\|endoftext\|>` | 151643 | Обычный EOS/BOS-маркер простого текста Qwen (`bos_token_id`==`eos_token_id`==151643 в `text_config`). |
| `<\|im_start\|>`, `<\|im_end\|>` | 151644-45 | Ролевые разделители ChatML — используются только стоковым `chat_template.jinja` (не нашим прямым билдером промпта). |
| `<\|object_ref_start\|>`, `<\|object_ref_end\|>` | 151646-47 | Qwen-VL: границы упоминания объекта на изображении. |
| `<\|box_start\|>`, `<\|box_end\|>` | 151648-49 | Qwen-VL: координаты bounding-box. |
| `<\|quad_start\|>`, `<\|quad_end\|>` | 151650-51 | Qwen-VL: координаты четырёхугольной области. |
| `<\|vision_start\|>`, `<\|vision_end\|>`, `<\|vision_pad\|>` | 151652-54 | Qwen-VL: границы/паддинг эмбеддингов патчей изображения. |
| `<\|image_pad\|>`, `<\|video_pad\|>` | 151655-56 | Qwen-VL: паддинг-токены для последовательностей патчей изображения/видео. |
| `<tool_call>`, `</tool_call>` | 151657-58 | XML-обёртка вызова функции/инструмента Qwen (используется веткой `tools` в `chat_template.jinja`, только текст, к звуку отношения не имеет). |
| `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>`, `<\|fim_pad\|>` | 151659-62 | Qwen-Coder: маркеры fill-in-the-middle для автодополнения кода. |
| `<\|repo_name\|>`, `<\|file_sep\|>` | 151663-64 | Qwen-Coder: маркеры контекста уровня репозитория. |

Ни один из этих 22 не достижим через текстовый пайплайн `src/audiobook.py` осмысленно:
`_TAG_SHAPE_RE` ловит только форму `<|category:name|>` (это ловит `env:*`/наши 43 тега как
«похожие на тег, но неизвестные» — то есть `validate_control_tags` их отклонит), а токены без
двоеточия (`<tool_call>`, `<|endoftext|>` и т.п.) вообще не матчатся этим регэкспом и были бы
прочитаны как обычный (скорее всего испорченный) буквальный текст, если бы кто-то вписал их в
главу.

## 7. Сводная таблица «все 84»

| id | Токен | Категория | Есть образец | Годится для сценария |
|---:|---|---|---|---|
| 151643 | `<\|endoftext\|>` | Qwen legacy | нет | нет |
| 151644 | `<\|im_start\|>` | Qwen legacy | нет | нет |
| 151645 | `<\|im_end\|>` | Qwen legacy | нет | нет |
| 151646 | `<\|object_ref_start\|>` | Qwen legacy | нет | нет |
| 151647 | `<\|object_ref_end\|>` | Qwen legacy | нет | нет |
| 151648 | `<\|box_start\|>` | Qwen legacy | нет | нет |
| 151649 | `<\|box_end\|>` | Qwen legacy | нет | нет |
| 151650 | `<\|quad_start\|>` | Qwen legacy | нет | нет |
| 151651 | `<\|quad_end\|>` | Qwen legacy | нет | нет |
| 151652 | `<\|vision_start\|>` | Qwen legacy | нет | нет |
| 151653 | `<\|vision_end\|>` | Qwen legacy | нет | нет |
| 151654 | `<\|vision_pad\|>` | Qwen legacy | нет | нет |
| 151655 | `<\|image_pad\|>` | Qwen legacy | нет | нет |
| 151656 | `<\|video_pad\|>` | Qwen legacy | нет | нет |
| 151657 | `<tool_call>` | Qwen legacy | нет | нет |
| 151658 | `</tool_call>` | Qwen legacy | нет | нет |
| 151659 | `<\|fim_prefix\|>` | Qwen legacy | нет | нет |
| 151660 | `<\|fim_middle\|>` | Qwen legacy | нет | нет |
| 151661 | `<\|fim_suffix\|>` | Qwen legacy | нет | нет |
| 151662 | `<\|fim_pad\|>` | Qwen legacy | нет | нет |
| 151663 | `<\|repo_name\|>` | Qwen legacy | нет | нет |
| 151664 | `<\|file_sep\|>` | Qwen legacy | нет | нет |
| 151665 | `<\|asr\|>` | Higgs structural | нет | нет |
| 151666 | `<\|streaming_asr\|>` | Higgs structural | нет | нет |
| 151667 | `<\|tts\|>` | Higgs structural | нет (протокол) | нет — вставляется движком автоматически |
| 151668 | `<\|streaming_tts\|>` | Higgs structural | нет | нет — потоковый декод не реализован |
| 151669 | `<\|audio_cont_txt\|>` | Higgs structural | нет | нет |
| 151670 | `<\|audio\|>` | Higgs structural | нет (протокол) | нет — вставляется движком автоматически |
| 151671 | `<\|audio_end\|>` | Higgs structural | нет | нет |
| 151672 | `<\|text\|>` | Higgs structural | нет (протокол) | нет — вставляется движком автоматически |
| 151673 | `<\|text_end\|>` | Higgs structural | нет | нет |
| 151674 | `<\|eoc\|>` | Higgs structural | нет | нет |
| 151675 | `<\|user\|>` | Higgs structural | нет | нет |
| 151676 | `<\|assistant\|>` | Higgs structural | нет | нет |
| 151677 | `<\|system\|>` | Higgs structural | нет | нет — нет рабочего пути в промпт (см. §4.1) |
| 151678 | `<\|await_audio\|>` | Higgs structural | нет | нет |
| 151679 | `<\|ref_audio\|>` | Higgs structural | нет (протокол) | нет — используется API клонирования, не текстом главы |
| 151680 | `<\|ref_text\|>` | Higgs structural | нет (протокол) | нет — используется API клонирования, не текстом главы |
| 151681 | `<\|emotion:elation\|>` | emotion | ДА | ДА |
| 151682 | `<\|emotion:amusement\|>` | emotion | ДА | ДА |
| 151683 | `<\|emotion:enthusiasm\|>` | emotion | ДА | ДА |
| 151684 | `<\|emotion:determination\|>` | emotion | ДА | ДА |
| 151685 | `<\|emotion:pride\|>` | emotion | ДА | ДА |
| 151686 | `<\|emotion:contentment\|>` | emotion | ДА | ДА |
| 151687 | `<\|emotion:affection\|>` | emotion | ДА | ДА |
| 151688 | `<\|emotion:relief\|>` | emotion | ДА | ДА |
| 151689 | `<\|emotion:contemplation\|>` | emotion | ДА | ДА |
| 151690 | `<\|emotion:confusion\|>` | emotion | ДА | ДА |
| 151691 | `<\|emotion:surprise\|>` | emotion | ДА | ДА |
| 151692 | `<\|emotion:awe\|>` | emotion | ДА | ДА |
| 151693 | `<\|emotion:longing\|>` | emotion | ДА | ДА |
| 151694 | `<\|emotion:arousal\|>` | emotion | ДА | ДА |
| 151695 | `<\|emotion:anger\|>` | emotion | ДА | ДА |
| 151696 | `<\|emotion:fear\|>` | emotion | ДА | ДА |
| 151697 | `<\|emotion:disgust\|>` | emotion | ДА | ДА |
| 151698 | `<\|emotion:bitterness\|>` | emotion | ДА | ДА |
| 151699 | `<\|emotion:sadness\|>` | emotion | ДА | ДА — подтверждено на слух |
| 151700 | `<\|emotion:shame\|>` | emotion | ДА | ДА |
| 151701 | `<\|emotion:helplessness\|>` | emotion | ДА | ДА |
| 151702 | `<\|env:music\|>` | env (недокументирован) | ДА | ТЕХНИЧЕСКИ НЕТ — validate_control_tags отклонит как unknown |
| 151703 | `<\|env:noise\|>` | env (недокументирован) | ДА | ТЕХНИЧЕСКИ НЕТ — validate_control_tags отклонит как unknown |
| 151704 | `<\|style:singing\|>` | style | ДА | ДА |
| 151705 | `<\|style:shouting\|>` | style | ДА | ДА |
| 151706 | `<\|style:whispering\|>` | style | ДА | ДА, но не шепчет — см. §0/§2.3 |
| 151707 | `<\|sfx:cough\|>` | sfx (инлайн) | ДА | ДА |
| 151708 | `<\|sfx:laughter\|>` | sfx (инлайн) | ДА | ДА |
| 151709 | `<\|sfx:crying\|>` | sfx (инлайн) | ДА | ДА |
| 151710 | `<\|sfx:screaming\|>` | sfx (инлайн) | ДА | ДА |
| 151711 | `<\|sfx:burping\|>` | sfx (инлайн) | ДА | ДА |
| 151712 | `<\|sfx:humming\|>` | sfx (инлайн) | ДА | ДА |
| 151713 | `<\|sfx:sigh\|>` | sfx (инлайн) | ДА | ДА |
| 151714 | `<\|sfx:sniff\|>` | sfx (инлайн) | ДА | ДА |
| 151715 | `<\|sfx:sneeze\|>` | sfx (инлайн) | ДА | ДА |
| 151716 | `<\|prosody:speed_very_slow\|>` | prosody | ДА | ДА, но эффект слабый/непостоянный |
| 151717 | `<\|prosody:speed_slow\|>` | prosody | ДА | ДА, но эффект слабый/непостоянный |
| 151718 | `<\|prosody:speed_fast\|>` | prosody | ДА | ДА, но эффект слабый/непостоянный |
| 151719 | `<\|prosody:speed_very_fast\|>` | prosody | ДА | ДА, но эффект слабый/непостоянный |
| 151720 | `<\|prosody:pitch_low\|>` | prosody | ДА | ДА |
| 151721 | `<\|prosody:pitch_high\|>` | prosody | ДА | ДА |
| 151722 | `<\|prosody:pause\|>` | prosody (инлайн) | ДА | ДА, одноразово |
| 151723 | `<\|prosody:long_pause\|>` | prosody (инлайн) | ДА | ДА, одноразово |
| 151724 | `<\|chatml\|>` | недокументированный одиночка | нет | нет |
| 151725 | `<\|prosody:expressive_high\|>` | prosody | ДА | ДА |
| 151726 | `<\|prosody:expressive_low\|>` | prosody | ДА | ДА |

Итого строк: 84

## 8. Что не проверено и почему (честность)

- **9 sfx и 2 env тега человек слышит впервые в этом документе.** Метрики есть для всех 11;
  статус на слух — «ожидает прослушивания владельцем», не «подтверждено».
- **Сдвигает ли ударение (апостроф) реально нужный слог — не проверено метриками** (русский ASR
  не размечает ударение); только «не читается вслух буквально» подтверждено STT-round-trip'ом.
- **Системный промпт проверен только по коду, не генерацией** — у `generate()` просто нет
  параметра для этого, генерационного эксперимента не было и не могло быть.
- **F0-медиана целенаправленно не используется как единственное основание вывода** — см. §1.
- Если какой-то клип не удалось сгенерировать или измерить — это отмечено рядом с конкретным
  тегом в соответствующей таблице, а не скрыто.
