# Higgs Audio v3: Руководство по созданию многоголосых аудиокниг (Audiobook Production Guide)

Данное руководство описывает полный рабочий процесс создания аудиокниг с озвучкой по ролям (рассказчик + персонажи), динамическим переключением голосов, вставкой пауз и сшивкой готовых глав в **Higgs Audio v3**.

---

## 1. Архитектура проекта аудиокниги

Для создания аудиокниги организуется следующая структура папок:

```text
higgs-local-test/
├── voices/                     # Библиотека профилей голосов (.npy или .wav)
│   ├── narrator.npy            # Голос рассказчика / диктора
│   ├── narrator.txt            # Текст референса рассказчика
│   ├── character_krishna.npy   # Голос Кришны
│   ├── character_krishna.txt
│   ├── character_arjuna.npy    # Голос Арджуны
│   └── character_arjuna.txt
├── book/                       # Сценарий книги по главам
│   ├── chapter_01.json
│   └── chapter_02.json
└── output/
    └── audiobook/              # Готовые сшитые главы аудиокниги
        ├── chapter_01.wav
        └── chapter_02.wav
```

---

## 2. Подготовка и сохранение голосов персонажей

Каждый персонаж создаётся из короткого 7–12 секундного образца речи (см. [Voice Cloning Guide](voice_cloning_guide.md)) и один раз сохраняется в папку `voices/`:

```python
import numpy as np
from pathlib import Path
from mlx_audio.tts.utils import load

model = load("bosonai/higgs-tts-3-4b", model_type="higgs_audio_v3")
Path("voices").mkdir(exist_ok=True)

def register_voice(name: str, wav_path: str, ref_text: str):
    codes = model.encode_reference_audio(wav_path)
    np.save(f"voices/{name}.npy", np.array(codes))
    Path(f"voices/{name}.txt").write_text(ref_text, encoding="utf-8")
    print(f"Голос зарегистрирован: voices/{name}.npy")

# Пример регистрации персонажей:
# register_voice("narrator", "samples/narrator_ref.wav", "В тихой заводи реки плещется чистая вода...")
# register_voice("arjuna", "samples/arjuna_ref.wav", "Сегодня мы начинаем новый важный проект...")
```

---

## 3. Формат разметки главы (Сценарий книги)

Глава книги сохраняется в виде структурированного файла (например, `book/chapter_01.json`), где для каждого абзаца или реплики задаётся персонаж, эмоциональные теги и текст:

```json
[
  {
    "speaker": "narrator",
    "text": "Глава первая. На поле битвы Курукшетра собрались воины двух армий."
  },
  {
    "speaker": "arjuna",
    "text": "<|emotion:sadness|><|prosody:speed_slow|>О Кришна, видя моих родственников, готовых к бою, мои члены слабеют, а в горле пересыхает."
  },
  {
    "speaker": "krishna",
    "text": "<|emotion:contentment|>О Арджуна, откуда пришла к тебе эта скверна в столь критический час? Она не подобает арию и не ведёт в высшие миры."
  },
  {
    "speaker": "narrator",
    "text": "Услышав слова Господа, Арджуна опустил лук и погрузился в глубокие раздумья."
  }
]
```

---

## 4. Скрипт генерации и сшивки аудиокниги (Python Pipeline)

Скрипт последовательно загружает коды каждого персонажа, синтезирует фрагмент, вставляет естественные паузы нужной длительности и склеивает всё в единый трек главы:

```python
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_audio.tts.utils import load
from mlx_audio.audio_io import write as audio_write

MODEL_ID = "bosonai/higgs-tts-3-4b"

def generate_silence(duration_sec: float, sample_rate: int = 24000) -> np.ndarray:
    """Генерация тишины заданной длительности для пауз между репликами."""
    return np.zeros(int(duration_sec * sample_rate), dtype=np.float32)

def generate_chapter(chapter_json_path: Path, output_wav_path: Path):
    print(f"\n=== Генерация главы: {chapter_json_path.name} ===")
    with open(chapter_json_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    # 1. Загрузка модели MLX
    model = load(MODEL_ID, model_type="higgs_audio_v3")
    sample_rate = model.sample_rate

    # 2. Кэширование голосов в память
    voice_cache = {}
    for item in script:
        spk = item["speaker"]
        if spk not in voice_cache:
            codes_file = Path(f"voices/{spk}.npy")
            text_file = Path(f"voices/{spk}.txt")
            if not codes_file.exists() or not text_file.exists():
                raise FileNotFoundError(f"Голос для {spk} не найден в папке voices/")
            voice_cache[spk] = {
                "codes": mx.array(np.load(codes_file)),
                "text": text_file.read_text(encoding="utf-8").strip()
            }

    # 3. Последовательная генерация реплик и склейка
    audio_tracks = []
    prev_speaker = None

    for idx, line in enumerate(script, 1):
        spk = line["speaker"]
        text = line["text"]
        print(f"[{idx}/{len(script)}] Озвучка ({spk}): {text[:50]}...")

        # Вставка паузы между репликами
        if prev_speaker is not None:
            # Если сменился персонаж — пауза 1.0 сек, если тот же диктор — пауза 0.5 сек
            pause_sec = 1.0 if spk != prev_speaker else 0.5
            audio_tracks.append(generate_silence(pause_sec, sample_rate))

        voice = voice_cache[spk]
        results = list(model.generate(
            text=text,
            ref_audio_codes=voice["codes"],
            ref_text=voice["text"],
            temperature=1.0,
            max_new_tokens=2048,
        ))

        line_audio = np.concatenate([np.asarray(r.audio).reshape(-1) for r in results])
        audio_tracks.append(line_audio)
        prev_speaker = spk

    # 4. Финальная сшивка главы
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    full_chapter_audio = np.concatenate(audio_tracks)
    audio_write(str(output_wav_path), full_chapter_audio, sample_rate)
    
    total_duration = len(full_chapter_audio) / sample_rate
    print(f"\n Готовая глава сохранена: {output_wav_path} (длительность: {total_duration/60:.2f} мин)")

if __name__ == "__main__":
    # Пример запуска:
    # generate_chapter(Path("book/chapter_01.json"), Path("output/audiobook/chapter_01.wav"))
    pass
```

---

## 5. Таблица рекомендуемых пауз в аудиокниге

| Переход | Рекомендуемая пауза | Назначение |
| :--- | :---: | :--- |
| **Между предложениями одного абзаца** | `0.3 – 0.5 сек` | Естественный вдох диктора |
| **Между абзацами** | `0.7 – 1.0 сек` | Смена мысли или сцены |
| **Между репликами разных персонажей** | `0.9 – 1.2 сек` | Обозначение смены говорящего |
| **Перед и после эпиграфов / стихов** | `1.5 – 2.0 сек` | Выделение важного фрагмента |
| **Между главами книги** | `2.5 – 3.5 сек` | Завершение смыслового блока |

---

## 6. Экспорт в финальные форматы (MP3 / M4B)

Для удобного прослушивания на телефонах и в плеерах готовый WAV легко конвертируется через `ffmpeg`:

```bash
# Конвертация в MP3 (высокое качество 192 kbps):
ffmpeg -i output/audiobook/chapter_01.wav -codec:a libmp3lame -b:a 192k output/audiobook/chapter_01.mp3

# Конвертация всей книги в аудиокнигу M4B с закладками:
ffmpeg -i output/audiobook/chapter_01.wav -c:a aac -b:a 128k output/audiobook/audiobook.m4b
```
