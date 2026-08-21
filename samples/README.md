# Samples

`tts_ru.txt` is the committed Russian TTS smoke text.

For STT, record `stt_ru.wav` (20–60 seconds is ideal) using exactly this text:

> Сегодня мы проверяем качество распознавания русской речи системой Higgs Audio.
>
> Вриндаван находится в Индии.  
> Шри Чайтанья Махапрабху учил повторению святого имени.
>
> Кришна.  
> Радхарани.  
> Шримад-Бхагаватам.  
> Гопала Бхатта Госвами.  
> Радха-Раман.

The test runner normalizes it to mono 16 kHz WAV with `ffmpeg`. For optional voice cloning, add an authorized `reference.wav` and its exact `reference.txt`. These recordings are gitignored and never sent to an external API.

