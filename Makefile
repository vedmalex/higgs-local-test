.PHONY: setup info download-models tts stt benchmark clean-output

setup:
	./scripts/bootstrap.sh

info:
	./scripts/system_info.sh

download-models:
	./scripts/download_models.sh

tts:
	./scripts/test_tts.sh

stt:
	./scripts/test_stt.sh

benchmark:
	python3 src/benchmark.py

clean-output:
	find output logs -type f ! -name .gitkeep -delete
