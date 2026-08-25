.PHONY: setup info download-models tts stt benchmark clean-output upload-gdrive download-gdrive sentiment-survey

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

upload-gdrive:
	python3 scripts/gdrive_sync.py upload --all

download-gdrive:
	@test -n "$(FOLDER_ID)" || (echo "ERROR: specify FOLDER_ID, e.g.: make download-gdrive FOLDER_ID=<id>" && exit 1)
	python3 scripts/gdrive_sync.py download --folder-id $(FOLDER_ID) --dest output

clean-output:
	find output logs -type f ! -name .gitkeep -delete

# Local blind-listening survey app for Higgs sentiment/tag verification (issue #57).
# Stdlib-only (no model load, no GPU) -- safe to run while a benchmark or generation
# job is using the GPU. Opens a browser tab; Ctrl+C stops the server.
sentiment-survey:
	@if [ -x .venv-tts/bin/python3 ]; then \
		.venv-tts/bin/python3 src/sentiment_survey/server.py; \
	else \
		python3 src/sentiment_survey/server.py; \
	fi
