.PHONY: setup info download-models tts stt benchmark clean-output upload-gdrive download-gdrive

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
