# Model downloads guide

The local network is too slow/unstable for multi-gigabyte Hugging Face downloads. This
project fetches model weights through Google Colab instead and stores them on Google Drive,
per AGENTS.md's "Model downloads" rule. This guide is the how-to; the rule itself stays short
and lives in `AGENTS.md`.

## 1. Check before you fetch anything

- **Local cache**: `ls ~/.cache/huggingface/hub/ | grep -i <name>`
- **Drive**: `python3 scripts/gdrive_sync.py list --path higgs-benchmark/model-cache`
  (read-only, lists archive names/sizes/ids; add `--account=<email>` if you have multiple
  `gcloud` accounts and the default one lacks Drive access).
- **The catalog**: `notebooks/model_catalog.json` — the project's single inventory of every
  model it uses, has used, or has considered. Each entry records `status` (`local+drive`,
  `drive_only`, `local_only`, `needed`, `candidate`, `rejected`) and a `size_gb`/`license` where
  known. Treat `status` as a snapshot from whenever it was last verified, not live truth — the
  two checks above are the ground truth.

If it's already on Drive: `tar -xf <name>.tar -C ~/.cache/huggingface/hub/` and you're done —
`huggingface_hub` recognizes the extracted `models--<org>--<name>` directory as already cached
and every later `make tts` / `make stt` / Qwen local run skips downloading it again.

## 2. If it's missing from both

1. Add an entry to `notebooks/model_catalog.json` with the exact `repo_id` (and `revision` if
   the project pins one) and set `"fetch_now": true`. Read the file's own `_readme` field for
   the field list.
2. A download over a few GB: tell the owner and get a go-ahead *before* setting `fetch_now`.
3. Ask the owner to open `notebooks/model_prefetch_to_drive.ipynb` in Colab and run it — an
   agent never runs this notebook itself. It reads the catalog from `main`, downloads every
   `fetch_now: true` entry, packs each into a `.tar` preserving the Hugging Face cache's
   blob/snapshot symlink layout (Drive's FUSE mount does not preserve that layout on loose
   files), and uploads it to `MyDrive/higgs-benchmark/model-cache/`.
4. Once it lands on Drive, extract locally as in step 1, and set `fetch_now` back to `false`
   in the catalog (the entry's `status` becomes `local+drive`).

## 3. Adding a new model to the catalog later

Append an entry to `notebooks/model_catalog.json`'s `models` array. Leave `fetch_now: false`
and `status: "candidate"` until the model is actually decided on and about to be fetched — the
catalog is meant to hold considered-but-not-fetched models too (e.g. the music-generation
survey candidates for issue #115), so listing something there does not by itself queue a
download.
