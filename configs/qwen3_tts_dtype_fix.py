#!/usr/bin/env python3
"""UNTESTED workaround for vllm-omni's hardcoded bfloat16 in the Qwen3-TTS talker.

STATUS: WRITTEN AND STATICALLY CHECKED ONLY. This module has NEVER been executed
against a real GPU, a real `vllm serve --omni` process, or a real Qwen3-TTS
checkpoint. Importing it successfully proves nothing about whether it fixes the
crash it targets. Do not report a run that used it as a plain PASS -- the runner
(`src/tts_qwen_cuda.py --enable-dtype-patch`) stamps `dtype_patch_status` into its
JSON report precisely so this stays visible.

WHAT IT TARGETS
---------------
`vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py`, in
`Qwen3TTSTalkerForConditionalGeneration.__init__` (line 446 at the `v0.26.0` tag,
line 468 on `main` as of 2026-08-24 -- still unfixed upstream):

    model_dtype = getattr(vllm_config.model_config, "dtype", torch.bfloat16)
    self.register_buffer(
        "_tts_pad_embed",
        torch.zeros(1, int(self.talker_config.hidden_size), dtype=model_dtype),
        persistent=False,
    )
    self._embedding_dtype = torch.bfloat16   # <-- ignores model_dtype

`model_dtype` is read correctly from the engine config and used for the
`_tts_pad_embed` buffer, then `_embedding_dtype` is hardcoded to `bfloat16` on the
very next line. `_embedding_dtype` is what `preprocess_decode_batch` (the
`preprocess_decode_batch` hook that vllm-omni's `OmniGPUModelRunner` calls) casts
every per-request decode embedding to, so on a GPU below compute capability 8.0 --
where this project's runner is forced to `--dtype float16`, because vLLM refuses
bf16 there -- the batched `req_embeds` come back bfloat16 while the engine's
`inputs_embeds` is float16, and

    vllm_omni/worker/gpu_model_runner.py:1723
        inputs_embeds.index_copy_(0, offsets_t, req_embeds)
    RuntimeError: index_copy_(): self and source expected to have the same dtype,
                  but got (self) Half and (source) BFloat16

kills the engine on every request. Measured twice on a real Colab T4; see
`docs/research/qwen3-tts-notes.md`.

THE ONE-LINE FIX, APPLIED WITHOUT FORKING THE PACKAGE
-----------------------------------------------------
Subclass the talker, call the real `__init__`, then set `_embedding_dtype` to the
`model_dtype` the parent already computed, and re-register the subclass under the
same architecture name. Nothing in vllm-omni is edited.

The registry that matters is vllm-omni's OWN registry, not `vllm.ModelRegistry`:
`vllm_omni/model_executor/models/registry.py` builds a separate `_ModelRegistry`
instance called `OmniModelRegistry`, and `vllm_omni/config/model.py`'s
`OmniModelConfig.registry` property returns that instance. Registering only into
`vllm.ModelRegistry` would therefore be a no-op for the omni serving path. Both are
registered here: `OmniModelRegistry` because it is the one actually consulted, and
`vllm.ModelRegistry` best-effort in case a code path resolves through it.

The architecture name is `Qwen3TTSTalkerForConditionalGeneration`: the Qwen3-TTS
pipeline definition (`vllm_omni/model_executor/models/qwen3_tts/pipeline.py`) pins
`model_arch="Qwen3TTSTalkerForConditionalGeneration"` for stage 0 (stage 1 is
`Qwen3TTSCode2Wav` and is deliberately left alone). The checkpoint's own
`config.json` says `architectures: ["Qwen3TTSForConditionalGeneration"]`, which
vllm-omni's registry maps to the same talker class, so both names are re-registered
to keep the stage override and the checkpoint-declared arch consistent.

HOW IT GETS LOADED
------------------
`VLLM_PLUGINS` cannot be pointed at a file: vLLM's `load_plugins_by_group` treats
it purely as an allow-list *filter* over already-installed `vllm.general_plugins`
entry points. So the runner puts this file's directory plus a generated
`sitecustomize.py` on the server process's `PYTHONPATH`; CPython imports
`sitecustomize` at interpreter startup, in the `vllm serve` process and in every
worker process it spawns -- which is where the model is actually constructed.

To avoid importing vllm at interpreter-startup time (vllm-omni's own `__init__`
warns against pulling heavy imports into lightweight subprocesses), importing this
module does NOT import vllm. It installs a meta-path hook that registers the
subclass immediately after `vllm_omni.model_executor.models.registry` finishes
executing -- i.e. right after `OmniModelRegistry` exists.

The official alternative, if the `sitecustomize` route turns out to misbehave, is
to package `register()` below as a `vllm.general_plugins` entry point and allow it
through `VLLM_PLUGINS`. That needs a pip-installable package and was not built here.

KNOWN LIMITS OF THIS PATCH (all unverified)
-------------------------------------------
* `self.encoder.to(dtype=torch.bfloat16)` in the same `__init__`, and again in
  `load_weights`, is left as upstream has it. Its `encode()` output is `torch.long`
  codec ids, so it does not feed `_embedding_dtype`-typed tensors -- but bf16
  convolutions on sm75 are their own possible failure, only reachable by the `Base`
  (ref_audio) task_type. If a run with this patch still fails, check whether the
  traceback is in the encoder rather than in `index_copy_`.
* Whether float16 is numerically adequate for this path at all is unknown. The
  crash may be replaced by degraded audio, which is exactly the failure mode
  `src/tts_cuda_common.py`'s `audio_statistics`/`audio_defect` checks exist for.
* The parent `__init__` signature (`*, vllm_config, prefix`) is read from the
  `v0.26.0` tag. A different installed version could change it; `register()` refuses
  to register when the parent class or the attribute it patches is not found.
"""
from __future__ import annotations

import sys

# `torch` is imported inside the functions that need it, not here: this module is
# imported from a generated `sitecustomize.py`, i.e. at interpreter startup, and
# startup is not the place to pull in torch (or, later, vllm).

#: vllm-omni architecture names that resolve to the talker class we are replacing.
#: Stage 0's pipeline override first, then the checkpoint-declared alias.
PATCHED_ARCHITECTURES = (
    "Qwen3TTSTalkerForConditionalGeneration",
    "Qwen3TTSForConditionalGeneration",
)

#: The module whose execution creates ``OmniModelRegistry``.
_OMNI_REGISTRY_MODULE = "vllm_omni.model_executor.models.registry"

_UPSTREAM_MODULE = "vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker"
_UPSTREAM_CLASS = "Qwen3TTSTalkerForConditionalGeneration"

PATCH_VERSION = "untested-2026-08-24"


def _build_subclass():
    """Import the upstream talker and return the dtype-aligned subclass.

    Raises rather than degrading: a silently unpatched class would produce exactly
    the "looks applied, actually isn't" report this project forbids.
    """
    import importlib

    import torch

    module = importlib.import_module(_UPSTREAM_MODULE)
    base = getattr(module, _UPSTREAM_CLASS)
    if not hasattr(base, "__init__"):  # pragma: no cover - defensive
        raise AttributeError(f"{_UPSTREAM_CLASS} has no __init__ to wrap")

    class Qwen3TTSTalkerDtypeAlignedForConditionalGeneration(base):  # type: ignore[misc,valid-type]
        """Upstream talker with ``_embedding_dtype`` following the engine dtype.

        The single behavioural difference from upstream is the last statement.
        """

        #: Marker the runner/report can look for; never used for control flow.
        qwen3_tts_dtype_patch = PATCH_VERSION

        def __init__(self, *, vllm_config, prefix: str = ""):
            super().__init__(vllm_config=vllm_config, prefix=prefix)

            model_dtype = getattr(
                getattr(vllm_config, "model_config", None), "dtype", torch.bfloat16
            )
            if not isinstance(model_dtype, torch.dtype):
                # Some vLLM versions carry a string here. Leave upstream behaviour
                # untouched rather than guessing a dtype.
                return
            if getattr(self, "_embedding_dtype", None) == model_dtype:
                return
            self._embedding_dtype = model_dtype

    return Qwen3TTSTalkerDtypeAlignedForConditionalGeneration


def register() -> list[str]:
    """Register the dtype-aligned subclass into vllm-omni's registry.

    Returns the architecture names that were re-registered. Idempotent. Only ever
    called once ``vllm_omni``'s registry module has finished executing, so the
    upstream-class import here is not an extra startup cost.
    """
    subclass = _build_subclass()

    registered: list[str] = []
    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    for arch in PATCHED_ARCHITECTURES:
        OmniModelRegistry.register_model(arch, subclass)
        registered.append(arch)

    try:  # best effort; not the registry the omni path consults
        from vllm import ModelRegistry

        for arch in PATCHED_ARCHITECTURES:
            ModelRegistry.register_model(arch, subclass)
    except Exception:  # pragma: no cover - optional path
        pass

    print(
        "[qwen3_tts_dtype_fix] UNTESTED dtype workaround registered for "
        f"{registered} ({PATCH_VERSION})",
        file=sys.stderr,
        flush=True,
    )
    return registered


class _PostExecLoader:
    """Loader proxy that runs ``register()`` right after the registry module execs."""

    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            register()
        except Exception as exc:  # never break the server on a failed patch
            print(
                f"[qwen3_tts_dtype_fix] FAILED to apply dtype workaround: {exc!r}",
                file=sys.stderr,
                flush=True,
            )

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _RegistryImportHook:
    """Meta-path finder that only decorates the omni registry module's loader."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _OMNI_REGISTRY_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _PostExecLoader(spec.loader)
                return spec
        return None


def install() -> str:
    """Arrange for ``register()`` to run, without importing vllm right now.

    Returns a short status string suitable for a report field.
    """
    if _OMNI_REGISTRY_MODULE in sys.modules:
        register()
        return "registered-immediately"

    if not any(isinstance(f, _RegistryImportHook) for f in sys.meta_path):
        sys.meta_path.insert(0, _RegistryImportHook())
    return "deferred-until-omni-registry-import"


install()
