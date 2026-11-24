"""Both oMLX engines attach a tracer to the model their scheduler runs: VLMBatchedEngine
(vision-language models, through oMLX's VLMModelAdapter) and BatchedEngine (text-only
mlx_lm models: gpt-oss, GLM, DeepSeek, Qwen text MoE)."""
import asyncio
import types

from metalscope.omlx import patches as omlx_patches, patches_mtp


class FakeManager:
    def __init__(self):
        self.attached, self.detached = [], []

    def attach(self, model, name, tokenizer):
        self.attached.append((model, name, tokenizer))

    def detach(self, model):
        self.detached.append(model)


def engine_class(model_attr):
    class Engine:
        def __init__(self):
            self._model_name, self.tokenizer = "m", "tok"
            setattr(self, model_attr, None)

        async def start(self):
            setattr(self, model_attr, object())

        async def stop(self):
            setattr(self, model_attr, None)

        def set_dflash_drafter(self, drafter):          # oMLX's engine pool calls it once the engine has started
            getattr(self, model_attr)._omlx_mtp_decode_enabled = True
            return "set"

    return Engine


def test_text_and_vlm_engines_attach_the_scheduler_model(monkeypatch):
    mgr = FakeManager()
    monkeypatch.setattr(omlx_patches, "MANAGER", mgr)
    text, vlm = engine_class("_model"), engine_class("_adapter")
    omlx_patches.patch_batched_engine(types.SimpleNamespace(BatchedEngine=text))
    omlx_patches.patch_vlm_engine(types.SimpleNamespace(VLMBatchedEngine=vlm))
    for cls in (text, vlm):
        mgr.attached.clear(), mgr.detached.clear()
        engine = cls()
        asyncio.run(engine.start())
        model = engine._model if cls is text else engine._adapter
        asyncio.run(engine.stop())
        assert mgr.attached == [(model, "m", "tok")]
        assert mgr.detached == [model]


def test_an_engine_whose_model_attribute_moved_still_starts_and_stops(monkeypatch):
    mgr = FakeManager()
    monkeypatch.setattr(omlx_patches, "MANAGER", mgr)

    class Engine:                                    # no _model
        _model_name, tokenizer = "m", "tok"

        async def start(self):
            return "started"

        async def stop(self):
            return "stopped"

        def set_dflash_drafter(self, drafter):
            return "set"

    omlx_patches.patch_batched_engine(types.SimpleNamespace(BatchedEngine=Engine))
    engine = Engine()
    assert asyncio.run(engine.start()) == "started" and asyncio.run(engine.stop()) == "stopped"
    assert mgr.attached == mgr.detached == []


def test_an_mtp_model_is_attached_when_the_qwen35_verifier_moved(monkeypatch, caplog):
    """Only the Qwen3.5 verify patch is skipped, and it is reported missing (doctor's FAIL line), not dropped
    silently: every other MTP model (Flash Next) still traces."""
    import sys
    from metalscope.omlx import boot
    monkeypatch.setattr(boot, "_TARGETS", {})            # no oMLX module patched at import here (one may be imported)
    mgr = FakeManager()
    monkeypatch.setattr(omlx_patches, "MANAGER", mgr)
    monkeypatch.setattr(omlx_patches, "APPLIED", set())
    monkeypatch.setattr(omlx_patches, "FAILED", set())
    for name in ("patch_mtp_next", "patch_mtp_priming", "patch_mtp_reconcile"):
        monkeypatch.setattr(patches_mtp, name, lambda: None)
    monkeypatch.setitem(sys.modules, "mlx_vlm.models.qwen3_5.speculative_verifier", None)     # import fails
    vlm = engine_class("_adapter")

    async def start(self):
        self._adapter = types.SimpleNamespace(_omlx_mtp_decode_enabled=True)
    vlm.start = start
    omlx_patches.patch_vlm_engine(types.SimpleNamespace(VLMBatchedEngine=vlm))
    engine = vlm()
    asyncio.run(engine.start())
    assert mgr.attached == [(engine._adapter, "m", "tok")]
    assert omlx_patches.patch_state()["missing"] == ["patch_mtp_verify"]
    assert "patch_mtp_verify not applied" in caplog.text


def test_a_drafter_given_after_the_start_brings_the_mtp_patches(monkeypatch, caplog):
    """oMLX loads a DSpark or DFlash draft (Inkling 2, MiniMax) once the engine runs: the model decodes with MTP only
    from then, so the MTP patches go in then. One that fails is logged and reported missing; the drafter stays set."""
    from metalscope.omlx import boot
    monkeypatch.setattr(boot, "_TARGETS", {})
    monkeypatch.setattr(omlx_patches, "MANAGER", FakeManager())
    monkeypatch.setattr(omlx_patches, "APPLIED", set())
    monkeypatch.setattr(omlx_patches, "FAILED", set())
    applied = []
    for name in ("patch_mtp_next", "patch_mtp_verify", "patch_mtp_priming", "patch_mtp_reconcile"):
        monkeypatch.setattr(patches_mtp, name, lambda name=name: applied.append(name))
    vlm = engine_class("_adapter")

    async def start(self):
        self._adapter = types.SimpleNamespace()
    vlm.start = start
    omlx_patches.patch_vlm_engine(types.SimpleNamespace(VLMBatchedEngine=vlm))
    engine = vlm()
    asyncio.run(engine.start())
    assert applied == []
    assert engine.set_dflash_drafter("draft") == "set"
    assert applied == ["patch_mtp_next", "patch_mtp_verify", "patch_mtp_priming", "patch_mtp_reconcile"]

    def moved():
        raise AttributeError("GenerationBatch.next moved")
    monkeypatch.setattr(patches_mtp, "patch_mtp_next", moved)
    assert engine.set_dflash_drafter("draft") == "set"
    assert omlx_patches.patch_state()["missing"] == ["patch_mtp_next"] and "MTP patches failed" in caplog.text


def test_an_mtp_patch_that_fails_at_attach_is_reported_missing(monkeypatch, caplog):
    """A core MTP patch that raises leaves the model untraced (fail closed); status and doctor must say which."""
    from metalscope.omlx import boot
    monkeypatch.setattr(boot, "_TARGETS", {})
    mgr = FakeManager()
    monkeypatch.setattr(omlx_patches, "MANAGER", mgr)
    monkeypatch.setattr(omlx_patches, "APPLIED", set())
    monkeypatch.setattr(omlx_patches, "FAILED", set())

    def moved():
        raise AttributeError("GenerationBatch.next moved")
    monkeypatch.setattr(patches_mtp, "patch_mtp_next", moved)
    vlm = engine_class("_adapter")

    async def start(self):
        self._adapter = types.SimpleNamespace(_omlx_mtp_decode_enabled=True)
    vlm.start = start
    omlx_patches.patch_vlm_engine(types.SimpleNamespace(VLMBatchedEngine=vlm))
    asyncio.run(vlm().start())
    assert mgr.attached == []
    assert omlx_patches.patch_state()["missing"] == ["patch_mtp_next"]
    assert "attach failed" in caplog.text


def test_batched_engine_is_patched_on_import():
    from metalscope.omlx import boot
    assert boot._TARGETS["omlx.engine.batched"] == "patch_batched_engine"
