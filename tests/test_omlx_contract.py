"""Every oMLX name omlx/patches.py, omlx/patches_mtp.py and omlx/backend.py wrap, call or read (and what the adapters, mtp.py and lookup.py read of the model
code oMLX bundles; Inkling 2's only where the app has 0.8.3's or 0.8.4's, INKLING), checked against the installed app's source: it
exists, it is sync or async as wrapped, and it takes the parameters the wrapper names. AST only (nothing of oMLX is imported);
skipped without the app. Another oMLX: ``OMLX_APP=<bundle>`` (a kept older app) or ``OMLX_SOURCE=<checkout>`` (a
source tree, the folder holding ``omlx/``; what oMLX's bundled packages define is then skipped)."""
import ast
import os
from pathlib import Path

import pytest

from metalscope.omlx import install

RES = Path(os.environ.get("OMLX_SOURCE") or Path(os.environ.get("OMLX_APP") or install.DEFAULT_APP) / "Contents/Resources")
pytestmark = pytest.mark.skipif(not (RES / "omlx").is_dir(), reason=f"no oMLX at {RES}")
SITE = "Python/framework-mlx-base/lib/python*/site-packages/"
Q4 = "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/qwen4_exp/"
PATCHES = [Path(install.__file__).with_name(f) for f in ("patches.py", "patches_mtp.py")]
# oMLX 0.8.4 on: DeepSeek V4's and Inkling 2's expert offload share one slot cache, its flag and slots defined there.
SLOT_CACHE = (RES / "omlx/patches/moe_slot_cache.py").is_file()

# (file under Contents/Resources, class, function, async, the leading parameters the wrapper names,
#  parameters it passes by keyword)
WRAPPED = [
    ("omlx/engine/vlm.py", "VLMBatchedEngine", "start", True, ["self"], []),
    ("omlx/engine/vlm.py", "VLMBatchedEngine", "stop", True, ["self"], []),
    ("omlx/engine/batched.py", "BatchedEngine", "start", True, ["self"], []),
    ("omlx/engine/batched.py", "BatchedEngine", "stop", True, ["self"], []),
    ("omlx/engine/vlm.py", "VLMBatchedEngine", "set_dflash_drafter", False, ["self", "drafter"], []),
    ("omlx/engine/batched.py", "BatchedEngine", "set_dflash_drafter", False, ["self", "drafter"], []),
    ("omlx/engine_core.py", "EngineCore", "add_request", True, ["self", "prompt", "sampling_params", "request_id"],
     ["skip_cache_store"]),
    ("omlx/engine_core.py", "EngineCore", "_cleanup_request", False, ["self", "request_id"], []),
    ("omlx/scheduler.py", "Scheduler", "step", False, ["self"], []),
    ("omlx/scheduler.py", "Scheduler", "_do_external_prefill", False, ["self", "request"], []),
    ("omlx/scheduler.py", "Scheduler", "_run_prefill_chunks", False, ["self", "plans"], []),
    ("omlx/scheduler.py", "Scheduler", "add_request", False, ["self", "request"], []),
    ("omlx/scheduler.py", "Scheduler", "_start_helper_prefill", False, ["self", "request"], []),
    ("omlx/scheduler.py", "Scheduler", "_cleanup_finished", False, ["self", "finished_ids"], []),
    ("omlx/scheduler.py", "Scheduler", "_do_abort_request", False, ["self", "request_id"], []),
    ("omlx/scheduler.py", "Scheduler", "fail_all_requests", False, ["self"], []),
    (SITE + "mlx_lm/generate.py", "GenerationBatch", "_step", False, ["self"], []),
    (SITE + "mlx_lm/generate.py", "GenerationBatch", "next", False, ["self"], []),
    (SITE + "mlx_vlm/models/qwen3_5/speculative_verifier.py", "Qwen3_5BatchInvariantForward", "_layer", False,
     ["self", "layer", "hidden"], []),
    ("omlx/cache/prefix_cache.py", "BlockAwarePrefixCache", "fetch_cache", False, ["self", "request_id", "tokens"], []),
    ("omlx/cache/prefix_cache.py", "BlockAwarePrefixCache", "store_cache", False, ["self", "request_id"], []),
    ("omlx/cache/prefix_cache.py", "BlockAwarePrefixCache", "fetch_exact_prefix", False, ["self", "request_id"], []),
    ("omlx/patches/mlx_lm_mtp/prompt_priming.py", None, "_publish_boundary_candidate", False, ["ctx"], []),
    ("omlx/patches/mlx_lm_mtp/batch_generator.py", None, "_reconcile_mtp_to_standard", False, ["gen_batch", "state"], []),
    ("omlx/engine_pool.py", "EnginePool", "_load_engine", True, ["self", "model_id"], ["runtime_settings"]),
    ("omlx/engine_pool.py", "EnginePool", "get_engine", True, ["self", "model_id"], ["runtime_settings"]),
    ("omlx/engine_pool.py", "EnginePool", "_unload_engine", True, ["self", "model_id"], []),
    (SITE + "uvicorn/server.py", "Server", "handle_exit", False, ["self", "sig", "frame"], []),
]
# Called, not wrapped.
CALLED = [
    ("omlx/scheduler.py", "Scheduler", "_get_stop_tokens", False, ["self"], []),
    ("omlx/scheduler.py", "Scheduler", "_get_detokenizer", False, ["self", "request_id"], []),
    ("omlx/scheduler.py", "Scheduler", "_cleanup_detokenizer", False, ["self", "request_id"], []),
    ("omlx/scheduler.py", "Scheduler", "_get_output_parser_session", False, ["self", "request_id"], []),
    ("omlx/scheduler.py", "Scheduler", "_cleanup_output_parser_session", False, ["self", "request_id"], []),
    (SITE + "mlx_lm/tokenizer_utils.py", "StreamingDetokenizer", "add_token", False, ["self", "token"], []),
    (SITE + "mlx_lm/tokenizer_utils.py", "StreamingDetokenizer", "last_segment", False, ["self"], []),
    ("omlx/engine_pool.py", "EnginePool", "get_entry", False, ["self", "model_id"], []),
    ("omlx/engine_pool.py", "EnginePool", "get_model_ids", False, ["self"], []),
    ("omlx/engine_pool.py", "EnginePool", "_entry_runtime_resident_size", False, ["self", "entry", "runtime_settings"], []),
    ("omlx/engine_pool.py", "EnginePool", "acquire", True, ["self", "model_id"], []),     # lookup_lp's lease
    ("omlx/memory_monitor.py", "MemoryMonitor", "estimate_prefill_peak_bytes", False,     # its price (_room_for)
     ["self", "new_tokens", "chunk_size"], []),
    ("omlx/model_settings.py", "ModelSettingsManager", "get_settings", False, ["self", "model_id"], []),
    ("omlx/exceptions.py", "ModelUnavailableError", "__init__", False, ["self", "model_id", "message"], []),
    ("omlx/server.py", None, "resolve_model_id", False, ["model_id"], []),
    ("omlx/server.py", None, "get_server_state", False, [], []),
    ("omlx/server.py", None, "verify_api_key", True, ["request", "credentials"], []),      # check_key, before /scope
]
# Attributes read (and one flag set): the text that defines each, in the file that should.
READ = [
    ("omlx/scheduler.py", "self.uid_to_request_id"), ("omlx/scheduler.py", "self.requests"),
    ("omlx/scheduler.py", "self.model = model"), ("omlx/scheduler.py", "self._output_parser_factory"),
    ("omlx/scheduler.py", "state.request"), ("omlx/scheduler.py", "class _PrefillChunkPlan"),
    ("omlx/scheduler.py", "request.packed_prefill_excluded"), ("omlx/scheduler.py", '"skip_cache_store"'),
    ("omlx/request.py", "packed_prefill_excluded: bool"), ("omlx/request.py", "skip_cache_store: bool"),
    ("omlx/request.py", "ignore_eos: bool"), ("omlx/request.py", "stop_token_ids: Optional[List[int]]"),
    ("omlx/request.py", "prompt_token_ids: Optional[List[int]]"), ("omlx/request.py", "output_token_ids: List[int]"),
    ("omlx/request.py", "request_id: str"), ("omlx/request.py", "sampling_params: SamplingParams"),
    # The edit (omlx/edit.py): oMLX's completion route, a prompt of token ids fed as given, a requeued reply; the
    # request's text from its own detokenizer (seeded by _continued), the route's synthetic <think> strip, the
    # client's disconnect signal the sub-requests share, and /v1/models checking the key as the route does.
    ("omlx/server.py", '@app.post("/v1/completions")'), ("omlx/engine_core.py", "prompt: Union[str, List[int]]"),
    ("omlx/scheduler.py", "request.prompt_token_ids = list(request.prompt)"),
    ("omlx/request.py", "requeued_output_tokens: int"), (SITE + "mlx_lm/tokenizer_utils.py", "    def text(self)"),
    ("omlx/scheduler.py", "detokenizer = self._get_detokenizer(request_id)"),
    ("omlx/scheduler.py", "new_text = detokenizer.last_segment"),
    # The same through a protocol parser (patches._continued, _refuses_edit, _forget, add_request): the factory's kind,
    # the scheduler's sessions by request id, each token's stream text, and the think prefix the edit marks sent.
    ("omlx/adapter/output_parser.py", "    kind: str\n    create_session:"),
    ("omlx/adapter/output_parser.py", '    stream_text: str = ""\n'),
    ("omlx/scheduler.py", "self._output_parser_sessions: dict[str, OutputParserSession] = {}"),
    ("omlx/scheduler.py", "parser_session = self._get_output_parser_session(request_id)"),
    ("omlx/scheduler.py", "parser_result = parser_session.process_token(response.token)"),
    ("omlx/scheduler.py", "new_text = parser_result.stream_text"),
    ("omlx/request.py", "think_prefix_sent: bool"),
    ("omlx/scheduler.py", 'if not getattr(request, "think_prefix_sent", False):'),
    ("omlx/server.py", "chunk_text = _strip_synthetic_think_prefix(chunk_text, think_tag)"),
    ("omlx/server.py", '_DISCONNECT_SIGNAL_SCOPE_KEY = "omlx.client_disconnect_signal"'),
    ("omlx/server.py", "async def list_models(_: bool = Depends(verify_inference_api_key))"),
    ("omlx/scheduler.py", "self._output_parser_factory.stop_token_ids"),
    ("omlx/engine_core.py", "self.model = model"), ("omlx/engine_core.py", "self.scheduler = Scheduler("),
    ("omlx/engine/batched.py", "self._model_name"), ("omlx/engine/batched.py", "self._model ="),
    ("omlx/engine/batched.py", "def tokenizer(self)"), ("omlx/engine/vlm.py", "def tokenizer(self)"),
    ("omlx/engine/vlm.py", "self._adapter ="), ("omlx/engine/vlm.py", "self._model_name"),
    ("omlx/engine_pool.py", "self._settings_manager"), ("omlx/engine_pool.py", "model_path: str"),
    # Each model's flags (status ``chat``): its family (backend.chat_formats; the admin API's for `metalscope
    # status`) and, loaded, its scheduler (backend._scheduler: entry.engine._engine.engine.scheduler).
    ("omlx/engine_pool.py", "config_model_type: str"), ("omlx/engine_pool.py", "    engine: ("), ("omlx/admin/routes.py", '"config_model_type": model_info.get('),
    ("omlx/engine/batched.py", "self._engine = AsyncEngineCore("), ("omlx/engine/vlm.py", "self._engine = AsyncEngineCore("),
    ("omlx/engine_core.py", "self.engine = EngineCore("),
    ("omlx/server.py", "engine_pool: Optional[EnginePool]"),
    ("omlx/server.py", "\nsecurity = HTTPBearer(auto_error=False)"), ("omlx/server.py", "    api_key: Optional[str] = None"),
    ("omlx/server.py", "app.add_middleware(RequestBodySizeLimitMiddleware)"),
    ("omlx/patches/mlx_lm_mtp/prompt_priming.py", "ctx.request_id"),
    (SITE + "mlx_lm/generate.py", "self.uids"), (SITE + "mlx_lm/generate.py", "self.model = model"),
    (SITE + "starlette/applications.py", "self.user_middleware"),
    (SITE + "starlette/applications.py", "self.middleware_stack"),
    # The Prefill Accelerator gives a request its helper only there, called through the class (our wrapper).
    ("omlx/scheduler.py", "state.helper = session.start("), ("omlx/scheduler.py", "Scheduler._start_helper_prefill("),
    ("omlx/engine_core.py", "self._mlx_executor = "),                               # lookup_lp runs on it
    ("omlx/scheduler.py", "self.memory_monitor: MemoryMonitor | None = MemoryMonitor("),     # _room_for prices with it
    # MlxEngine.lookup_chunk: an offloaded MoE module's ``cache`` (flagged), its ``capacity`` slots, and the exact path
    # a call takes while tokens * k fits them (lookup_lp's chunks stay on it).
    *((f"omlx/patches/{f}", t) for f, made in (("moe_expert_offload.py", "ExpertCache"),
                                               ("glm_moe_dsa/moe_offload.py", "_SlotCache"),
                                               *([] if SLOT_CACHE else [("deepseek_v4/moe_offload.py", "_SlotCache")]))
      for t in ("moe_offload_cache = True", f"self.cache = {made}(", "self.capacity = ", "n_tok * k <= c.capacity")),
    *(("omlx/patches/moe_slot_cache.py", t) for t in ("moe_offload_cache = True", "self.capacity = ") if SLOT_CACHE),
    *((f"omlx/patches/{f}", t) for f in ("deepseek_v4/moe_offload.py", "inkling2/moe_offload.py") if SLOT_CACHE
      for t in ("self.cache = SlotCache(", "n_tok * k <= c.capacity")),
    ("omlx/patches/moe_expert_offload.py", "self.floor = "),        # top-k unknown: capacity // floor
    ("omlx/engine_pool.py", "executor, release, request_id, shortfall"),  # slots given back on the MLX executor
    # mtp.mtp_decode's flag, on the models metalscope was run on; MTP windows are placed by each row's cache offset,
    # which the deferred realignment of a ragged rollback still lowers as the stock one does.
    (Q4 + "language.py", "self._omlx_mtp_decode_enabled = True"),
    ("omlx/patches/mlx_vlm_mtp/qwen35_moe_vlm_runtime.py", "self._omlx_mtp_decode_enabled = bool("),
    ("omlx/patches/mlx_vlm_mtp/qwen35_vlm_runtime.py", "self._omlx_mtp_decode_enabled = bool("),
    ("omlx/patches/qwen35_deferred_realign.py", "self.offset = self.offset - self._right_padding"),
    # The longest verify window: a context copy's drafts + 1 (test_mtp_windows_hold_omlxs_longest_verify_window).
    ("omlx/patches/mlx_lm_mtp/context_copy.py", "\nMAX_COPY = 15\n"),
    # jlens.fit's prefix_cache / restore: an ArraysCache's state is its live slot list (with its padding and lengths),
    # which a snapshot copies and the setter installs.
    (SITE + "mlx_lm/models/cache.py", "        return self.cache, self.left_padding, self.lengths\n"),
    (SITE + "mlx_lm/models/cache.py", "        self.cache, self.left_padding, self.lengths = v\n"),
    # A CacheList's offset is its first cache's (window_starts): GLM-5.3's sparse-attention layers and Inkling 1's.
    ("omlx/patches/mlx_vlm_glm5_next_compat/vendor/mlx_vlm/models/glm5_next/language.py",
     "KVCache(), PoolingCache(layer.self_attn.indexer.index_kpool)"),
    ("omlx/patches/mlx_vlm_inkling_compat/vendor/mlx_vlm/models/inkling/language.py",
     "return [CacheList(KVCache(), ArraysCache(4)) for _ in self.model.layers]"),
    (SITE + "mlx_vlm/models/cache.py", "        self.caches = caches"), (SITE + "mlx_lm/models/cache.py", "        self.caches = caches"),
    # patch_mtp_verify: the Qwen3.5 verify loop computes every layer through _layer (never calls it), and oMLX's
    # own wrapper of _layer, applied before the model loads, is the one ours wraps.
    (SITE + "mlx_vlm/models/qwen3_5/speculative_verifier.py", "hidden = self._layer(\n                layer,"),
    ("omlx/patches/qwen35_verify_qmm.py", "Qwen3_5BatchInvariantForward._layer = _layer"),
    # adapters/qwen4_exp.py on oMLX's vendored Flash Next (the live copy: ahead of mlx_vlm's own on its path): the
    # mixer, the MoE input under MTP verify, the deferred write's tuple, the lookup (lookup.py) and the flags fit_setup
    # turns the fused kernels off with.
    ("omlx/patches/mlx_vlm_qwen4_exp_compat/__init__.py", '_append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")'),
    (Q4 + "language.py", "self.hyper_connection_mixer = Qwen4ExpGatedResidual("), (Q4 + "config.py", "hc_count: int"),
    (Q4 + "language.py", "self.mlp_hyper_connection = "), (Q4 + "language.py", "self.ple = "),
    (Q4 + "language.py", "branch = _VERIFIER._feed_forward(mlp, mixed) if target_verify else mlp(mixed)"),
    (Q4 + "language.py", "return hyper_input, branch, injection_weights"),
    (Q4 + "hc_fused.py", "\n_DISABLED = False"), (Q4 + "hc_fused.py", "return not (_DISABLED or _WRITE_DISABLED)"),
    (Q4 + "attn_fused.py", "\n_DISABLED = False"), (Q4 + "language.py", "\n_DEPTHWISE_CONV_STATE = {"),
    ("omlx/patches/m5_gather_qmm.py", "\n_nax_host: bool"),
    # jlens/fit.py: oMLX's DeepSeek V4 ends a forward evaluating its cache through mlx.core's async_eval, which a fit's
    # forward makes a no-op while mx.vjp builds its graph.
    ("omlx/patches/deepseek_v4/deepseek_v4_model.py", "        mx.async_eval(*cache_arrays)\n"),
    ("omlx/patches/deepseek_v4/deepseek_v4_model.py", "        _materialize_cache_arrays(cache)\n"),
    # oQh checkpoints: the norms in front of rotated linears still return the plain residual (the routers and the
    # MTP router hook read it), and the MoE block keeps its class (routing.py's rule): only its experts are rebased.
    ("omlx/oqh/runtime.py", "        return plain\n"), ("omlx/oqh/runtime.py", "readers = _install_mlp(switch)"),
]

# adapters/inkling2.py on oMLX's own Inkling 2 (oMLX 0.8.3 on; checked only where the app has 0.8.3's layout, its
# cache.py: an older oMLX's Inkling 2 is refused, inkling2.matches). 0.8.4's (INK084: its fused kernels, fused.py) moves
# the MoE's input and the router's into ``_post_attn`` and adds LoopCD's visits; INK_083 and INK_084 pin each layout.
INK = "omlx/patches/inkling2/"
INKLING = (RES / INK / "cache.py").is_file()
INK084 = INKLING and (RES / INK / "fused.py").is_file()
INK_CALLED = [
    (INK + "model.py", "Inkling2Layer", "__call__", False, ["self", "h", "blocks"], ["cache", "bank", "tau"]),  # hooked
    (INK + "model.py", "Inkling2Layer", "_post", False, ["self", "out", "gate", "h", "mlp_blocks"], []),   # run plain
    (INK + "model.py", "Inkling2MoE", "__call__", False, ["self", "x"], []),                                 # hooked
    (INK + "model.py", "Model", "logits_from_hidden", False, ["self", "h"], []),                             # the lens
    (INK + "kernels.py", None, "route_topk", False, ["logits", "bias", "top_k", "route_scale"], []),         # the rule
    (INK + "model.py", None, "_row_matmul", False, ["x", "w"], []),                     # the rule's logits, as route's
    # The J-lens: the state's mix (the head's ``final_input``) and the sums' unit-RMS, which a fit freezes; each visit's
    # cache and the full-attention visits' shared log scaling, which a fit's loop takes as the model's does.
    (INK + "model.py", "AttnResQuery", "__call__", False, ["self", "blocks", "partial_n"], []),
    (INK + "model.py", None, "_unit_rms", False, ["x", "eps"], []),
    (INK + "model.py", "Inkling2TextModel", "visit_caches", False, ["self", "cache"], []),
    # jlens.fit.prefix_cache calls the text model with these by keyword (0.8.4: no defaults).
    (INK + "model.py", "Inkling2TextModel", "__call__", False, ["self", "inputs", "cache", "input_embeddings", "capture"],
     ["cache", "input_embeddings", "capture"]),
    (INK + "model.py", "Inkling2Attention", "log_tau", False, ["self", "kv", "length"], []),
    # An edit seeds Inkling 2's parser session token by token (patches._continued).
    ("omlx/adapter/output_parser.py", "InklingOutputParserSession", "process_token", False, ["self", "token_id"], []),
    # 0.8.4: the second half's first part, run while recording (inkling2._eager_while_recording), the MoE as it calls
    # it, and LoopCD's visits after the schedule's (inkling2._visits reads _reference_visits).
    *([(INK + "model.py", "Inkling2Layer", "_post_attn", False, ["self", "out", "gate", "h", "mlp_blocks"], []),
       (INK + "model.py", "Inkling2MoE", "unreduced", False, ["self", "x", "x32"], []),
       (INK + "model.py", "Inkling2MoE", "route", False, ["self", "x", "x32"], []),
       (INK + "model.py", "Inkling2TextModel", "_visit_order", False, ["self"], [])] if INK084 else []),
] if INKLING else []
INK_READ = [
    # The text model (matches, its layer visits, the vocabulary, the lens's norm) and how each visit calls its layer.
    (INK + "model.py", "class Inkling2TextModel(nn.Module)"), (INK + "model.py", "self.model = Inkling2TextModel("),
    (INK + "model.py", "        self.layers = [\n            Inkling2Layer("),
    (INK + "model.py", "        return h\n\n\nclass Inkling2TextModel"),                 # a layer returns its sum, bare
    (INK + "model.py", "self._schedule = tuple(args.layer_schedule())"),
    (INK + "model.py", "self.embed = nn.Embedding(args.vocab_size"), (INK + "model.py", "self.norm = nn.RMSNorm("),
    # The head's norm reads float32 through a float32 copy of its weight: what readout's norm(h.astype(float32)) gives.
    (INK + "model.py", "out = _norm(self.norm, x).astype(h.dtype)"),
    (INK + "model.py", "        copy = weight.astype(mx.float32)\n"),
    (INK + "model.py", "    if x.dtype == mx.float32 and held is not None and held[0] is norm.weight:\n"
                       "        return mx.fast.rms_norm(x, held[1], norm.eps)\n"),
    (INK + "model.py", "            h = layer(\n                h,\n                blocks,\n                cache=c,\n"
                       "                bank=bank if loop_pass else None,\n"
                       "                tau=None if layer.sliding else tau,\n            )\n"),
    # The MoE: its block, its router's tensors and rule, called on [B, L, H] inside the compiled second half; route
    # takes the rows flattened, as the rule's logits do (the layouts' own pins below). Row-exact verify rows take the
    # router's matmul a row at a time (_row_matmul, read when the rule's logits are made).
    (INK + "model.py", "self.mlp = Inkling2MoE("), (INK + "model.py", "self.top_k = args.num_experts_per_tok"),
    (INK + "model.py", "self.route_scale = args.route_scale"), (INK + "model.py", "self.router_weight = mx.zeros("),
    (INK + "model.py", "self.router_bias = mx.zeros("),
    (INK + "model.py", "    if rows > 1 and is_row_exact_armed():\n"), (INK + "model.py", "    return x @ w\n"),
    (INK + "model.py", "logits, self.router_bias, self.top_k, self.route_scale"),
    (INK + "model.py", "self._post_compiled = mx.compile("), (INK + "model.py", "self._post_exact = mx.compile("),
    (INK + "model.py", "        if compiled and length > 1 and is_row_exact_armed():\n"
                       "            pre, post = self._pre_exact, self._post_exact\n        elif compiled:\n"
                       "            pre, post = self._pre_compiled, self._post_compiled\n        else:\n"
                       "            pre, post = self._pre, self._post\n"),
    (INK + "kernels.py", "inds = mx.argpartition(-(scores + bias), kth=top_k - 1, axis=-1)[..., :top_k]"),
    (INK + "kernels.py", "w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20) * route_scale"),
    (INK + "kernels.py", "key[j] = s + bias[e];"),
    (INK + "kernels.py", "float norm = route_scale[0] / (total + 1e-20f);"),
    # Served by the VLM engine: generic.text_model reaches the text model through the wrapper oMLX's adapter keeps.
    ("omlx/models/vlm.py", "self._vlm_model = vlm_model"),
    (INK + "vlm.py", "def language_model(self) -> _LanguageModel:"),
    (INK + "vlm.py", "        return getattr(self._model, name)"),
    # Its output parser's kind, the one patches.SEEDED_PARSERS names.
    ("omlx/adapter/output_parser.py", 'kind="inkling",'),
    # MTP (the bundled DSpark draft): the flag mtp.mtp_decode reads, and windows placed by the first full-attention
    # visit's KVCache (a sliding visit's state, an ArraysCache, has no offset).
    ("omlx/speculative/dflash/__init__.py", "language_model._omlx_mtp_decode_enabled = True"),
    (INK + "model.py", "                cache.append(windows.ArraysCache(window=window))\n"),
    (INK + "model.py", "                cache += [KVCache(), ArraysCache(4)]\n"),
    (INK + "cache.py", "class ArraysCache(_ArraysCache):"),
    # A restored sliding state (jlens.fit.restore): its first append takes up the buffer the prefix left while the slot
    # is still the snapshot's array, else starts one from it; the append writes in place once the slots are dropped
    # (test_each_pass_from_a_restored_prefix_cache_is_the_models_own).
    (INK + "cache.py", "    if buffer is not None and buffer.view is keys:\n        return buffer\n"),
    (INK + "cache.py", "    state.cache[KEYS] = state.cache[VALUES] = buffer.view = None\n"
                       "    buffer.keys[:, :, buffer.n : buffer.n + length] = keys\n"),
    # The J-lens. The state after a visit: a block's first layer appends its incoming sum, unit-RMS, to ``blocks`` in
    # place; later passes restart from the prelude and read the pass before as their bank; the head reads its mix of
    # the blocks and the sum. inkling2.differentiable runs this loop itself, inkling2.lens_sources counts it from the config.
    (INK + "model.py", "self.is_boundary = layer_idx % args.n_layers_per_attn_res_block == 0"),
    (INK + "model.py", "        h_n = _unit_rms(h.astype(mx.float32))\n"),
    # A fit's restart hands a block's first layer only the gradient of its incoming sum's unit-RMS (``carried``): that
    # layer reads the sum through h_n alone and starts its own sum from the attention output.
    (INK + "model.py", "            mixed = self.attn_input(blocks, h_n)\n"),
    (INK + "model.py", "            residual=None if self.is_boundary else h,\n"),
    # ``frozen`` re-implements _unit_rms with its scale held fixed for hidden-wide values and tells them apart by width:
    # the attention's own call on head-wide values keeps the real one; the messages' mix takes the frozen one.
    (INK + "model.py", "    return mx.fast.rms_norm(x, None, eps)\n"),
    (INK + "model.py", "v = _unit_rms(v.reshape(b, length, self.n_kv, self.head_dim))"),
    (INK + "model.py", "z = mx.stack([_unit_rms(x), _unit_rms(message)], axis=1)"),
    (INK + "model.py", "y = self.mlp_input(mlp_blocks, _unit_rms(residual))[:, :, 0]"),
    (INK + "model.py", "        if self.is_boundary:\n            blocks.append(h_n)\n"),
    (INK + "model.py", "messages = self.loop_message(*bank) if bank is not None else None"),
    (INK + "model.py", "self.final_input = AttnResQuery(args.hidden_size, 1)"),
    (INK + "model.py", "self._core_start = args.loop_core.start if args.loop_config else None"),
    (INK + "model.py", "            h = self.embed_norm(self.embed(inputs))\n"),
    (INK + "model.py", "            if not layer.sliding and tau is None:\n"),
    (INK + "model.py", "tau = layer.attn.log_tau(c[0] if c is not None else None, h.shape[1])"),
    (INK + "model.py", "                    blocks[len(prelude[1]) + 1 :],\n                    _unit_rms(h.astype(mx.float32)),\n"),
    (INK + "model.py", "                h, blocks = prelude[0], list(prelude[1])\n"),
    (INK + "model.py", "x = self.final_input(blocks, _unit_rms(h.astype(mx.float32)))[:, :, 0]"),
    (INK + "model.py", "    def from_dict(cls, params: dict) -> TextArgs:"), (INK + "model.py", "    hidden_size: int = "),
    (INK + "model.py", "    n_layers_per_attn_res_block: int = 6\n"),
    (INK + "model.py", 'range(loop["core_start"], loop["core_end"]) if loop else range(0)'),
    (INK + "model.py", 'passes = int(self.loop_config["num_passes"])'),
    (INK + "model.py", '        if args.loop_config and "loop_num_passes" in params:\n'),
    (INK + "model.py", '                "num_passes": int(params["loop_num_passes"]),\n'),
    (INK + "model.py", "            [(i, 0) for i in range(core.start)]\n            + [(i, p) for p in range(passes) for i in core]\n"
                       "            + [(i, 0) for i in range(core.stop, n)]\n"),
    # What a fit swaps: the layers' compiled halves (no hooks, no gradients through the threshold) and the kernels'
    # mlx.core, whose default device decides each Metal kernel's plain path (the kernels counted below).
    # 16: the tests' 5-token step runs the halves compiled, their 20-token chunk plain; Inkling 2's verify rows are
    # capped by it (test_mtp_windows_hold_omlxs_longest_verify_window).
    (INK + "model.py", "\n_COMPILED_MAX_TOKENS = 16\n"), (INK + "model.py", "compiled = length <= _COMPILED_MAX_TOKENS"),
    (INK + "model.py", "    _omlx_mtp_max_verify_rows = _COMPILED_MAX_TOKENS\n"),
    (INK + "kernels.py", "\nimport mlx.core as mx\n"),
    (INK + "kernels.py", "    if mx.default_device() != mx.gpu:\n        src = mx.stack([*blocks, partial_n], axis=-2)\n"),
    (INK + "kernels.py", "if length < _FLASH_MIN_QUERIES or mx.default_device() != mx.gpu:"),
    (INK + "kernels.py", "if mx.default_device() != mx.gpu or experts % 32:"),
    (INK + "kernels.py", "    if mx.default_device() != mx.gpu:\n        return short_conv_reference("),
    (INK + "kernels.py", "    if mx.default_device() != mx.gpu:\n        return dynamic_conv_reference("),
    # The experts go through mx.gather_qmm, which oMLX's engine reroutes to a Metal kernel while _on_nax_host() (the
    # switch pinned above for Flash Next): a fit turns it off.
    (INK + "model.py", "        return mx.gather_qmm(\n"),
    ("omlx/patches/m5_gather_qmm.py", "    if _nax_host is None:\n"), ("omlx/patches/m5_gather_qmm.py", "    return _nax_host\n"),
    ("omlx/patches/m5_gather_qmm.py", "    mx.gather_qmm = _gather_qmm_rerouted\n"),
] if INKLING else []
INK_083 = [
    (INK + "model.py", "    _omlx_drafter_single_row = True"),                           # window_starts: one row
    # The MoE's input: x, called as the block (the router hook), the rule's logits from x in float32.
    (INK + "model.py", "        flat = x.reshape(-1, shape[-1])\n        inds, w = self.route(flat)\n"),
    (INK + "model.py", "logits = _row_matmul(x.astype(mx.float32), self.router_weight.T)"),
    (INK + "model.py", "m = self.mlp(_norm(self.mlp_norm, y).astype(h.dtype))"),
    # The loop differentiable runs, as the model's.
    (INK + "model.py", "visits = [None] * self.num_visits if cache is None else self.visit_caches(cache)"),
    (INK + "model.py", "        blocks: list = []\n        prelude = bank = tau = None\n"),
    (INK + "model.py", "for visit, ((idx, loop_pass), c) in enumerate(zip(self._schedule, visits)):"),
    (INK + "model.py", "            if idx == self._core_start and prelude is None:\n                prelude = (h, list(blocks))\n"
                       "            elif idx == self._core_start:\n"),
    (INK + "model.py", "\nfrom . import kernels\n"),
] if INKLING and not INK084 else []
INK_084 = [
    # MTP verify on several rows: rows that accept different counts are realigned, each row's offset lowered (the
    # deferred realignment pinned above), so window_starts places each row's window.
    (INK + "model.py", "            elif any(extra):\n                kv.prepare(right_padding=extra)\n                kv.finalize()\n"),
    # The second half calls _post_attn, then the MoE's unreduced on its input and the router's float32 copy (the fused
    # mix and norm's, else None: route casts x), as _eager_while_recording hands the router hook; the plain mix is
    # mlp_input's call on the sum's unit-RMS, as it hands _Reads.
    (INK + "model.py", "        residual, x, x32, s2 = self._post_attn(out, gate, h, mlp_blocks, s2, messages)\n"
                       "        z, zw = self.mlp.unreduced(x, x32)\n"),
    (INK + "model.py", "        return residual, x, x32, s2\n"),
    (INK + "model.py", "            x, x32 = _part(fused[0], 0), _part(fused[1], 0)\n"),
    (INK + "model.py", "                x, x32 = _norm(self.mlp_norm, y).astype(h.dtype), None\n"),
    (INK + "model.py", "        flat = x.reshape(-1, shape[-1])\n        if x32 is not None:\n"
                       "            x32 = x32.reshape(-1, shape[-1])\n        inds, w = self.route(flat, x32)\n"),
    (INK + "model.py", "        if x32 is None:\n            x32 = x.astype(mx.float32)\n"
                       "        logits = _row_matmul(x32, self.router_weight.T)\n"),
    # The loop differentiable runs, as the model's; LoopCD's visits come after the schedule's.
    (INK + "model.py", "        order = self._visit_order()\n"
                       "        visits = [None] * len(order) if cache is None else self.visit_caches(cache)\n"),
    (INK + "model.py", "        blocks: list = []\n        prelude = bank = tau = first_pass = None\n"),
    (INK + "model.py", "for visit, ((idx, loop_pass), c) in enumerate(zip(order, visits)):"),
    (INK + "model.py", "            if visit == self.num_visits:\n                out = self._readout(h, blocks)\n"
                       "                h, blocks = first_pass\n"
                       "            elif idx == self._core_start and prelude is None:\n                prelude = (h, list(blocks))\n"
                       "            elif idx == self._core_start:\n"),
    (INK + "model.py", "        return self._schedule[: self.num_visits] + self._reference_visits\n"),
    (INK + "model.py", "        self.model._reference_visits = tuple((idx, 0) for idx in coda)\n"),
    # The modules a fit switches (inkling2._fit_paths), each kernel behind a default-device check, and the NAX banded
    # attention, which asks none (the fit's stand-in returns None, as unsupported).
    (INK + "model.py", "\nfrom . import expert_mma, fused, kernels, loopcd, verify_experts\n"),
    *((INK + f, "\nimport mlx.core as mx\n") for f in ("fused.py", "verify_experts.py", "expert_mma.py")),
    (INK + "kernels.py", "        mx.default_device() != mx.gpu\n        or threads > 1024\n"),
    (INK + "kernels.py", "def loop_bf16_supported(hidden: int, rank: int) -> bool:\n    return (\n"
                         "        mx.default_device() == mx.gpu\n"),
    (INK + "kernels.py", "    hidden = mixed.shape[-1]\n    return (\n        mx.default_device() == mx.gpu\n"),
    (INK + "kernels.py", "        and mx.default_device() == mx.gpu\n        and q.shape[-1] % 8 == 0\n"),
    (INK + "kernels.py", "        mx.default_device() != mx.gpu\n        or head_dim != 128\n"),
    (INK + "kernels.py", "if mx.default_device() != mx.gpu or experts > _MOE_SUM_MAX_EXPERTS:"),
    (INK + "kernels.py", "\nfrom ...utils import nax_attention\n"),
    (INK + "kernels.py", "    out = nax_attention.nax_banded_sink_attention(\n"),
    (INK + "fused.py", "    return (\n        mx.default_device() == mx.gpu\n        and q.shape[0] == 1\n"),
    (INK + "verify_experts.py", '    if mx.default_device() != mx.gpu or proj.mode != "nvfp4":\n        return False\n'),
    (INK + "expert_mma.py", "        dtype != mx.float16\n        or mx.default_device() != mx.gpu\n"),
] if INK084 else []
INK_READ += INK_083 + INK_084


# Names a patch function takes from an oMLX module that defines them elsewhere.
IMPORTED = [("omlx/scheduler.py", "GenerationBatch"), ("omlx/engine_pool.py", "ModelUnavailableError"),
            ("omlx/server.py", "app")]
# Whose attributes each ``_apply`` in PATCHES assigns, a class a group (None: the module's own functions).
OWNERS = {"patch_engine_core": ["EngineCore"], "patch_scheduler": ["Scheduler", "GenerationBatch"],
          "patch_mtp_next": ["GenerationBatch"], "patch_mtp_verify": ["Qwen3_5BatchInvariantForward"],
          "patch_prefix_cache": ["BlockAwarePrefixCache"],
          "patch_mtp_priming": [None], "patch_mtp_reconcile": [None], "patch_engine_pool": ["EnginePool"],
          "patch_server": [None], "name": ["VLMBatchedEngine", "BatchedEngine"]}     # name: _patch_engine, both engines


def source(rel: str) -> str:
    found = sorted(RES.glob(rel))
    if not found and rel.startswith(SITE) and "OMLX_SOURCE" in os.environ:
        pytest.skip("a bundled package: not in a source tree")
    assert found, f"{rel} is not in {RES}"
    return found[0].read_text()


@pytest.mark.parametrize("rel,cls,fn,is_async,lead,by_keyword", WRAPPED + CALLED + INK_CALLED,
                         ids=[f"{c + '.' if c else ''}{f}" for _, c, f, *_ in WRAPPED + CALLED + INK_CALLED])
def test_omlx_has_the_callable_as_wrapped(rel, cls, fn, is_async, lead, by_keyword):
    scope = ast.parse(source(rel)).body
    if cls:
        scope = next((n.body for n in scope if isinstance(n, ast.ClassDef) and n.name == cls), [])
    defs = [n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn]
    assert defs, f"{rel}: {cls or 'module'} has no {fn}"
    d = defs[-1]
    assert isinstance(d, ast.AsyncFunctionDef) == is_async
    assert not d.args.posonlyargs
    assert [a.arg for a in d.args.args][:len(lead)] == lead
    assert set(by_keyword) <= {a.arg for a in d.args.args + d.args.kwonlyargs}


@pytest.mark.parametrize("rel,text", READ + INK_READ, ids=[f"{Path(r).name}:{t}" for r, t in READ + INK_READ])
def test_omlx_has_what_the_wrappers_read(rel, text):
    assert text in source(rel)


def test_an_omlx_from_0_8_3_has_the_inkling_2_its_pins_are_checked_on():
    """INK_CALLED and INK_READ are empty without 0.8.3's cache.py, and hold 0.8.3's layout pins without 0.8.4's
    fused.py: from those versions on that would be pins dropped unseen."""
    version = RES / "omlx/_version.py"
    if not version.is_file():
        pytest.skip("no omlx/_version.py in this source")
    v = tuple(int(n) for n in version.read_text().split('"')[1].split(".")[:3])
    assert (INKLING or v < (0, 8, 3)) and (INK084 or v < (0, 8, 4))


@pytest.mark.skipif(not INKLING, reason="no Inkling 2 of 0.8.3's layout in this oMLX")
def test_inklings_metal_kernels_each_have_a_plain_path():
    """A J-lens fit needs gradients, which no Metal kernel has: each of kernels.py's five (the attention-residual mix,
    the flash prefill, the router, the short and the dynamic conv) is behind a default-device check (pinned in
    INK_READ) that inkling2._fit_paths turns to the plain path. 0.8.4's eleven add the fused mix and norm, the loop
    messages', two of flash decode, q/k/v's clips and the MoE sum's conv; fused.py's long attention and the experts'
    kernels (verify_experts.py, expert_mma.py) are switched the same way, and its NAX banded attention (one call)
    by a stand-in. verify_attention.py's and row_exact_gemv.py's run only on row-exact verify rows, never in a fit. A
    new one would need its own."""
    want = {"kernels.py": 11, "fused.py": 1, "verify_experts.py": 1, "expert_mma.py": 1} if INK084 else {"kernels.py": 5}
    assert {f: source(INK + f).count("mx.fast.metal_kernel(") for f in want} == want
    assert source(INK + "kernels.py").count("nax_attention.") == INK084


def _constant(rel: str, name: str):
    return next(n.value.value for n in ast.parse(source(rel)).body
                if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets))


def test_mtp_windows_hold_omlxs_longest_verify_window():
    """mtp.MAX_WINDOW: a forward of at most that many tokens with logits is a verify window (MtpLive.step), a longer
    one a prefill chunk. oMLX's longest: a context copy's MAX_COPY drafts + 1, Inkling 2's verify rows (capped by its
    _COMPILED_MAX_TOKENS). capture.BATCH_MAX_TOKENS is the same 16 for speed only: a longer window would be recorded a
    layer at a time instead of stacked."""
    from metalscope.mtp import MAX_WINDOW
    longest = [_constant("omlx/patches/mlx_lm_mtp/context_copy.py", "MAX_COPY") + 1]
    if INKLING:
        longest.append(_constant(INK + "model.py", "_COMPILED_MAX_TOKENS"))
    assert max(longest) <= MAX_WINDOW


@pytest.mark.skipif(not INKLING, reason="no Inkling 2 of 0.8.3's layout in this oMLX")
def test_inklings_sliding_state_keeps_mlx_lms_state():
    """jlens.fit's restore sets a sliding visit's ``state`` (mlx-lm's ArraysCache's, pinned in READ); its window size
    is the instance's own and stays: oMLX's ArraysCache does not redefine ``state``."""
    cls = next(n for n in ast.parse(source(INK + "cache.py")).body if isinstance(n, ast.ClassDef) and n.name == "ArraysCache")
    assert not [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "state"]


@pytest.mark.parametrize("rel,name", IMPORTED, ids=[n for _, n in IMPORTED])
def test_omlx_module_has_the_name(rel, name):
    bound = set()
    for node in ast.parse(source(rel)).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            bound |= {t.id for t in (node.targets if isinstance(node, ast.Assign) else [node.target]) if isinstance(t, ast.Name)}
    assert name in bound


def test_the_list_above_is_what_patches_wraps():
    """Each ``_apply`` in PATCHES names the attributes it assigns, class by class (OWNERS); uvicorn's
    handle_exit is middleware.py's."""
    assigned = set()
    for call in (c for p in PATCHES for c in ast.walk(ast.parse(p.read_text()))):
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_apply":
            name, groups = call.args[0], call.args[1:]
            owners = OWNERS[name.value if isinstance(name, ast.Constant) else name.id]
            if len(groups) == 1:                 # one group, each owner (_patch_engine runs once an engine class)
                groups = groups * len(owners)
            assert len(groups) == len(owners)
            assigned |= {(owner, kw.arg) for owner, group in zip(owners, groups) for kw in group.elts[1].keywords}
    assert assigned | {("Server", "handle_exit")} == {(cls, fn) for _, cls, fn, *_ in WRAPPED}


def test_omlx_assigns_the_qwen35_verifier_layer_once():
    """patch_mtp_verify wraps ``_layer`` at attach, after oMLX's one wrapper of it (applied before the model loads).
    A second oMLX assignment, perhaps made later, could land outside ours and skip the layer hooks again."""
    sites = [(p.relative_to(RES), n) for p in sorted((RES / "omlx").rglob("*.py"))
             for n in [p.read_text(errors="replace").count("Qwen3_5BatchInvariantForward._layer =")] if n]
    assert sites == [(Path("omlx/patches/qwen35_verify_qmm.py"), 1)]
