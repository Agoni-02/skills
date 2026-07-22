# -*- coding: utf-8 -*-
"""
vLLM / vLLM-Ascend 口径的显存与 KV 公式（对齐本地仓库最新代码）。

源码锚点：
- requested / available_kv:
  vllm_ascend/worker/worker.py::determine_available_memory / compile_or_warm_up_model
- non_kv 组成:
  vllm/utils/mem_utils.py::memory_profiling + Ascend override
- page_size / num_blocks / concurrency:
  vllm/v1/kv_cache_interface.py::AttentionSpec.real_page_size_bytes
  vllm/v1/kv_cache_interface.py::MLAAttentionSpec.real_page_size_bytes  (indexer key-only)
  vllm/v1/core/kv_cache_utils.py::get_num_blocks / get_max_concurrency_for_kv_cache_config
  / get_kv_cache_capacity
- TP 下 local kv heads:
  vllm/models/minimax_m3/.../model.py  (max(1, n_kv // tp))
- Ascend block_size 默认倾向 128:
  vllm_ascend/utils.py refresh_block_size
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


GiB = 1024**3
MiB = 1024**2
REDUNDANCY_BUFFER = 150 * (1 << 20)  # worker.py: 150 MiB safety margin


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def local_num_kv_heads(n_kv: int, tp: int) -> int:
    """Per-rank KV heads after TP (vLLM MiniMax / GQA convention)."""
    if n_kv >= tp:
        assert n_kv % tp == 0
        return n_kv // tp
    assert tp % n_kv == 0
    return max(1, n_kv // tp)  # == 1 when tp > n_kv


@dataclass
class MemoryBudget:
    hbm_gib: float
    util: float
    weights_gib: float
    peak_activation_gib: float
    non_torch_gib: float
    npu_graph_gib: float
    # optional: if free on startup known; else assume exclusive ≈ hbm
    init_free_gib: float | None = None

    def requested_bytes(self) -> int:
        # worker.py: total_memory * gpu_memory_utilization
        return int(self.hbm_gib * GiB * self.util)

    def non_kv_profile_bytes(self) -> int:
        # Ascend determine_available_memory override:
        # non_kv = weights + peak_activation + non_torch
        # (graph NOT included at this stage)
        return int(
            (self.weights_gib + self.peak_activation_gib + self.non_torch_gib) * GiB
        )

    def available_kv_bytes(self) -> int:
        # available_kv_cache_memory_bytes = requested - non_kv
        return self.requested_bytes() - self.non_kv_profile_bytes()

    def current_kv_gib(self) -> float:
        return self.available_kv_bytes() / GiB

    def non_kv_after_warmup_bytes(self) -> int:
        # compile_or_warm_up_model suggestion:
        # weights + peak_act + non_torch + npugraph
        return int(
            (
                self.weights_gib
                + self.peak_activation_gib
                + self.non_torch_gib
                + self.npu_graph_gib
            )
            * GiB
        )

    def kv_fit_requested_bytes(self) -> int:
        return self.requested_bytes() - self.non_kv_after_warmup_bytes() - REDUNDANCY_BUFFER

    def kv_fully_utilize_free_bytes(self) -> int:
        free = (self.init_free_gib if self.init_free_gib is not None else self.hbm_gib) * GiB
        return int(free) - self.non_kv_after_warmup_bytes() - REDUNDANCY_BUFFER


@dataclass
class KVLayout:
    """Per-rank KV layout after TP. Supports GQA, MLA (DeepSeek), MiniMax-M3 hybrid index, and MTP draft layers."""

    L: int
    L_sparse: int
    Hkv_global: int
    Hkv_local: int
    D: int
    IdxD: int
    bytes_per_elem: int
    block_size: int
    tp: int
    # MLA (DeepSeek-V2/V3): KV cache stores compressed latent (kv_lora_rank) + rope (qk_rope_head_dim), no ×2
    is_mla: bool = False
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    # MTP / EAGLE draft layers (each draft layer reuses the main per-layer KV layout)
    draft_layers: int = 0

    @property
    def per_layer_main_bytes(self) -> int:
        """Per-token bytes of one full-attention layer (GQA: 2*KV*H*D; MLA: latent+rope, no ×2)."""
        if self.is_mla:
            return (self.kv_lora_rank + self.qk_rope_head_dim) * self.bytes_per_elem
        return 2 * self.Hkv_local * self.D * self.bytes_per_elem

    @property
    def effective_main_layers(self) -> int:
        """Main full-attn layers incl. draft (draft layers reuse main layout)."""
        return self.L + self.draft_layers

    @property
    def main_bytes_per_token(self) -> int:
        # Main-model full-attn layers only (excludes draft; draft reported separately for the table).
        return self.L * self.per_layer_main_bytes

    @property
    def draft_bytes_per_token(self) -> int:
        return self.draft_layers * self.per_layer_main_bytes

    @property
    def index_bytes_per_token(self) -> int:
        # MLAAttentionSpec / MiniMaxM3IndexerCache: key-only, 1 head, dim=IdxD
        if self.L_sparse <= 0 or self.IdxD <= 0:
            return 0
        return self.L_sparse * 1 * self.IdxD * self.bytes_per_elem

    @property
    def sum_bytes_per_token(self) -> int:
        return self.main_bytes_per_token + self.index_bytes_per_token + self.draft_bytes_per_token

    @property
    def main_page_size(self) -> int:
        # AttentionSpec.real_page_size_bytes (GQA: 2*block*Hkv_local*D*dtype)
        # MLA: block * (kv_lora_rank + qk_rope_head_dim) * dtype (no ×2, compressed latent)
        if self.is_mla:
            return self.block_size * (self.kv_lora_rank + self.qk_rope_head_dim) * self.bytes_per_elem
        return 2 * self.block_size * self.Hkv_local * self.D * self.bytes_per_elem

    @property
    def index_page_size_unpadded(self) -> int:
        if self.L_sparse <= 0 or self.IdxD <= 0:
            return 0
        # MLAAttentionSpec: storage_block_size * 1 * IdxD * dtype (NO ×2)
        return self.block_size * 1 * self.IdxD * self.bytes_per_elem

    def unify_index_block_size(self) -> int:
        """After unify_kv_cache_spec_page_size: scale indexer block_size to match main page."""
        main_p = self.main_page_size
        idx_p = self.index_page_size_unpadded
        if idx_p <= 0:
            return self.block_size
        if main_p % idx_p != 0:
            return self.block_size
        return self.block_size * (main_p // idx_p)

    def num_blocks_uniform(self, available_bytes: int) -> int:
        """Single-group full-attn (dense / Qwen3-MoE / DeepSeek MLA): get_num_blocks(L_eff, avail, page)."""
        page = self.main_page_size
        group_size = self.effective_main_layers
        if page <= 0 or group_size <= 0:
            return 0
        return max(0, available_bytes // page // group_size)

    def num_blocks_hybrid_m3(self, available_bytes: int) -> int:
        """
        M3 hybrid: group0 = (L + draft) full layers, group1 = L_sparse indexer layers.
        General hybrid path: group_size = max(len(g) for g in groups),
        page_size = unified main page, num_blocks = avail // page // group_size.
        """
        page = self.main_page_size
        group_size = max(self.effective_main_layers, self.L_sparse if self.L_sparse else self.effective_main_layers)
        if page <= 0 or group_size <= 0:
            return 0
        return max(0, available_bytes // page // group_size)

    def concurrency_and_tokens(
        self, num_blocks: int, max_model_len: int, hybrid: bool
    ) -> tuple[float, int, int, int]:
        """
        Returns (max_concurrency, gpu_kv_tokens, blocks_per_req, index_block_size).
        Mirrors get_max_concurrency_for_kv_cache_config + get_kv_cache_capacity.
        Draft layers add their own cdiv(max_model_len, block_size) blocks per request.
        """
        if hybrid and self.L_sparse > 0:
            idx_bs = self.unify_index_block_size()
            blocks_per_req = cdiv(max_model_len, self.block_size) + cdiv(
                max_model_len, idx_bs
            )
        else:
            idx_bs = self.block_size
            blocks_per_req = cdiv(max_model_len, self.block_size)
        # draft layers: each draft layer group contributes cdiv(max_model_len, block_size)
        if self.draft_layers > 0:
            blocks_per_req += self.draft_layers * cdiv(max_model_len, self.block_size)
        if blocks_per_req <= 0:
            return 0.0, 0, 0, idx_bs
        max_conc = num_blocks / blocks_per_req
        gpu_tokens = int(max_conc * max_model_len)
        return max_conc, gpu_tokens, blocks_per_req, idx_bs


def estimate_peak_activation_bytes(
    T: int,
    H: int,
    n_h_local: int,
    D: int,
    top_k: int,
    ep: int,
    dtype_act: int = 2,
    overhead: float = 1.25,
) -> int:
    """
    理论峰值激活（非源码；源码用 profile_run 的 torch peak）。
    覆盖 residual/hidden、MoE expand、Q、attn out、A2A workspace。
    """
    raw = (
        2 * T * H * dtype_act
        + T * top_k * H * dtype_act
        + T * n_h_local * D * dtype_act
        + T * H * dtype_act
        + 2 * (T * top_k / max(ep, 1)) * H * dtype_act
    )
    return int(raw * overhead)


def format_formula_requested(hbm: float, util: float) -> str:
    return (
        f"requested = total_memory × gpu_memory_utilization "
        f"= {hbm} GiB × {util}  "
        f"[vllm_ascend/worker/worker.py]"
    )


def format_formula_available_kv() -> str:
    return (
        "available_kv = requested − (weights + peak_activation + non_torch)  "
        "※不含 NPU graph  "
        "[worker.py::determine_available_memory Ascend override]"
    )


def format_formula_fit_requested() -> str:
    return (
        "kv_fit_requested = requested − (W + peak_act + non_torch + graph) − 150MiB  "
        "[worker.py::compile_or_warm_up_model]"
    )


def format_formula_full_free() -> str:
    return (
        "kv_full_free = init_free − (W + peak_act + non_torch + graph) − 150MiB  "
        "[worker.py::compile_or_warm_up_model]"
    )


def format_formula_main_kv(L, Hkv_local, D, bpe, tp, Hkv_global) -> str:
    return (
        f"per_token_main = L × 2 × Hkv_local × D × dtype "
        f"= {L}×2×{Hkv_local}×{D}×{bpe}  "
        f"(Hkv_local=max(1,{Hkv_global}//{tp})={Hkv_local})  "
        f"[AttentionSpec.real_page_size_bytes / block_size]"
    )


def format_formula_index(L_sparse, IdxD, bpe) -> str:
    if L_sparse <= 0:
        return "无 indexer → 0"
    return (
        f"per_token_index = L_sparse × 1 × IdxD × dtype "
        f"= {L_sparse}×1×{IdxD}×{bpe}  "
        f"(MLAAttentionSpec key-only，无 ×2)  "
        f"[MiniMaxM3IndexerCache.get_kv_cache_spec]"
    )


def format_formula_num_blocks(hybrid: bool) -> str:
    if hybrid:
        return (
            "hybrid: page=2·block·Hkv_local·D·dtype; group_size=max(L,L_sparse); "
            "num_blocks = available_kv // page // group_size  "
            "[kv_cache_utils.get_kv_cache_config_from_groups]"
        )
    return (
        "uniform: page=2·block·Hkv_local·D·dtype; "
        "num_blocks = available_kv // page // L  "
        "[kv_cache_utils.get_num_blocks]"
    )


def format_formula_concurrency(hybrid: bool) -> str:
    if hybrid:
        return (
            "blocks_per_req = cdiv(max_model_len, block_size) + cdiv(max_model_len, index_block_size); "
            "max_concurrency = num_blocks / blocks_per_req; "
            "GPU_KV_tokens = max_concurrency × max_model_len  "
            "[kv_cache_utils.get_kv_cache_capacity]"
        )
    return (
        "blocks_per_req = cdiv(max_model_len, block_size); "
        "max_concurrency = num_blocks / blocks_per_req; "
        "GPU_KV_tokens = max_concurrency × max_model_len  "
        "[kv_cache_utils.get_kv_cache_capacity]"
    )
