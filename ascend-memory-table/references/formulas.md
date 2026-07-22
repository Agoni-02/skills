# 显存参数计算公式对应表（对齐 vllm / vllm-Ascend 最新代码）

> **本表所有公式已内置在 `scripts/vllm_ascend_memory_formulas.py`，使用本 skill 无需本地 vllm / vllm-ascend 源码或安装包。**
> 下列「源码位置」仅为溯源锚点，便于核对口径与未来同步上游。
>
> 仓库版本（已与 GitHub upstream 对齐）：
> - `vllm` HEAD: `d6dbdb9b0` (main)
> - `vllm-ascend` HEAD: `f5b5514af` (main)
>
> 单位：源码中除 `format_gib` 显示外，内部一律 **bytes**；本表 GiB=2³⁰ B，MiB=2²⁰ B。
> 标注「实测」=来自 warmup 日志；「理论」=无日志时的占位估算。

---

## A. 自定义配置项 / 模型结构（行 1–16）

| 行 | 字段 | 公式 / 取值 | 源码位置 |
|----|------|------------|----------|
| 2 | B 平均请求长度 | 用户输入 | — |
| 3 | max-model-len | `--max-model-len` | `vllm/config/model.py` |
| 4 | gpu-memory-utilization | `--gpu-memory-utilization` | `vllm/config/cache.py: CacheConfig.gpu_memory_utilization` |
| 7 | L 模型层数 | `config.num_hidden_layers` | HF `config.json` |
| 8 | L_sparse 稀疏/index层数 | `sum(sparse_attention_freq==1)`（M2.7=0） | `MiniMaxM3TextConfig.sparse_attention_config` |
| 9 | Hkv kv头数 | `config.num_key_value_heads`（**全局**） | HF config |
| 10 | D head_dim | `config.head_dim` | HF config |
| 11 | IdxD index_head_dim | `sparse_attention_config.sparse_index_dim`（M2.7=0） | HF config |
| 12 | bytes_per_elem | KV/Index dtype 字节数（BF16=2；W8A8C8 时 KV=1） | `CacheConfig.cache_dtype` → `get_dtype_size` |
| 13 | hidden_size | `config.hidden_size` | HF config |
| 14 | num_experts_per_tok | `config.num_experts_per_tok` (top-k) | HF config |
| 15 | num_experts | `config.num_local_experts` | HF config |
| 16 | experts_per_device | `num_experts // EP` | 推导 |

---

## B. 部署参数（行 18–21）

| 行 | 字段 | 公式 | 源码位置 |
|----|------|------|----------|
| 19 | TP | `--tensor-parallel-size` | `vllm/config/parallel.py` |
| 20 | DP | `--data-parallel-size` | `vllm/config/parallel.py` |
| 21 | EP | `TP × DP`（开 `--enable-expert-parallel`） | `vllm/distributed/...`；vLLM-Ascend MoE EP |

---

## C. 实测显存（行 23–35）— 对齐 warmup 日志字段

> 核心关系（`vllm_ascend/worker/worker.py`）：
> ```
> requested = total_memory × gpu_memory_utilization
> non_kv(profiling) = weights + peak_activation + non_torch      # Ascend override，不含 graph
> available_kv (Current KV) = requested − non_kv(profiling)
> non_kv(warmup建议) = weights + peak_act + non_torch + npugraph
> kv-cache-memory fit requested = requested − non_kv(warmup) − 150MiB
> kv-cache-memory fully utilize free = init_free − non_kv(warmup) − 150MiB
> ```

| 行 | 字段 | 公式（源码口径） | 源码位置 |
|----|------|----------------|----------|
| 24 | 权重 (GiB) | `model_runner.model_memory_usage`（实测）；理论 W8A8 = emb/TP·2 + norms·2 + gate·4 + (attn[+dense+shared+idx])/TP·1 + experts/EP·1 | `worker.py::load_model`；`model_runner.model_memory_usage` |
| 25 | 激活占用显存汇总 (GiB) | `peak_activation = torch_peak_increase`（profile_run 实测）；理论占位 ≈ 1.25×(2·T·H + T·topk·H + T·n_h_local·D + T·H + 2·(T·topk/EP)·H)×2B | `worker.py:563-568`；`mem_utils.py:310 torch_peak_increase` |
| 26 | 激活占用显存汇总 (MB) | `activation_bytes / 1e6`（十进制 MB） | 推导（对齐部分日志） |
| 27 | non-torch memory (GiB) | `non_torch_increase = after_profile.non_torch − before_create.non_torch`（HCCL/CANN/驱动） | `mem_utils.py:311`；`worker.py:569` |
| 28 | NPU graph memory (GiB) | `npugraph_memory_bytes = model_runner.capture_model()`（cudagraph 捕获，**不进** Current KV） | `worker.py:710,727` |
| 29 | Current KV cache memory (GiB) | `available_kv = requested − (weights + peak_act + non_torch)` **※不含 graph** | `worker.py:581` |
| 30 | kv-cache-memory fit requested (GiB) | `requested − (W + peak_act + non_torch + graph) − 150MiB` | `worker.py:720,728` |
| 31 | kv-cache-memory fully utilize free (GiB) | `init_free − (W + peak_act + non_torch + graph) − 150MiB` | `worker.py:729` |
| 32 | num_blocks | `available_kv // page_size // num_layers`（uniform）；hybrid 时 `// max(L, L_sparse)` | `kv_cache_utils.py:1009 get_num_blocks`；`:1380` |
| 33 | block_size | `cache_config.block_size`（Ascend `refresh_block_size` 默认/常用 128） | `vllm_ascend/utils.py:1241 refresh_block_size` |
| 34 | GPU KV cache size (tokens) | `max_concurrency × max_model_len` | `kv_cache_utils.py:1833 get_kv_cache_capacity` |
| 35 | Maximum concurrency for max_model_len | `num_blocks / blocks_per_req`，`blocks_per_req = Σ cdiv(max_model_len, group.block_size)` | `kv_cache_utils.py:951,958` |

---

## D. 显存占用与并发（行 37–47）

| 行 | 字段 | 公式（源码口径） | 源码位置 |
|----|------|----------------|----------|
| 39 | 权重 (GiB) | 同行 24 | `worker.py` |
| 40 | 激活占用显存汇总 (GiB) | 同行 25 | `worker.py` |
| 41 | 可用KV Cache (GiB) | 同行 29（Current KV） | `worker.py:581` |
| 42 | 单个token占用的kv cache (MiB) | **GQA**：`L × 2 × Hkv_local × D × dtype`，`Hkv_local = max(1, Hkv//TP)`；**MLA(DeepSeek)**：`L × (kv_lora_rank + qk_rope_head_dim) × dtype`（无 ×2） | `kv_cache_interface.py:212 AttentionSpec.real_page_size_bytes` / `MLAAttentionSpec` |
| 43 | 单个token占用的index cache (MiB) | `L_sparse × 1 × IdxD × dtype`（**key-only，无 ×2**） | `kv_cache_interface.py:411 MLAAttentionSpec`；`minimax_m3/common/indexer.py:161` |
| 44 | 单个token占用的draft kv cache (MiB) | `draft_layers × per_layer_main_kv`（MTP/NextN/EAGLE draft 层复用主层 KV 布局）；无 draft → 0 | `vllm/v1/spec_decode/...` |
| 45 | 单个token合计cache (MiB) | `main + index + draft`（每卡口径，已含 TP 分片） | 推导 |
| 46 | KV Cache 可容纳token数（日志块口径） | `max_concurrency × max_model_len`（= 行 34） | `kv_cache_utils.py:1833`；日志 `GPU KV cache size: %s tokens` |
| 47 | 最大并发（KV容量理论） | `max_concurrency = num_blocks / blocks_per_req`；按平均长度 B：`(available_kv / sum_bytes_per_token) / B` | `kv_cache_utils.py:958` |

---

## D2. MLA（DeepSeek-V2/V3）KV 布局

DeepSeek 的 Multi-head Latent Attention 把 K/V 压缩到低秩 latent，KV cache 只存压缩向量 + rope 部分，**与 num_kv_heads 解耦**：

- 每层每 token KV bytes = `(kv_lora_rank + qk_rope_head_dim) × dtype`（**无 ×2**，K/V 由同一 latent 重建）
- `main_page_size = block_size × (kv_lora_rank + qk_rope_head_dim) × dtype`
- 检测条件：`config.json` 同时含 `kv_lora_rank` 与 `q_lora_rank`
- 字段：`kv_lora_rank`、`qk_rope_head_dim`、`qk_nope_head_dim`、`v_head_dim`、`q_lora_rank`（后两者用于权重估算的 q_a/q_b/kv_a/kv_b 投影）

权重估算（MLA 每层）= `H·q_lora + q_lora·n_h·(qk_nope+qk_rope) + H·(kv_lora+qk_rope) + kv_lora·n_h·v_head + n_h·v_head·H`

---

## D3. Draft / MTP / NextN 层

MiniMax-M3（`num_mtp_modules`）、DeepSeek-V3（`num_nextn_predict_layers`）等带推测解码 draft 层：

- draft 层复用主层 KV 布局（GQA 或 MLA），每层每 token KV = `per_layer_main_kv`
- `effective_main_layers = L + draft_layers`（uniform）/ `max(L+draft, L_sparse)`（hybrid）
- `blocks_per_req` 增加 `draft_layers × cdiv(max_model_len, block_size)`
- 行 44 单独展示 draft 部分，行 45 合计含 draft

---

## E. 关键 page_size 公式（决定 num_blocks / 单 token cache）

| 组件 | 公式 | 源码 |
|------|------|------|
| 主 KV (FullAttentionSpec) | `page = 2 × block_size × num_kv_heads_local × head_size × dtype_size` | `kv_cache_interface.py:204-218` |
| Index (MLAAttentionSpec, key-only) | `page = storage_block_size × 1 × IdxD × dtype_size`（**无 ×2**） | `kv_cache_interface.py:398-416` |
| Hybrid 对齐 | 若 `main_page % index_page == 0`：indexer `block_size ← block_size × (main_page/index_page)` | `kv_cache_utils.py:unify_kv_cache_spec_page_size` |

---

## F. 本表相对旧版的修正点

1. **Current KV 不再扣 NPU graph** — `available_kv = requested − (W+act+non_torch)`（`worker.py:581`），graph 在 warmup 建议值里才扣。
2. **单 token KV 用 `Hkv_local`** — TP 后 `max(1, Hkv//TP)`，不能用全局 Hkv。
3. **Index 按 key-only** — MLAAttentionSpec 无 ×2（`indexer.py:161 num_kv_heads=1`）。
4. **fit requested / full free 扣 graph + 150MiB** — `worker.py:720,728,729`。
5. **num_blocks / 并发用 `get_num_blocks` + `get_kv_cache_capacity`** — `kv_cache_utils.py:1009,1833`。

---

## G. 实测 vs 理论

- 行 24/25/27/28/29/30/31/32/34/35：源码均由 `profile_run` + `capture_model` 实测得到，日志关键字见 `显存分析与OOM定位指南.md`。
- 无实测日志时，本表用理论占位（绿底）：权重按 W8A8 解析式、激活按峰值近似、non_torch/graph 按经验值。
- 拉起后用 warmup 日志覆盖：
  - `Loading model weights took ... GB` → 行 24
  - `Available KV cache memory: ... GiB` → 行 29
  - `... for weights, ... for peak activation, ... for non-torch memory, ... for NPU graph memory` → 行 24/25/27/28
  - `GPU KV cache size: ... tokens` / `Maximum concurrency for ... tokens per request: ...x` → 行 34/35

---

## H. 各模型系列 config.json 字段归一化（HuggingFace / ModelScope 实测）

不同模型系列 `config.json` 字段名差异由 `arch_from_config` 自动归一化，无需用户干预：

| 含义 | MiniMax-M3 | Qwen3-MoE | DeepSeek-V2/V3 | Dense(Llama等) |
|------|------------|-----------|----------------|----------------|
| 专家数 | `num_local_experts` | `num_experts` | `n_routed_experts` | — |
| 每专家 intermediate | `intermediate_size` | `moe_intermediate_size` | `moe_intermediate_size` | — |
| 稠密层 intermediate | `dense_intermediate_size` | `intermediate_size` | `intermediate_size` | `intermediate_size` |
| shared expert | `n_shared_experts` + `shared_intermediate_size` | 无 | `n_shared_experts`（用 `moe_intermediate_size`） | — |
| 稠密层数 | `moe_layer_freq[list]` 中 0 的个数 | `mlp_only_layers` / `decoder_sparse_step` | `first_k_dense_replace` | 全部 |
| head_dim | `head_dim` | `head_dim` | 无（`qk_nope+qk_rope` / `v_head_dim`） | `head_dim` |
| 注意类型 | GQA + sparse index | GQA | **MLA**（`kv_lora_rank`+`qk_rope_head_dim`） | GQA |
| draft 层 | `num_mtp_modules` | 无 | `num_nextn_predict_layers` | 无 |

下载源：
- HuggingFace：`https://huggingface.co/<id>/resolve/<rev>/config.json`（镜像 `HF_ENDPOINT=https://hf-mirror.com`）
- ModelScope：`https://modelscope.cn/api/v1/models/<id>/repo?Revision=<rev>&FilePath=config.json`
- `--source auto`（默认）：先 HF 后 ModelScope 自动回退；`--ms-model` 强制 ModelScope
