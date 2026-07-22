---
name: ascend-memory-table
description: Given a new model (HuggingFace id / local config.json / URL) and its vLLM serve command, fill a memory-analysis spreadsheet (.xlsx) matching the standard template, computing weights / activation / KV cache / num_blocks / concurrency using formulas aligned to the vllm and vllm-ascend source code. Use when the user wants to estimate Ascend NPU per-card memory and KV concurrency for a model, says "显存填表", "参数分析表格", "拉起服务参数分析", "单卡权重估算", "KV并发估算", or provides a model + serve command and wants a filled xlsx. Trigger also when the user points to a config.json plus a vllm serve / api_server command.
---

# Ascend NPU 显存填表

为给定模型 + 拉起命令，按 vllm / vllm-ascend 源码口径计算显存与并发，输出与标准模板一致的 `.xlsx`。

> **自包含 skill，无需本地 vllm / vllm-ascend 源码或安装包。**
> 所有公式已内置在 `scripts/vllm_ascend_memory_formulas.py`，源码锚点（`worker.py:581` 等）仅作溯源说明。
> 用户只需 Python 3.8+ 和 `openpyxl`，给一个 HuggingFace model id 即可开始估算。

## 前置条件

- Python 3.8+
- 安装依赖（仅 `openpyxl` 一个第三方库）：
  ```bash
  pip install -r <skill_dir>/requirements.txt
  ```
- 不需要：vllm、vllm-ascend、torch、npu 环境、模型权重文件。

## 快速开始（最常见用法）

```bash
cd <skill_dir>/scripts
python fill_table.py \
  --hf-model MiniMaxAI/MiniMax-M3 \
  --serve "vllm serve MiniMaxAI/MiniMax-M3 --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --max-model-len 65536 --quantization ascend --enable-expert-parallel" \
  --hbm 64 --B 8192
# 输出: <skill_dir>/outputs/MiniMax-M3-memory-table.xlsx
```

只需 3 个输入：① 模型（`--hf-model` / `--config` / `--config-dir` 之一）② 拉起命令 ③ 单卡 HBM 容量。

## 输入

用户需提供（缺一不可）：

1. **模型** — 四选一：
   - `--hf-model <HF id 或 URL>`：如 `MiniMaxAI/MiniMax-M3`、`Qwen/Qwen3-235B-A22B`、`deepseek-ai/DeepSeek-V3`，脚本自动从 HuggingFace 下载 `config.json`（仅几 KB，不下载权重），缓存到系统临时目录
   - `--ms-model <ModelScope id 或 URL>`：国内推荐，从 ModelScope 下载 `config.json`（如 `Qwen/Qwen3-235B-A22B`）
   - `--config <本地 config.json 路径>`（或含它的模型目录用 `--config-dir`）
   - 已下载到本地的模型目录（`--config-dir`）
2. **拉起命令** — `vllm serve ...` 或 `python -m vllm.entrypoints.openai.api_server ...` 的完整命令字符串（含 `--tensor-parallel-size`、`--gpu-memory-utilization`、`--max-model-len`、`--quantization` 等参数）

> **数据源**：`--source auto`（默认，先 HF 后 ModelScope 自动回退）| `hf`（仅 HF）| `ms`（仅 ModelScope）。
> 国内访问 HF 慢？`--ms-model` 直接走 ModelScope，或设 `$env:HF_ENDPOINT="https://hf-mirror.com"`。
> 私有 / gated 模型？设 `HF_TOKEN` 或 `MODELSCOPE_API_TOKEN`。

## 支持的模型系列（config.json 字段自动适配）

脚本会自动识别 `config.json` 的字段差异，无需用户指定模型类型。已核对 HuggingFace / ModelScope 上游真实 config：

| 模型系列 | model_type | MoE 字段 | 注意点 |
|---------|-----------|---------|--------|
| MiniMax-M3 / M2.7 | `minimax_m3_vl` | `num_local_experts` + `dense_intermediate_size` + `shared_intermediate_size` + `moe_layer_freq[list]` | 稀疏 index 注意（`sparse_attention_config`），MTP draft 层（`num_mtp_modules`） |
| Qwen3-MoE | `qwen3_moe` | `num_experts` + `moe_intermediate_size` + `decoder_sparse_step` / `mlp_only_layers` | 无 shared expert、无 sparse index |
| DeepSeek-V2 / V3 | `deepseek_v2` / `deepseek_v3` | `n_routed_experts` + `n_shared_experts` + `moe_intermediate_size` + `first_k_dense_replace` | **MLA 注意**：KV cache 用 `kv_lora_rank + qk_rope_head_dim`（无 ×2），NextN draft（`num_nextn_predict_layers`） |
| Dense（Llama / Qwen2 / GLM / 等） | `llama` / `qwen2` / `chatglm` ... | 无 MoE | 标准 GQA |

字段归一化规则：
- 专家数：`num_experts` / `num_local_experts` / `n_routed_experts` → 统一为 `num_experts`
- 专家 intermediate：`moe_intermediate_size` / `intermediate_size` → `routed_intermediate`
- 稠密层数：`first_k_dense_replace` / `moe_layer_freq[list]` / `mlp_only_layers` / `decoder_sparse_step` → `n_dense`
- head_dim：缺失时（DeepSeek MLA）由 `qk_nope_head_dim + qk_rope_head_dim` 推导

## 输出前先问用户要这些信息（提高准确率）

在生成表格前，用 `AskQuestion` 或对话方式询问以下信息（每项都直接影响数据精度）：

| 信息 | 影响 | 默认值（用户不给时） |
|------|------|---------------------|
| **单卡 HBM 容量**（64 / 96 / 80 GiB？） | 决定 requested_memory 与可用 KV 上限 | 64 GiB |
| **warmup 日志文件路径**（可选） | 有日志→权重/激活/non_torch/graph/可用KV 全部用实测值（蓝底）；无→理论占位（绿底） | 无 |
| **平均请求长度 B**（输入+输出 tokens） | 决定「最大并发（KV容量理论）」行 | 8192 |
| **KV cache dtype**（BF16 还是 C8/INT8？） | 决定 bytes_per_elem=2 还是 1，影响单 token cache 与 num_blocks | BF16（2） |
| **non-torch / NPU graph 经验值** | TP/DP 越大通信缓冲越大 | non_torch=3.2, graph=2.0 GiB |

> 提示话术示例：「为了让表格数据更准，建议补充：① 单卡 HBM 是 64 还是 96 GiB？② 有没有 vllm-ascend 的 warmup 日志（有日志就能用实测权重/激活/KV，否则只能理论估算）？③ KV cache 是 BF16 还是 C8？④ 平均请求长度大概多少？」

## 工作流程

### Step 1: 解析输入
- 从拉起命令提取：TP、DP、EP（=TP×DP if `--enable-expert-parallel`）、util、max-model-len、max-num-seqs、max-num-batched-tokens、block-size、quantization、kv-cache-dtype
- 读 `config.json` 取架构参数：L、L_sparse、Hkv、D、IdxD、hidden_size、num_experts、top-k 等
- 若用户给 HF id/URL，**脚本会自动下载 `config.json` 到系统临时目录**（`$TEMP/ascend-mem-table/`），**不下载权重**，**不写入 skill 文件夹**。
- 若拉起命令未指定 `--max-model-len`，回退到 `config.max_position_embeddings` / `seq_length`。

> ⚠️ 文件卫生规则（重要）：
> - **模型文件（config.json）下载到系统临时目录**，不要在 skill 目录下创建 `models/` 等过程文件夹。
> - **临时 runner 脚本**（用于绕开 shell 引号问题的 `python -c` 包装）写到系统临时目录，用完即删，**不要留在 `scripts/`**。
> - **输出 xlsx 统一写到 `outputs/` 子文件夹**（脚本默认行为，无需手动指定 `--out`）。
> - skill 目录应始终保持只有：`SKILL.md`、`requirements.txt`、`references/`、`scripts/`（固定文件）、`outputs/`。

### Step 2: 运行填表脚本

```bash
cd <skill_dir>/scripts
python fill_table.py \
  --hf-model <HF model id> \          # 或 --config <path> / --config-dir <dir>
  --serve "<拉起命令字符串>" \
  --model-name "<表格显示名，可选>" \
  --hbm <64|80|96> \
  --B <平均请求长度> \
  --non-torch <3.2|3.5> \
  --graph <2.0|2.5> \
  --warmup-log <日志路径，可选> \
  [--dry-run]                          # 只算不写表，快速预览
# 不传 --out 时，默认输出到 <skill_dir>/outputs/<model-name>-memory-table.xlsx
```

脚本会：
1. 解析拉起命令 → 部署参数（TP/DP/EP/util/max-model-len/quant/...）
2. 读 config.json → 架构参数（L/Hkv/D/hidden_size/experts/...）
3. 按 W8A8/BF16 估算单卡权重（解析式分项）
4. 按 `vllm_ascend_memory_formulas.py`（内置源码口径）算 available_kv / num_blocks / 并发
5. 若有 warmup 日志 → 用实测值覆盖理论占位
6. 基于预填模板 `template.xlsx` 填入 B 列数值，并在 C 列追加具体数字（保留模板里已固定的公式/源码说明）
7. 输出到 `outputs/`（含「假设与说明」页）

> `--dry-run` 只打印计算结果不写 xlsx，适合快速对比不同 TP/HBM 配置。

### Step 3: 汇报结果

向用户报告：
- 输出文件路径
- 关键数：单卡权重 GiB、Current KV GiB、可容纳 tokens、满长并发、单 token 合计 cache MiB
- 是否用了实测值（有日志）还是理论占位（无日志，提示用户拉起后用 warmup 日志替换绿底单元格）

## 公式口径

所有公式对齐 vllm / vllm-ascend 源码（**公式已内置在 skill 中，无需本地源码**），详见 [references/formulas.md](references/formulas.md)。核心：

- `requested = total_memory × gpu_memory_utilization`（`worker.py:460`）
- `available_kv = requested − (weights + peak_act + non_torch)` **不含 graph**（`worker.py:581`）
- `fit_requested = requested − (W+act+non_torch+graph) − 150MiB`（`worker.py:728`）
- **GQA 单 token KV** = `L × 2 × Hkv_local × D × dtype`，`Hkv_local = max(1, Hkv//TP)`（`AttentionSpec`）
- **MLA 单 token KV**（DeepSeek-V2/V3）= `L × (kv_lora_rank + qk_rope_head_dim) × dtype`，**无 ×2**（压缩 latent + rope，`MLAAttentionSpec`）
- Index = `L_sparse × IdxD × dtype`，**key-only 无 ×2**（`MLAAttentionSpec` + `indexer.py:161`）
- Draft（MTP/NextN）= `draft_layers × per_layer_main_kv`（draft 层复用主层 KV 布局，`vllm/v1/spec_decode`）
- `num_blocks = available_kv // page // effective_layers`（`kv_cache_utils.py:1009`）；`effective_layers = L + draft_layers`（uniform）或 `max(L+draft, L_sparse)`（hybrid）
- `max_concurrency = num_blocks / blocks_per_req`（`kv_cache_utils.py:958`）；`blocks_per_req` 含 draft 层组
- `GPU_KV_tokens = max_concurrency × max_model_len`（`kv_cache_utils.py:1833`）

## 表格字段说明

模板 47 行，分三段（A 列标签 + C 列公式/源码说明已预填固定，B 列由脚本按模型/部署参数填入）：
- **自定义配置项**（行1–16）：B、max-model-len、util、模型结构参数
- **实测显存**（行23–35）：权重/激活/non_torch/graph/Current KV/fit_requested/full_free/num_blocks/block_size/GPU_KV_tokens/满长并发
- **显存占用与并发**（行37–47）：权重/激活/可用KV/单token kv/index/draft/合计/可容纳tokens/最大并发

> 模板中 C 列的「计算公式 + 源码锚点」已逐行核对 `references/formulas.md` 并固定写入 `template.xlsx`，
> 脚本运行时只在含动态数值的行（如行9/16/21/24/32/35/39/41/42/43/46/47）向 C 列追加具体数字，
> 不再每次重新生成公式文本。若公式口径需更新，先改 `build_template.py` 并重新生成模板（见下）。

## 增强信息（可选附加）

填完表后，可附加一段分析：
- 该配置在 64/96 GiB 卡上是否放得下（权重占比）
- 满长并发 vs max-num-seqs 的关系（是否 OOM 风险）
- 建议：增 EP / 降 max-model-len / 上 C8 KV / 用 96 GiB 卡

## 文件

- `SKILL.md` — 本说明
- `requirements.txt` — 依赖清单（仅 `openpyxl`）
- `scripts/fill_table.py` — 主计算填表脚本（执行；只写 B 列值 + 向 C 列追加动态数字）
- `scripts/vllm_ascend_memory_formulas.py` — 源码口径公式封装（内置，无需 vllm/vllm-ascend）
- `scripts/template.xlsx` — 预填模板（A 列标签 + C 列公式/源码说明已固定）
- `scripts/build_template.py` — 模板生成脚本（公式口径变更时改它并重跑 `python build_template.py`）
- `outputs/` — 所有生成的 xlsx 表格存放处（脚本默认输出目录）
- `references/formulas.md` — 完整公式-源码对应表

## 常见问题

- **Q: 没有 vllm / vllm-ascend 源码能用吗？** A: 能。公式已内置在 `vllm_ascend_memory_formulas.py`，源码锚点只是溯源标注。
- **Q: 没装 torch / NPU 驱动能用吗？** A: 能。脚本只用 `openpyxl` + 标准库，纯计算，不调用任何 NPU / torch API。
- **Q: 不想下载几 GB 权重怎么办？** A: 用 `--hf-model` / `--ms-model`，脚本只下 `config.json`（几 KB）。
- **Q: 国内访问 HF 慢？** A: 用 `--ms-model` 直接走 ModelScope，或 `set HF_ENDPOINT=https://hf-mirror.com`（PowerShell）/ `export HF_ENDPOINT=https://hf-mirror.com`（bash）。
- **Q: HF 不可达时？** A: `--source auto`（默认）会自动回退到 ModelScope；或直接 `--ms-model <id>`。
- **Q: 私有 / gated 模型？** A: 设置环境变量 `HF_TOKEN` 或 `MODELSCOPE_API_TOKEN`。
- **Q: 拉起命令里没写 --max-model-len？** A: 脚本自动回退到 `config.max_position_embeddings` / `seq_length`。
- **Q: 支持 DeepSeek MLA / Qwen3-MoE / MiniMax-M3 吗？** A: 支持。脚本自动识别 `model_type` 并归一化 MoE / MLA / sparse / MTP 字段差异（见上表）。
- **Q: CurrentKV 是负数？** A: 表示该配置在指定 HBM 上放不下（OOM），需提高 TP / 上 W8A8 量化 / 换更大 HBM。这是正确信号，不是 bug。
