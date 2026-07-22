---
name: ascend-memory-table
description: Given a new model (HuggingFace id / local config.json / URL) and its vLLM serve command, fill a memory-analysis spreadsheet (.xlsx) matching the standard template, computing weights / activation / KV cache / num_blocks / concurrency using formulas aligned to the local vllm and vllm-ascend source code. Use when the user wants to estimate Ascend NPU per-card memory and KV concurrency for a model, says "显存填表", "参数分析表格", "拉起服务参数分析", "单卡权重估算", "KV并发估算", or provides a model + serve command and wants a filled xlsx. Trigger also when the user points to a config.json plus a vllm serve / api_server command.
---

# Ascend NPU 显存填表

为给定模型 + 拉起命令，按本地 vllm / vllm-ascend 源码口径计算显存与并发，输出与标准模板一致的 `.xlsx`。

## 输入

用户需提供（缺一不可）：

1. **模型** — 三选一：
   - 本地 `config.json` 路径（或含它的模型目录）
   - HuggingFace URL / model id（如 `MiniMaxAI/MiniMax-M3`）
   - 已下载到本地的模型目录
2. **拉起命令** — `vllm serve ...` 或 `python -m vllm.entrypoints.openai.api_server ...` 的完整命令字符串（含 `--tensor-parallel-size`、`--gpu-memory-utilization`、`--max-model-len`、`--quantization` 等参数）

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
- 若用户给 HF URL/id 且本地无 config，**下载到系统临时目录**（如 `$TEMP/ascend-mem-table/<model>/`），**不要下载到 skill 文件夹内**，避免破坏 skill 文件结构。只下 `config.json` 等小文件，不要下权重。

> ⚠️ 文件卫生规则（重要）：
> - **模型文件（config.json / 建模代码）下载到系统临时目录**，不要在 skill 目录下创建 `models/` 等过程文件夹。
> - **临时 runner 脚本**（用于绕开 shell 引号问题的 `python -c` 包装）写到系统临时目录，用完即删，**不要留在 `scripts/`**。
> - **输出 xlsx 统一写到 `outputs/` 子文件夹**（脚本默认行为，无需手动指定 `--out`）。
> - skill 目录应始终保持只有：`SKILL.md`、`references/`、`scripts/`（4 个固定文件）、`outputs/`。

### Step 2: 运行填表脚本

```bash
cd <skill_dir>/scripts
python fill_table.py \
  --config <config.json路径（系统临时目录）> \
  --serve "<拉起命令字符串>" \
  --model-name "<表格显示名>" \
  --hbm <64|96> \
  --B <平均请求长度> \
  --non-torch <3.2|3.5> \
  --graph <2.0|2.5> \
  --warmup-log <日志路径，可选>
# 不传 --out 时，默认输出到 <skill_dir>/outputs/<model-name>-memory-table.xlsx
```

脚本会：
1. 解析拉起命令 → 部署参数
2. 读 config.json → 架构参数
3. 按 W8A8/BF16 估算单卡权重
4. 按 `vllm_ascend_memory_formulas.py`（源码口径）算 available_kv / num_blocks / 并发
5. 若有 warmup 日志 → 用实测值覆盖理论占位
6. 基于预填模板 `template.xlsx` 填入 B 列数值，并在 C 列追加具体数字（保留模板里已固定的公式/源码说明）
7. 输出到 `outputs/`（含「假设与说明」页）

### Step 3: 汇报结果

向用户报告：
- 输出文件路径
- 关键数：单卡权重 GiB、Current KV GiB、可容纳 tokens、满长并发、单 token 合计 cache MiB
- 是否用了实测值（有日志）还是理论占位（无日志，提示用户拉起后用 warmup 日志替换绿底单元格）

## 公式口径

所有公式对齐本地 vllm / vllm-ascend 源码，详见 [references/formulas.md](references/formulas.md)。核心：

- `requested = total_memory × gpu_memory_utilization`（`worker.py:460`）
- `available_kv = requested − (weights + peak_act + non_torch)` **不含 graph**（`worker.py:581`）
- `fit_requested = requested − (W+act+non_torch+graph) − 150MiB`（`worker.py:728`）
- 单 token KV = `L × 2 × Hkv_local × D × dtype`，**Hkv_local = max(1, Hkv//TP)**（`AttentionSpec`）
- Index = `L_sparse × IdxD × dtype`，**key-only 无 ×2**（`MLAAttentionSpec` + `indexer.py:161`）
- `num_blocks = available_kv // page // layers`（`kv_cache_utils.py:1009`）
- `max_concurrency = num_blocks / blocks_per_req`（`kv_cache_utils.py:958`）
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

- `scripts/fill_table.py` — 主计算填表脚本（执行；只写 B 列值 + 向 C 列追加动态数字）
- `scripts/vllm_ascend_memory_formulas.py` — 源码口径公式封装
- `scripts/template.xlsx` — 预填模板（A 列标签 + C 列公式/源码说明已固定）
- `scripts/build_template.py` — 模板生成脚本（公式口径变更时改它并重跑 `python build_template.py`）
- `outputs/` — 所有生成的 xlsx 表格存放处（脚本默认输出目录）
- `references/formulas.md` — 完整公式-源码对应表
