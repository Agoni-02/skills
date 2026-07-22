# -*- coding: utf-8 -*-
"""
Ascend NPU 显存填表脚本：输入 config.json + 拉起命令 → 输出 xlsx。

用法:
    python fill_table.py --config /path/to/config.json --serve "vllm serve ... --tensor-parallel-size 8 ..." \
        --hbm 64 --B 8192 --out filled.xlsx
    python fill_table.py --config-dir /path/to/model_dir --serve @serve.sh --hbm 64 --out filled.xlsx
    python fill_table.py --config config.json --serve "..." --warmup-log vllm.log --out filled.xlsx

公式口径见 references/formulas.md（对齐本地 vllm/vllm-ascend 源码）。
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

import openpyxl
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, PatternFill

from vllm_ascend_memory_formulas import (
    GiB, MiB, REDUNDANCY_BUFFER, KVLayout, MemoryBudget,
    estimate_peak_activation_bytes, local_num_kv_heads,
)

HEADER = PatternFill("solid", fgColor="D9EAF7")
YELLOW = PatternFill("solid", fgColor="FFF7D6")
COL_HDR = PatternFill("solid", fgColor="E8F1FF")
EST = PatternFill("solid", fgColor="E8F8E8")
WARN = PatternFill("solid", fgColor="FCE8E6")
MEAS = PatternFill("solid", fgColor="D6EAF8")  # measured (from log)


# ---------- serve command parsing ----------
def parse_serve(cmd: str) -> dict:
    """Extract deploy params from a vllm serve / api_server command string."""
    d = {
        "TP": 1, "DP": 1, "EP": None, "util": 0.9, "max_model_len": None,
        "max_num_seqs": None, "max_num_batched_tokens": None,
        "block_size": 128, "quantization": None, "kv_cache_dtype": None,
        "enable_expert_parallel": False, "enable_async_scheduling": False,
    }
    tokens = shlex.split(cmd)
    def val(flag, cast=str):
        for i, t in enumerate(tokens):
            if t == flag and i + 1 < len(tokens):
                return cast(tokens[i + 1])
            if t.startswith(f"{flag}="):
                return cast(t.split("=", 1)[1])
        return None
    if v := val("--tensor-parallel-size", int): d["TP"] = v
    if v := val("--data-parallel-size", int): d["DP"] = v
    if v := val("--gpu-memory-utilization", float): d["util"] = v
    if v := val("--max-model-len", int): d["max_model_len"] = v
    if v := val("--max-num-seqs", int): d["max_num_seqs"] = v
    if v := val("--max-num-batched-tokens", int): d["max_num_batched_tokens"] = v
    if v := val("--block-size", int): d["block_size"] = v
    if v := val("--quantization", str): d["quantization"] = v
    if v := val("--kv-cache-dtype", str): d["kv_cache_dtype"] = v
    if "--enable-expert-parallel" in tokens: d["enable_expert_parallel"] = True
    if "--async-scheduling" in tokens: d["enable_async_scheduling"] = True
    d["EP"] = d["TP"] * d["DP"] if d["enable_expert_parallel"] else d["TP"]
    return d


# ---------- config.json parsing ----------
def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    # unwrap text_config for VL models
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        tc = cfg["text_config"]
    else:
        tc = cfg
    return {"raw": cfg, "text": tc, "vision": cfg.get("vision_config")}


def arch_from_config(cfg: dict) -> dict:
    tc = cfg["text"]
    sac = tc.get("sparse_attention_config", {}) or {}
    moe_freq = tc.get("moe_layer_freq", [])
    sparse_freq = sac.get("sparse_attention_freq", [])
    L = tc["num_hidden_layers"]
    L_sparse = sum(1 for x in sparse_freq if x == 1) if sparse_freq else 0
    return {
        "L": L, "L_sparse": L_sparse,
        "Hkv": tc["num_key_value_heads"],
        "n_heads": tc["num_attention_heads"],
        "D": tc["head_dim"],
        "IdxD": sac.get("sparse_index_dim", 0) or 0,
        "hidden_size": tc["hidden_size"],
        "num_experts": tc.get("num_local_experts", 0) or 0,
        "num_experts_per_tok": tc.get("num_experts_per_tok", 0) or 0,
        "vocab_size": tc["vocab_size"],
        "tie": tc.get("tie_word_embeddings", False),
        "dense_intermediate_size": tc.get("dense_intermediate_size", tc.get("intermediate_size", 0)),
        "intermediate_size": tc.get("intermediate_size", 0),
        "shared_intermediate_size": tc.get("shared_intermediate_size", 0),
        "n_shared_experts": tc.get("n_shared_experts", 0) or 0,
        "idx_heads": sac.get("sparse_num_index_heads", 0) or 0,
        "has_vision": cfg["vision"] is not None,
    }


def estimate_weights(a: dict, tp: int, ep: int, quant: str) -> dict:
    """W8A8 text-only weight breakdown (GiB). Returns dict with 'weight' total."""
    H = a["hidden_size"]; vocab = a["vocab_size"]; L = a["L"]
    n_h = a["n_heads"]; n_kv = a["Hkv"]; D = a["D"]
    I_dense = a["dense_intermediate_size"]; I_moe = a["intermediate_size"]
    I_shared = a["shared_intermediate_size"]; E = a["num_experts"]
    n_shared = a["n_shared_experts"]; idx_heads = a["idx_heads"]; idx_dim = a["IdxD"]
    moe_freq = [1] * L  # default all-MoE; refined below if known
    # rough: if dense_intermediate given, assume first few layers dense
    n_dense = 0; n_moe = L; n_sparse = a["L_sparse"]
    if I_dense and I_dense != I_moe:
        n_dense = 3; n_moe = L - n_dense  # common pattern; refine from moe_layer_freq if available

    emb = vocab * H; lm = 0 if a["tie"] else vocab * H
    attn = H * (n_h * D) + H * (n_kv * D) * 2 + (n_h * D) * H
    qk_norm = (n_h + n_kv) * D
    rms = (2 * L + 1) * H
    indexer = (H * idx_heads * idx_dim + H * idx_dim + (idx_heads + 1) * idx_dim) if idx_dim else 0
    dense_mlp = 3 * H * I_dense if I_dense else 0
    shared = n_shared * 3 * H * I_shared if I_shared else 0
    routed = E * 3 * H * I_moe if I_moe else 0
    gate = (H * E + E) if E else 0

    b = 1.0 if (quant and quant.lower() in ("ascend", "w8a8")) else 2.0
    parts = {
        "K_emb_lm_tp": round((emb + lm) / tp * 2 / GiB, 3),
        "K_norms": round((rms + L * qk_norm) * 2 / GiB, 3),
        "K_gate_fp32": round(n_moe * gate * 4 / GiB, 3) if E else 0,
        "Q_attn_tp": round(L * attn / tp * b / GiB, 3),
        "Q_dense_tp": round(n_dense * dense_mlp / tp * b / GiB, 3) if n_dense else 0,
        "Q_shared_tp": round(n_moe * shared / tp * b / GiB, 3) if shared else 0,
        "Q_indexer_tp": round(n_sparse * indexer / tp * b / GiB, 3) if idx_dim else 0,
        "Q_experts_ep": round(n_moe * routed / ep * b / GiB, 3) if E else 0,
    }
    parts["weight"] = round(sum(v for v in parts.values() if v), 3)
    return parts


# ---------- warmup log parsing (optional, for measured values) ----------
def parse_warmup_log(path: Path | None) -> dict:
    """Extract measured memory from vllm-ascend warmup log. Returns {} if no log."""
    m = {}
    if not path or not path.exists():
        return m
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Loading model weights took X GB
    if x := re.search(r"Loading model weights took ([\d.]+)", text):
        m["weights_gib"] = float(x.group(1))
    # Available KV cache memory: X GiB
    if x := re.search(r"Available KV cache memory: ([\d.]+)", text):
        m["available_kv_gib"] = float(x.group(1))
    # Actual usage: X for weights, Y for peak activation, Z for non-torch, W for NPU graph
    if x := re.search(r"Actual usage: ([\d.]+) GiB for weights.*?([\d.]+) GiB for peak activation.*?([\d.]+) GiB for non-torch.*?([\d.]+) GiB for NPU graph", text):
        m["weights_gib"] = float(x.group(1)); m["peak_act_gib"] = float(x.group(2))
        m["non_torch_gib"] = float(x.group(3)); m["npu_graph_gib"] = float(x.group(4))
    # GPU KV cache size: X tokens
    if x := re.search(r"GPU KV cache size: ([\d,]+) tokens", text):
        m["gpu_kv_tokens"] = int(x.group(1).replace(",", ""))
    # Maximum concurrency for X tokens per request: Yx
    if x := re.search(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", text):
        m["max_concurrency"] = float(x.group(1))
    return m


# ---------- xlsx filling ----------
def sc(ws, r, c, v, fill=None):
    cell = ws.cell(r, c, v)
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    if fill is not None:
        cell.fill = fill
    return cell


def fill_xlsx(template_path: Path, out_path: Path, model_name: str, a: dict,
              deploy: dict, wbk: dict, budget: MemoryBudget, layout: KVLayout,
              hybrid: bool, num_blocks: int, max_conc: float, gpu_tokens: int,
              blocks_per_req: int, idx_bs: int, measured: dict, B: int) -> Path:
    wb = openpyxl.load_workbook(template_path)
    ws = wb.active
    ws.title = "Sheet1"
    for s in list(wb.sheetnames):
        if s != "Sheet1":
            del wb[s]

    Hkv_local = layout.Hkv_local
    kv_mib = round(layout.main_bytes_per_token / MiB, 6)
    idx_mib = round(layout.index_bytes_per_token / MiB, 6)
    sum_mib = round(layout.sum_bytes_per_token / MiB, 6)
    cur_kv = round(budget.current_kv_gib(), 3)
    fit_req = round(budget.kv_fit_requested_bytes() / GiB, 3)
    full_free = round(budget.kv_fully_utilize_free_bytes() / GiB, 3)
    w_gib = measured.get("weights_gib", wbk["weight"])
    act_gib = round(measured.get("peak_act_gib", budget.peak_activation_gib), 3)
    nt_gib = measured.get("non_torch_gib", budget.non_torch_gib)
    g_gib = measured.get("npu_graph_gib", budget.npu_graph_gib)
    has_log = bool(measured)
    wfill = MEAS if has_log else EST

    # 模板已预填 A 列标签 + C 列公式/源码说明；本函数只写 B 列值，
    # 并在含动态数值的行向 C 列「追加」具体数字（保留模板里的固定公式文本）。
    def put(r, v, fill=None):
        cell = ws.cell(r, 2, v)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        if fill is not None:
            cell.fill = fill
        return cell

    def app_c(r, text):
        cur = ws.cell(r, 3).value or ""
        ws.cell(r, 3, (cur + text) if cur else text)
        ws.cell(r, 3).alignment = Alignment(wrap_text=True, vertical="center")

    put(2, B, YELLOW)
    put(3, deploy["max_model_len"], YELLOW)
    put(4, deploy["util"], YELLOW)
    put(6, model_name, HEADER)
    put(7, a["L"]); put(8, a["L_sparse"]); put(9, a["Hkv"]); put(10, a["D"])
    put(11, a["IdxD"]); put(12, layout.bytes_per_elem); put(13, a["hidden_size"])
    put(14, a["num_experts_per_tok"]); put(15, a["num_experts"])
    put(16, a["num_experts"] // deploy["EP"])
    put(19, deploy["TP"], YELLOW); put(20, deploy["DP"], YELLOW); put(21, deploy["EP"], YELLOW)
    put(24, w_gib, wfill); put(25, act_gib, wfill); put(26, round(act_gib * GiB / 1e6, 1), wfill)
    put(27, nt_gib, wfill); put(28, g_gib, wfill); put(29, cur_kv, wfill)
    put(30, fit_req, wfill); put(31, full_free, wfill); put(32, num_blocks, wfill)
    put(33, layout.block_size); put(34, gpu_tokens, wfill); put(35, round(max_conc, 4), wfill)
    put(39, w_gib); put(40, act_gib); put(41, cur_kv); put(42, kv_mib)
    put(43, idx_mib); put(44, 0); put(45, sum_mib); put(46, gpu_tokens)
    put(47, round(cur_kv * GiB / (layout.sum_bytes_per_token * B), 3) if layout.sum_bytes_per_token else 0)

    # 仅向含动态数值的行追加具体数字到 C 列
    app_c(9, f" → Hkv_local={Hkv_local}=max(1,{a['Hkv']}//{deploy['TP']})")
    app_c(16, f"={a['num_experts']}/{deploy['EP']}")
    app_c(21, f"={deploy['TP']}×{deploy['DP']}")
    app_c(24, f" → 理论W8A8={wbk['weight']}")
    app_c(32, f" → 本例{'hybrid' if hybrid else 'uniform'}")
    app_c(35, f"={num_blocks}/{blocks_per_req}")
    app_c(39, " → " + "+".join(f"{k}={v}" for k, v in wbk.items() if k != "weight") + f"={wbk['weight']}")
    app_c(41, f"={round(budget.requested_bytes()/GiB,3)}−W−act−non_torch={cur_kv}")
    app_c(42, f"={a['L']}×2×{Hkv_local}×{a['D']}×{layout.bytes_per_elem}")
    app_c(43, f"={a['L_sparse']}×1×{a['IdxD']}×{layout.bytes_per_elem}" if a["IdxD"] else " → 无indexer→0")
    app_c(46, f"={round(max_conc,4)}×{deploy['max_model_len']}")
    app_c(47, f"={cur_kv}/{sum_mib}/{B}")

    asum = wb.create_sheet("假设与说明")
    rows = [("项目", "说明"), ("HBM", f"{budget.hbm_gib} GiB"), ("量化", deploy.get("quantization") or "BF16"),
            ("权重/激活/non_torch/graph", "实测(蓝底)" if has_log else "理论占位(绿底)，需warmup日志替换"),
            ("关键结果", f"W={w_gib}GiB, CurrentKV={cur_kv}GiB, tokens={gpu_tokens}, 满长并发={round(max_conc,4)}"),
            ("源码版本", "vllm d6dbdb9b0 / vllm-ascend f5b5514af")]
    for i, (x, y) in enumerate(rows, 1):
        asum.cell(i, 1, x); asum.cell(i, 2, y)
        if i == 1: asum.cell(i, 1).fill = HEADER; asum.cell(i, 2).fill = HEADER
    asum.column_dimensions["A"].width = 28; asum.column_dimensions["B"].width = 90
    wb.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Ascend NPU 显存填表")
    ap.add_argument("--config", help="config.json 路径")
    ap.add_argument("--config-dir", help="含 config.json 的模型目录")
    ap.add_argument("--serve", required=True, help="vllm serve / api_server 拉起命令字符串")
    ap.add_argument("--model-name", default="", help="表格中显示的模型名")
    ap.add_argument("--hbm", type=float, default=64.0, help="单卡 HBM GiB（默认64）")
    ap.add_argument("--B", type=int, default=8192, help="平均请求长度（默认8192）")
    ap.add_argument("--non-torch", type=float, default=3.2, help="non-torch 经验值 GiB")
    ap.add_argument("--graph", type=float, default=2.0, help="NPU graph 经验值 GiB")
    ap.add_argument("--warmup-log", help="vllm-ascend warmup 日志路径（可选，提供则用实测值）")
    ap.add_argument("--template", help="模板 xlsx（默认用脚本同目录 template.xlsx）")
    ap.add_argument("--out", default=None, help="输出 xlsx 路径（默认写到 skill 目录下 outputs/<模型名>-memory-table.xlsx）")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else (Path(args.config_dir) / "config.json")
    cfg = load_config(cfg_path)
    a = arch_from_config(cfg)
    deploy = parse_serve(args.serve)
    if not deploy["max_model_len"]:
        deploy["max_model_len"] = a.get("L", 8192)  # fallback
    if not deploy["max_num_batched_tokens"]:
        deploy["max_num_batched_tokens"] = deploy["max_model_len"]
    name = args.model_name or cfg_path.parent.name
    # 默认输出到 skill 目录下的 outputs/ 子文件夹，避免污染 skill 结构
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).resolve().parent.parent / "outputs" / f"{name}-memory-table.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quant = deploy.get("quantization") or ""
    is_w8a8 = quant.lower() in ("ascend", "w8a8")
    wbk = estimate_weights(a, deploy["TP"], deploy["EP"], quant)
    Hkv_local = local_num_kv_heads(a["Hkv"], deploy["TP"])
    n_h_local = max(1, a["n_heads"] // deploy["TP"])
    T = deploy["max_num_batched_tokens"]
    act_b = estimate_peak_activation_bytes(T, a["hidden_size"], n_h_local, a["D"], a["num_experts_per_tok"], deploy["EP"])
    bpe = 1 if (deploy.get("kv_cache_dtype") and "8" in str(deploy["kv_cache_dtype"]).lower()) else 2
    budget = MemoryBudget(args.hbm, deploy["util"], wbk["weight"], act_b / GiB, args.non_torch, args.graph, args.hbm)
    layout = KVLayout(a["L"], a["L_sparse"], a["Hkv"], Hkv_local, a["D"], a["IdxD"], bpe, deploy["block_size"], deploy["TP"])
    hybrid = a["L_sparse"] > 0
    avail = budget.available_kv_bytes()
    num_blocks = layout.num_blocks_hybrid_m3(avail) if hybrid else layout.num_blocks_uniform(avail)
    max_conc, gpu_tokens, blocks_per_req, idx_bs = layout.concurrency_and_tokens(num_blocks, deploy["max_model_len"], hybrid)
    measured = parse_warmup_log(Path(args.warmup_log) if args.warmup_log else None)
    if measured:
        if "available_kv_gib" in measured:
            budget2 = MemoryBudget(args.hbm, deploy["util"], measured.get("weights_gib", wbk["weight"]),
                                   measured.get("peak_act_gib", act_b / GiB), measured.get("non_torch_gib", args.non_torch),
                                   measured.get("npu_graph_gib", args.graph), args.hbm)
            avail = budget2.available_kv_bytes()
            num_blocks = layout.num_blocks_hybrid_m3(avail) if hybrid else layout.num_blocks_uniform(avail)
            max_conc, gpu_tokens, blocks_per_req, idx_bs = layout.concurrency_and_tokens(num_blocks, deploy["max_model_len"], hybrid)
            budget = budget2
        if "gpu_kv_tokens" in measured:
            gpu_tokens = measured["gpu_kv_tokens"]
        if "max_concurrency" in measured:
            max_conc = measured["max_concurrency"]
    tpl = Path(args.template) if args.template else Path(__file__).parent / "template.xlsx"
    out = fill_xlsx(tpl, out_path, name, a, deploy, wbk, budget, layout, hybrid,
                    num_blocks, max_conc, gpu_tokens, blocks_per_req, idx_bs, measured, args.B)
    print(f"OK -> {out}")
    print(f"  W={wbk['weight']}GiB  CurrentKV={budget.current_kv_gib():.3f}GiB  "
          f"tokens={gpu_tokens}  满长并发={max_conc:.4f}  单token合计={layout.sum_bytes_per_token/MiB:.6f}MiB")


if __name__ == "__main__":
    main()
