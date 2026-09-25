"""PHD-Net 1B 档完整检查点（train_1b 子项目专用）。

与 tools/train_production.py 的 save_ckpt/load_ckpt 相比，1B 档额外完整保存：

1. **大空间事件驱动表**（1B 主体）—— compact_csr 快照（row_ids/indptr/indices/data）
   + 突触迹与时间戳（t_pre/t_post/stamp_pre/stamp_post）+ step_count + in_deg。
   恢复时逐位重建邻接结构（dict 版 / 在线 CSR 版均支持），续训零状态损失。
2. **词表与分词器** —— seg.vocab / seg.max_len / tokens 全量保存；
   SDR 哈希由 (tokens 顺序, n_sdr, n_active, seed) 确定性重建，逐位一致。
3. **稀疏读出 CSR**（readout_conn_k > 0 时）—— train_production 明确不支持恢复，
   此处补齐（indptr/idx/val 三元组直接落盘）。
4. 主干 CSR 四权重（结构不变 → 只存值，与生产版一致）。

注意：恢复要求用**同一语料与 --max-chars** 重建词表（检查点内词表用于校验，
不一致即 fail-fast，避免读出矩阵形状错配的静默错训）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

CKPT_VERSION = 1


# ────────────────────────────── 保存 ──────────────────────────────
def save_model(path: Path, lm, cfg, done: int, extra: dict | None = None) -> dict:
    """保存完整可续训状态到 path（.npz + 同名 .json 元数据）。返回元数据 dict。"""
    from dataclasses import asdict

    net = lm.net
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrs: dict = {
        "encoder_W": np.asarray(net.encoder.W), "encoder_b": np.asarray(net.encoder.b),
        "stdp_W": np.asarray(net.stdp.W),
        "wm_slots": np.asarray(net.wm.slots), "wm_strength": np.asarray(net.wm.strength),
    }
    # 主干：CSR 存三元组值（结构由 conn_k 决定，重建时形状校验）；稠密存整矩阵
    if hasattr(net.pc, "up0"):
        for nm in ("up0", "up1", "dn0", "dn1"):
            ip, idx, val = getattr(net.pc, nm)
            arrs[f"pc_{nm}_ip"], arrs[f"pc_{nm}_idx"], arrs[f"pc_{nm}_val"] = ip, idx, val
    else:
        for nm in ("W_up0", "W_up1", "W_dn0", "W_dn1"):
            arrs[f"pc_{nm}"] = np.asarray(getattr(net.pc, nm))

    # 读出：稠密 W / 稀疏 CSR（两级群体读出暂不支持检查点——fail-fast）
    if getattr(net.readout, "hidden", 0) > 0:
        raise NotImplementedError("两级群体读出（readout_hidden>0）暂不支持检查点保存")
    if net.readout.conn_k > 0:
        ip, idx, val = net.readout._csr
        arrs["ro_ip"], arrs["ro_idx"], arrs["ro_val"] = ip, idx, val
    else:
        arrs["ro_W"] = np.asarray(net.readout.W)

    # 大空间事件驱动表（1B 主体）：CSR 快照 + 迹/时间戳 + 步数
    if cfg.big_ltm:
        t = net.ltm.table
        snap = t.compact_csr()
        arrs["ltm_row_ids"] = snap["row_ids"]
        arrs["ltm_ip"] = snap["indptr"]
        arrs["ltm_idx"] = snap["indices"]
        arrs["ltm_data"] = snap["data"]
        for nm in ("t_pre", "t_post"):
            d = getattr(t, nm)
            arrs[f"ltm_{nm}_k"] = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
            arrs[f"ltm_{nm}_v"] = np.fromiter(d.values(), dtype=np.float64, count=len(d))
        for nm in ("stamp_pre", "stamp_post"):
            d = getattr(t, nm)
            arrs[f"ltm_{nm}_k"] = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
            arrs[f"ltm_{nm}_v"] = np.fromiter(d.values(), dtype=np.int64, count=len(d))
        arrs["ltm_step"] = np.array([t.step_count], dtype=np.int64)
        if t.growth_guidance and t.in_deg:
            arrs["ltm_indeg_k"] = np.fromiter(t.in_deg.keys(), dtype=np.int64, count=len(t.in_deg))
            arrs["ltm_indeg_v"] = np.fromiter(t.in_deg.values(), dtype=np.int64, count=len(t.in_deg))

    # 词表与分词器（seg.vocab / max_len / tokens；SDR 哈希确定性重建）
    # 用 numpy 原生 unicode 数组（非 object），保持 allow_pickle=False 可读
    arrs["tok_vocab"] = np.array(sorted(lm.tok.seg.vocab) or [""], dtype=np.str_)
    arrs["tok_tokens"] = np.array(lm.tok.tokens or [""], dtype=np.str_)
    arrs["tok_max_len"] = np.array([lm.tok.seg.max_len], dtype=np.int64)

    # ── 运行时状态（缺任何一项，续训路径即分叉——对拍 _verify_roundtrip.py 抓出）──
    if cfg.pc_predictive_target or cfg.multi_modulation or cfg.readout_replay:
        raise NotImplementedError(
            "pc_predictive_target / multi_modulation / readout_replay 开启时"
            "存在未序列化的内部缓存（_prev_pc / 多通道调制器 / 回放储备库），"
            "1B 档配置默认关闭；如需开启请先扩展 ckpt_1b 的状态覆盖。")
    st = net.stdp
    # STDP 内部迹：短窗 pre/post、长窗（dual）、二阶矩（adaptive）、BCM 阈值、E/I 结构
    arrs["stdp_traces"] = np.stack([np.asarray(st.t_pre), np.asarray(st.t_post),
                                    np.asarray(st.t_pre_slow), np.asarray(st.t_post_slow)])
    arrs["stdp_v"] = np.asarray(st.v)
    arrs["stdp_theta"] = np.asarray(st.theta)
    arrs["stdp_is_inh"] = np.asarray(st.is_inh)
    # 神经调制器 Welford 统计（mu/m2/count → gate/mode 与印迹触发节奏）
    for nm, mod in (("mod", net.modulator), ("tmod", net.task_mod)):
        arrs[f"{nm}_welford"] = np.array([mod.mu, mod.m2, mod.count], dtype=np.float64)
    # 网络顶层运行时：计步（检索/回放/摘要节奏均取模于它）/ 任务门控 / DA / 经验 / 读出退火率
    arrs["net_step_count"] = np.array([net.step_count], dtype=np.int64)
    arrs["net_scalars"] = np.array([net._task_gate, net._last_da, net._exp, net._ro_eta],
                                   dtype=np.float64)
    arrs["net_prev_rate"] = np.asarray(net._prev_rate, dtype=np.float64)
    arrs["net_last_rate"] = np.asarray(net._last_rate, dtype=np.float64)
    # 大空间表的 imprint 前值缓存（缺了会丢续训后第一对 learn）
    if cfg.big_ltm:
        prev = net.ltm._prev
        arrs["ltm_prev"] = np.array(prev if prev is not None else [], dtype=np.int64)
        arrs["ltm_prev_flag"] = np.array([prev is not None], dtype=np.int64)

    meta = {
        "version": CKPT_VERSION, "done": int(done), "big_ltm": bool(cfg.big_ltm),
        "csr_pc": bool(hasattr(net.pc, "up0")), "readout_conn_k": int(net.readout.conn_k),
        "csr_online": bool(getattr(net.ltm.table, "csr_online", False)),
        "int8_store": bool(getattr(net.ltm.table, "int8_store", False)),
        "vocab_size": int(len(lm.tok)), "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cfg": asdict(cfg),
    }
    if extra:
        meta.update(extra)
    arrs["meta"] = np.array([json.dumps(meta, ensure_ascii=False)])
    # np.savez（不压缩）：1B 档读出矩阵数百 MB，fp64 随机权重压缩率低且耗时长
    np.savez(str(path), **arrs)
    Path(str(path)).with_suffix(".json").write_text(
        json.dumps({k: v for k, v in meta.items() if k != "cfg"},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ────────────────────────────── 恢复 ──────────────────────────────
def load_model(path: Path, lm) -> dict:
    """从 path 恢复完整状态到 lm（含词表/权重/大空间表）。返回元数据 dict。

    fail-fast：检查点词表大小与当前 tokenizer 不一致时抛 ValueError
    （读出矩阵形状将错配——请用与原训练相同的 --data 与 --max-chars）。
    """
    net = lm.net
    with np.load(str(path), allow_pickle=False) as z:
        meta = json.loads(str(z["meta"][0]))
        if meta.get("version", 0) != CKPT_VERSION:
            raise ValueError(f"检查点版本不支持: {meta.get('version')}")
        if meta["vocab_size"] != len(lm.tok):
            raise ValueError(
                f"词表大小不一致：检查点 {meta['vocab_size']} vs 当前 {len(lm.tok)}。"
                f"续训必须使用与原训练相同的 --data 与 --max-chars。")

        _restore_tokenizer(lm, z)

        net.encoder.W[:] = z["encoder_W"]
        net.encoder.b[:] = z["encoder_b"]
        net.stdp.W[:] = z["stdp_W"]
        net.wm.slots[:] = z["wm_slots"]
        net.wm.strength[:] = z["wm_strength"]

        if meta["csr_pc"]:
            for nm in ("up0", "up1", "dn0", "dn1"):
                ip, idx, val = getattr(net.pc, nm)
                if len(ip) != len(z[f"pc_{nm}_ip"]) or len(val) != len(z[f"pc_{nm}_val"]):
                    raise ValueError(f"主干 CSR {nm} 形状不一致（conn_k 配置变了？）")
                val[:] = z[f"pc_{nm}_val"]
        else:
            for nm in ("W_up0", "W_up1", "W_dn0", "W_dn1"):
                getattr(net.pc, nm)[:] = z[f"pc_{nm}"]

        if meta["readout_conn_k"] > 0:
            if net.readout.conn_k != meta["readout_conn_k"]:
                raise ValueError("读出 conn_k 配置与检查点不一致")
            ip, idx, val = net.readout._csr
            if len(val) != len(z["ro_val"]):
                raise ValueError("稀疏读出形状不一致（词表或 conn_k 变了？）")
            val[:] = z["ro_val"]
        else:
            net.readout.W = np.asarray(z["ro_W"])

        if meta["big_ltm"]:
            _restore_table(net.ltm.table, z)
            net.ltm._prev = (z["ltm_prev"].tolist()
                             if bool(z["ltm_prev_flag"][0]) else None)

        # ── 运行时状态恢复（与 save 端一一对应）──
        tr = z["stdp_traces"]
        net.stdp.t_pre[:] = tr[0]
        net.stdp.t_post[:] = tr[1]
        net.stdp.t_pre_slow[:] = tr[2]
        net.stdp.t_post_slow[:] = tr[3]
        net.stdp.v[:] = z["stdp_v"]
        net.stdp.theta[:] = z["stdp_theta"]
        net.stdp.is_inh[:] = z["stdp_is_inh"].astype(bool)
        for nm, mod in (("mod", net.modulator), ("tmod", net.task_mod)):
            mod.mu, mod.m2, mod.count = (float(x) for x in z[f"{nm}_welford"])
        net.step_count = int(z["net_step_count"][0])
        net._task_gate, net._last_da, net._exp, net._ro_eta = (
            float(x) for x in z["net_scalars"])
        net._prev_rate[:] = z["net_prev_rate"]
        net._last_rate[:] = z["net_last_rate"]
    return meta


def _restore_tokenizer(lm, z) -> None:
    """由检查点数组重建分词器状态（seg.vocab/max_len/tokens/stoi/_sdrs）。

    SDR 哈希逻辑与 WordTokenizer.__init__ 逐位一致（tokens 顺序 + seed 决定）。
    """
    from phdnet.word_encoder import WordSegmenter     # 延迟导入（避免循环依赖）

    tok = lm.tok
    vocab = {str(w) for w in z["tok_vocab"]}
    tokens = [str(t) for t in z["tok_tokens"]]
    max_len = int(z["tok_max_len"][0])

    seg = WordSegmenter.__new__(WordSegmenter)        # 跳过语料统计，直接注入状态
    seg.vocab = vocab
    seg.max_len = max_len
    tok.seg = seg
    tok.tokens = tokens
    tok.stoi = {t: i for i, t in enumerate(tokens)}
    _rebuild_sdrs(tok, tokens)


def _rebuild_sdrs(tok, tokens) -> None:
    """按 WordTokenizer 的确定性哈希重建 token → SDR 缓存（与 word_encoder.py 同式）。"""
    from phdnet.tokenizer import U64, mix64
    n_sdr, n_active, seed = tok.n_sdr, tok.n_active, tok.seed
    tok._sdrs = {}
    for i, t in enumerate(tokens):
        x = U64(i) * U64(2654435761) + np.arange(n_active, dtype=np.uint64)
        idx = (mix64(x + U64(seed)) % U64(n_sdr)).astype(np.int64)
        s = np.zeros(n_sdr)
        s[idx] = 1.0
        tok._sdrs[t] = s


def _restore_table(t, z) -> None:
    """由 CSR 快照逐位重建大空间表（dict 版 / 在线 CSR 版），并恢复迹与步数。"""
    rows = z["ltm_row_ids"]
    ip, idx, data = z["ltm_ip"], z["ltm_idx"], z["ltm_data"]
    int8 = t.int8_store

    # 先清空再注入（resume 时表应为空，防御性处理）
    if hasattr(t, "keys"):                            # OnlineCSRTable
        t.keys.clear(), t.vals.clear(), t.size.clear()
    t.out.clear()

    if hasattr(t, "keys"):                            # OnlineCSRTable：直接重建行数组
        from phdnet.sparse_table import OnlineCSRTable
        assert isinstance(t, OnlineCSRTable)
        for r in range(len(rows)):
            row = int(rows[r])
            s, e = int(ip[r]), int(ip[r + 1])
            cap = max(t._cap0, e - s)
            dt = np.int16 if int8 else np.float64
            t.keys[row] = np.zeros(cap, dtype=np.int64)
            t.vals[row] = np.zeros(cap, dtype=dt)
            t.size[row] = 0
            for j in range(s, e):
                # 直接写码值/真值（绕过 _append 的量化往返，逐位保真）
                t.keys[row][t.size[row]] = int(idx[j])
                t.vals[row][t.size[row]] = (np.int16(int(data[j])) if int8
                                            else np.float64(float(data[j])))
                t.size[row] += 1
    else:                                             # dict 版邻接表
        for r in range(len(rows)):
            row = int(rows[r])
            bucket: dict[int, float] = {}
            for j in range(int(ip[r]), int(ip[r + 1])):
                bucket[int(idx[j])] = int(data[j]) if int8 else float(data[j])
            t.out[row] = bucket

    # 迹 / 时间戳 / 步数 / 入度
    t.t_pre = dict(zip(z["ltm_t_pre_k"].tolist(), z["ltm_t_pre_v"].tolist()))
    t.t_post = dict(zip(z["ltm_t_post_k"].tolist(), z["ltm_t_post_v"].tolist()))
    t.stamp_pre = dict(zip(z["ltm_stamp_pre_k"].tolist(), z["ltm_stamp_pre_v"].tolist()))
    t.stamp_post = dict(zip(z["ltm_stamp_post_k"].tolist(), z["ltm_stamp_post_v"].tolist()))
    t.step_count = int(z["ltm_step"][0])
    if "ltm_indeg_k" in z:
        t.in_deg = dict(zip(z["ltm_indeg_k"].tolist(), z["ltm_indeg_v"].tolist()))
