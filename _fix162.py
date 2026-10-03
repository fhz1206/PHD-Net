"""P162：让 [readout] 摘要行报**实际落地**的精度，而非请求值。"""
import io

P = "train/train.py"
s = io.open(P, encoding="utf-8").read()

old1 = '''    if cfg.readout_dtype in ("bf16", "fp16", "fp8", "fp4", "int8", "int4"):
        # P147：fp8 已是 **fhz 明确指定的默认**（模型本体 fp8 / 其余 fp16），
        # 所以这里**不再叫它「生产别用」**，而是如实陈述代价 + 给出复核手段。
        _isdef = (cfg.readout_dtype == "fp8")
        _tail = ("这是 fhz 2026-10-02 指定的默认（模型本体 fp8）。"
                 if _isdef else
                 "如需精确 p − t 规则请显式 --readout-dtype fp32。")
        print(f"[readout] compute precision = {cfg.readout_dtype"
              f" (checkpoint storage = {args.ckpt_dtype or 'fp32'}); "'''

new1 = '''    # ⚠⚠ **P162（fhz 2026-10-03 13:29 生产日志暴露）**：
    #   这一行原来打的是 **`cfg.readout_dtype`（请求值）**，而**实际落地的精度
    #   可能已被降级链改掉**。生产日志里两条行**自相矛盾**：
    #     [precision] 设备 npu 不支持 fp8 ... 已自动降级为 fp16   ← 警告说降了
    #     [readout] compute precision = fp8 ...                    ← 摘要说 fp8
    #   → 所有后续 A/B 判定都会**误以为跑的是 fp8 臂**，而实际是 fp16。
    # ✅ 修法：优先取读出实例**真实落地**的 dtype（降级链的结果），
    #   拿不到才回落到请求值（并明确标注「请求」）。
    _eff_dtype = cfg.readout_dtype
    _eff_note = ""
    try:
        _rd = getattr(lm.net, "readout", None)
        _res = getattr(_rd, "_precision_resolved", None) or {}
        if isinstance(_res, dict) and _res.get("dtype"):
            _eff_dtype = str(_res["dtype"])
            if _eff_dtype != str(cfg.readout_dtype):
                _eff_note = (f"；⚠ 请求 {cfg.readout_dtype} → **实际落地 "
                             f"{_eff_dtype}**（降级链，"
                             f"reason={_res.get('reason', '') or _res.get('fallback_reason', '?')}）")
    except Exception:                                        # noqa: BLE001
        pass

    if _eff_dtype in ("bf16", "fp16", "fp8", "fp4", "int8", "int4"):
        # P147：fp8 已是 **fhz 明确指定的默认**（模型本体 fp8 / 其余 fp16），
        # 所以这里**不再叫它「生产别用」**，而是如实陈述代价 + 给出复核手段。
        _isdef = (_eff_dtype == "fp8")
        _tail = ("这是 fhz 2026-10-02 指定的默认（模型本体 fp8）。"
                 if _isdef else
                 "如需精确 p − t 规则请显式 --readout-dtype fp32。")
        print(f"[readout] compute precision = {_eff_dtype} (requested "
              f"{cfg.readout_dtype}{_eff_note})"
              f" (checkpoint storage = {args.ckpt_dtype or 'fp32'}); "'''

assert s.count(old1) == 1, "anchor 1 not unique: %d" % s.count(old1)
s = s.replace(old1, new1)

old2 = '''    else:
        print(f"[readout] compute precision = {cfg.readout_dtype}"
              f" (checkpoint storage = {args.ckpt_dtype or 'fp32'})"
              f" — 精确 p − t 规则", flush=True)'''
new2 = '''    else:
        print(f"[readout] compute precision = {_eff_dtype} (requested "
              f"{cfg.readout_dtype}{_eff_note})"
              f" (checkpoint storage = {args.ckpt_dtype or 'fp32'})"
              f" — 精确 p − t 规则", flush=True)'''

assert s.count(old2) == 1, "anchor 2 not unique: %d" % s.count(old2)
s = s.replace(old2, new2)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("P162 applied: summary now reports the effective precision")