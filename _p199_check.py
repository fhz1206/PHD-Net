"""P199 验证：新 i18n 用真实坏例对比。"""
import sys

sys.path.insert(0, ".")
from phdnet.i18n import translate, set_lang, T

samples = [
    # (输入, 旧方案的坏输出)
    "突触参数容量验算（1B 口径：固定突触 + 大空间表容量）",
    "总突触参数容量           : 1,217,249,536   ≈ 1.217 × 10^9   ≥ 1B ✓",
    "  M4b 大空间表容量（事件驱动）: 1,207,959,552",
    "  静态内存 ≈ 76 MB（编码器 17 + 主干 6.3 + 读出 53）",
    "  大空间表索引映射 ≈ 537 MB；已生长突触按 100 万条 ≈ 10 MB",
    "  总容量 ≈ 1.217 × 10^9",
    "[big-ltm] grown 55,920 / capacity 62,914,560 (utilization 0.089%)",
    "already 生长突触按 100 万条",
    "大空间表容量（事件驱动）",
    "警告：读出后端不可用，已回退",
    "词表快照已保存",
    "token 1,000,000  sliding PPL  394.4687    62.31 ms/tok",
]
set_lang("en")
print("=== 新方案输出（en 模式）===")
bad = 0
for s in samples:
    out = translate(s)
    # 坏翻译判据：出现「单字夹英文」碎片（如 大 empty 间表 / already 生长突触）
    frag = ("empty 间" in out or "already 生长" in out
            or "backendunavailable" in out or "loadedcheckpoint" in out)
    if frag:
        bad += 1
    mark = "✗" if frag else "✓"
    print(f" {mark} {s[:46]}")
    print(f"      → {out[:80]}")
print(f"\n碎片化输出: {bad}/{len(samples)}")
assert bad == 0, "新方案仍有碎片化输出"

# zh 模式原样
set_lang("zh")
assert translate("大空间表容量") == "大空间表容量"
# T() 显式
set_lang("en")
assert T("训练完成", "training finished") == "training finished"
set_lang("zh")
assert T("训练完成", "training finished") == "训练完成"
# 诊断行跳过
set_lang("en")
diag = "[capability] accelerators detected ['npu']"
assert translate(diag) == diag
# 幂等安装
import io
from phdnet.i18n import install_stream_filter, _StreamFilter
install_stream_filter()
install_stream_filter()
assert isinstance(sys.stdout, _StreamFilter)
buf = io.StringIO()
sys.stdout = _StreamFilter(buf)
print("突触参数容量验算（1B 口径：固定突触 + 大空间表容量）")
sys.stdout = sys.__stdout__
out = buf.getvalue()
assert "Synapse capacity audit" in out, out
print("\nOK: P199 新 i18n 全部通过")
