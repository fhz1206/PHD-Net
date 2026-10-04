"""P175：把 M2 接入 SparsePCStack（配置开关，默认关 —— 铁律）。"""
import io

P = "phdnet/sparse_pc.py"
s = io.open(P, encoding="utf-8").read()

# ── 1) 构造参数加 m2_backend（默认 "numpy" = 走原numba 路径）────────
OLD_INIT = """                 exc_ratio: float = 0.8, fused: bool = True):
        self.eta_pc, self.eta_oja, self.w_max = eta_pc, eta_oja, w_max"""
NEW_INIT = """                 exc_ratio: float = 0.8, fused: bool = True,
                 m2_backend: str = "numpy"):
        """`m2_backend`（P175）：**"numpy"（默认，走 numba 参照）/ "rust"**。

        ⚠ **默认关**（项目铁律：新增行为一律配置开关默认关闭、默认路径逐位不变）。
        ⚠ **"rust" 需先通过 `verify_m2_rust_kernels.py`** —— 未通过不得启用。
        ⚠ `"rust"` 只换**算子实现**，**不改编排**（融合核仍是 numba）——
          编排属 Python 职责（见 `phdnet_rs/README.md` 架构边界）。
        """
        self.eta_pc, self.eta_oja, self.w_max = eta_pc, eta_oja, w_max
        if m2_backend not in ("numpy", "rust"):
            raise ValueError(
                "m2_backend 只能是 'numpy'（默认，numba 参照）或 'rust'；"
                "实得 %r" % (m2_backend,))
        self.m2_backend = m2_backend
        if m2_backend == "rust":
            # 延迟导入 + **启动即校验**（fail-fast，不留到第一次调用）
            from phdnet_rs import load as _rs_load
            _kernels, _why = _rs_load()
            if _kernels is None:
                raise RuntimeError(
                    "m2_backend='rust' 但 Rust 库加载失败：%s\\n"
                    "→ 先运行 bash phdnet_rs/build.sh；或改回 m2_backend='numpy'"
                    % _why)
            self._rs = _kernels
        else:
            self._rs = None"""
assert s.count(OLD_INIT) == 1
s = s.replace(OLD_INIT, NEW_INIT)

# ── 2) 算子分派 helper（插在 _dense 之前）─────────────────────────
ANCHOR = "    def _dense(indptr, idx, val, n_rows: int, n_cols: int) -> np.ndarray:"
DISPATCH = '''    # ── M2 算子分派（P175）───────────────────────────────────────
    # `_csr_matvec` / `_csr_add_outer` / ... 与 Rust 版**同名同语义**，
    # 这里只做「按开关选实现」+ 必要的 dtype 转换。
    #
    # ⚠ **dtype 必须 fp32**（P173 起M2 就是 fp32；Rust 侧 `argtypes` 是 f32p）。
    #   若数组不是 fp32 → **报错**而非静默转换（静默转换会改变语义）。

    def _m2(self, name: str, *a):
        """按 `m2_backend` 调M2 算子。`name` 是 Python 侧的算子名。"""
        if self._rs is None:
            return globals()[name](*a)
        #Rust 版方法名：`_csr_matvec` -> `m2_matvec`
        return getattr(self._rs, "m2_" + name[5:])(*a)

    def _rs_only(self, name: str):
        """只Rust 有的算子（无 Python 对应）—— 当前无，留作扩展点。"""
        raise NotImplementedError(name)

'''
assert s.count(ANCHOR) == 1
s = s.replace(ANCHOR, DISPATCH + ANCHOR, 1)
io.open(P, "w", encoding="utf-8", newline="").write(s)
print("SparsePCStack: m2_backend switch added")