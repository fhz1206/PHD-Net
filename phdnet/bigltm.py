"""M4 容量层适配器 —— 把 n_dim 维表示经确定性哈希接入大空间印迹表。

存储本体（SparseSynapseTable）已拆分至 phdnet/sparse_table.py，
此处保留 SparseLTM 并再导出 SparseSynapseTable 以兼容旧引用。"""

import numpy as np

from .sparse_table import OnlineCSRTable, SparseSynapseTable
from .tokenizer import U64, _mix64

__all__ = ["SparseLTM", "SparseSynapseTable", "OnlineCSRTable"]



class SparseLTM:
    """大容量长期记忆适配器（接口对齐 LongTermMemory：imprint / recall）。

    n_dim 维表示 → 每个维度经确定性哈希派生 k_hash 个大空间神经元索引
    （分布式编码，类 SDR 扩展）；联想存储与召回在 SparseSynapseTable 中完成。

    O1/O3（2026-09-22）：`csr_online=True` 时底层切换为在线可写 CSR 表
    （`OnlineCSRTable`，定长行 + 预留槽）；默认 False = dict 邻接表，
    默认路径逐位不变。两版 predict/learn 逐位等价（tests/verifiers/verify_csr_equiv.py）。
    """

    def __init__(self, n_dim: int, n_neurons: int = 1 << 24, m_out: int = 60,
                 k_hash: int = 4, lam: float = 0.7, eta: float = 0.08,
                 seed: int = 0xA5A55A5A, prune: int = 0,
                 growth_guidance: bool = False, int8_store: bool = False,
                 csr_online: bool = False, csr_grow_chunk: int = 64):
        self.n_dim = n_dim
        self.k_hash = k_hash
        self.csr_online = bool(csr_online)
        if self.csr_online:
            self.table = OnlineCSRTable(n_neurons, m_out, lam, eta, seed=seed,
                                        prune=prune, growth_guidance=growth_guidance,
                                        int8_store=int8_store,
                                        grow_chunk=csr_grow_chunk)
        else:
            self.table = SparseSynapseTable(n_neurons, m_out, lam, eta, seed=seed,
                                            prune=prune, growth_guidance=growth_guidance,
                                            int8_store=int8_store)
        s = U64(seed)
        self.idx = np.empty((n_dim, k_hash), dtype=np.int64)
        for j in range(n_dim):
            x = U64(j) * U64(7919) + np.arange(k_hash, dtype=np.uint64) * U64(2654435761)
            self.idx[j] = (_mix64(x + s) % U64(n_neurons)).astype(np.int64)
        # 反投影索引：大空间索引 → 与之绑定的表示维度（避免召回时逐维扫描 n_dim 次）
        self.rev: dict[int, list[int]] = {}
        for j in range(n_dim):
            for i in self.idx[j].tolist():
                self.rev.setdefault(i, []).append(j)
        self._prev: list[int] | None = None

    def _check_sparse(self, rate: np.ndarray) -> None:
        """A2 契约校验：SparseLTM 接受稀疏率（激活维度 ≪ n_dim）。

        传入 ±1 稠密模式（如 LongTermMemory 的 Hebb 接口）会一次性激活全部维度，
        生成 n_dim×k_hash 个哈希索引，造成性能崩塌 + 语义错误且无任何报错。
        此处显式拦截并提示正确用法。
        """
        dims = int((np.asarray(rate) > 0.0).sum())
        cap = max(1, self.n_dim // 4)
        if dims > cap:
            raise ValueError(
                f"SparseLTM 期望稀疏率输入（激活维度应 ≤ {cap}，实测 {dims}/{self.n_dim}）。"
                f"是否误传了 ±1 稠密模式？请改传稀疏率（rate），"
                f"或使用 LongTermMemory（稠密 Hebb 接口）。")

    def encode(self, rate: np.ndarray) -> list[int]:
        """稀疏率 → 大空间活跃神经元索引（去重，确定性）。"""
        dims = np.nonzero(rate > 0.0)[0]
        if dims.size == 0:
            return []
        out: list[int] = []
        seen = set()
        for j in dims.tolist():
            for i in self.idx[j].tolist():
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        return out

    def imprint(self, rate: np.ndarray) -> None:
        self._check_sparse(rate)                       # A2：契约校验（稠密模式显式报错）
        cur = self.encode(rate)
        if self._prev is not None and cur:
            self.table.learn(self._prev, cur)
            # P66 诊断（2026-09-29）：服务器 1B 档 `M4b_ltm` 段从 0.14 涨到
            # 19.5 ms/tok 且**超线性**——`learn` 的代价是 |prev|×|cur| 次
            # `_find_slot`，而活跃索引数 = f(rate 的稀疏度)，随训练可能变大。
            # 这里按固定间隔打印规模，下一份日志即可把「猜测」变成数据。
            self._diag_n = getattr(self, "_diag_n", 0) + 1
            if self._diag_n % 1000 == 0:
                t = self.table
                rows = len(getattr(t, "keys", getattr(t, "out", {})))
                print(f"[ltm-diag] imprints={self._diag_n} "
                      f"prev={len(self._prev) if self._prev else 0} "
                      f"cur={len(cur)} combos={len(self._prev or []) * len(cur)} "
                      f"rows={rows:,} k_hash={self.k_hash} n_dim={self.n_dim}",
                      flush=True)
        self._prev = cur
        self.table.step_count += 1

    def recall(self, cue: np.ndarray) -> np.ndarray:
        """线索 → 大空间预测 → 反投影为 n_dim 维分数向量（取 k_hash 个索引均值）。

        2026-09-18 优化：改为按「被激活的大空间索引」经反向索引累加，
        复杂度由 O(n_dim·k) 降到 O(活跃索引数·平均绑定维度)，语义不变
        （仍除以 k_hash，保持与原实现逐值一致）。
        """
        self._check_sparse(cue)                        # A2：契约校验
        active = self.encode(cue)
        if not active:
            return np.zeros(self.n_dim)
        scores = self.table.predict(active)
        if not scores:
            return np.zeros(self.n_dim)
        out = np.zeros(self.n_dim)
        n_s = 0
        for big_i, s in scores.items():
            js = self.rev.get(big_i, ())            # 仅遍历被激活索引绑定的维度
            n_s += len(js)
            for j in js:
                out[j] += s
        # P66：召回侧的遍历量（active × 平均绑定维度）也随表增长，一并记录
        self._diag_r = getattr(self, "_diag_r", 0) + 1
        if self._diag_r % 1000 == 0:
            print(f"[ltm-diag] recalls={self._diag_r} active={len(active)} "
                  f"scores={len(scores)} bindings={n_s}", flush=True)
        out /= float(self.k_hash)
        m = float(np.abs(out).max())
        return out / m if m > 0 else out

    def recall_scores(self, cue: np.ndarray) -> np.ndarray:
        """T3.2 top-k 融合用的分级召回分数（与 recall 同动力学，只读）。

        SparseLTM.recall 本就返回 max-abs 归一的分级分数向量，此处直接复用，
        与稠密 LTM 的 recall_scores 接口对齐。
        """
        return self.recall(cue)

    def begin_episode(self) -> None:
        """B4 修复：情节边界——清空跨情景的上一模式指针。

        分段训练（如逐段语料）时，前一段最后一个模式会被关联到后一段第一个模式，
        形成语义上不存在的边。调用本方法切断该伪关联（类比 ContextMemory.begin_episode）。
        """
        self._prev = None

    def consolidate(self, forget: float = 1.0) -> None:
        """大容量表无稠密快/慢权重分离：巩固退化为对已生长突触的轻微衰减。

        2026-09-28 修复：在线 CSR 表（OnlineCSRTable）存储在 keys/vals/size，
        旧实现遍历 self.out（恒为空 dict）→ consolidate **静默无操作**。

        ⚠ int8 量化台阶（诚实边界）：码值域 10~32（q=127）时，1% 乘性衰减
        （forget=1.0）落在量化台阶内、round 后回原码 → consolidate 零效果；
        需 forget ≳ 3 才开始起作用。属量化精度固有属性，非缺陷。
        """
        if forget <= 0.0:
            return
        f = 1.0 - 0.01 * forget
        table = self.table
        if hasattr(table, "vals") and hasattr(table, "size"):      # 在线 CSR 表
            if table.int8_store:
                q = table._q
                w_max = table.w_max
                for i, vals in table.vals.items():
                    n = table.size[i]
                    seg = vals[:n]
                    dec = np.round(np.clip(seg.astype(np.float64) / q * f,
                                           0.0, w_max) * q).astype(np.int16)
                    seg[:] = dec
            else:
                for i, vals in table.vals.items():
                    n = table.size[i]
                    vals[:n] *= f
            return
        if table.int8_store:                                       # dict 版 + int8
            q = table._q
            w_max = table.w_max
            for bucket in table.out.values():
                for k, code in bucket.items():
                    bucket[k] = int(round(min(max(code / q * f, 0.0), w_max) * q))
        else:
            for bucket in table.out.values():
                for k in bucket:
                    bucket[k] *= f
