"""M4 容量层适配器 —— 把 n_dim 维表示经确定性哈希接入大空间印迹表。

存储本体（SparseSynapseTable）已拆分至 phdnet/sparse_table.py，
此处保留 SparseLTM 并再导出 SparseSynapseTable 以兼容旧引用。"""

import numpy as np

from .ltm_kernel import NUMBA_LTM, _recall_project
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
        # P68：rev 是**静态**的（上面一次性构建完就不再变），所以零维护成本地
        # 预 CSR 化，供 numba 核 `_recall_project` 使用。维度序保持与 dict 的
        # list 序一致（j 升序）→ 与 Python 版逐位相同。
        self._rev_indptr, self._rev_indices = self._rev_to_csr()
        # P95：预构造 `idx` 的 python 列表形式。`encode` 每步对每个活跃维度做
        # `self.idx[j].tolist()`（numpy 花式索引 + 列表转换）——实测 256 活跃维
        # 时 183.7 µs/次，而 encode 在 imprint 与 recall 里**每步**都调（审计 M4）。
        # 构造期一次建好（n_dim 个小 list），循环内变成**零分配**的 list 取用。
        self._idx_lists: list[list[int]] = [
            self.idx[j].tolist() for j in range(n_dim)]
        self._prev: list[int] | None = None

    def _rev_to_csr(self):
        """把 `rev`（dict[big_i] → [维度…]）转成 (indptr, indices) CSR 数组。

        维度序 = 原 list 序（j 升序）；`indptr` 按 big_i 升序排布。数组只覆盖
        **rev 实际有键的域**（n_keys = max(键)+1）——查询时越界的 big_i 由核内
        边界检查跳过，等价于原 dict `.get(big_i, ())` 返回空元组。
        """
        max_key = 0
        if self.rev:
            max_key = max(max_key, max(self.rev))
        max_key = max(max_key, int(self.idx.max()) if self.idx.size else 0)
        n_keys = max_key + 1
        indptr = np.zeros(n_keys + 1, dtype=np.int64)
        if not self.rev:
            return indptr, np.empty(0, dtype=np.int64)
        # 审计 B2：原实现 `for i in range(n_keys)` 逐个查 dict —— 1B 档
        # n_keys = 16,774,731，即 **1677 万次 Python 循环**（本机 ~10 s 启动）。
        # 改为**只遍历实际存在的 key**（≤ n_dim·k_hash = 4096 条），并保持
        # **key 升序**（与原实现 range(n_keys) 的段序完全一致 → 逐位相同）。
        ks = np.array(sorted(self.rev), dtype=np.int64)
        cnts = np.fromiter((len(self.rev[int(i)]) for i in ks),
                           dtype=np.int64, count=len(ks))
        indptr[ks + 1] = cnts
        indptr = np.cumsum(indptr)                  # 稠密前缀和
        indices = (np.concatenate([np.asarray(self.rev[int(i)], dtype=np.int64)
                                   for i in ks]) if ks.size
                   else np.empty(0, dtype=np.int64))
        return indptr, indices

    def _check_sparse(self, rate: np.ndarray) -> None:
        """A2 契约校验：SparseLTM 接受稀疏率（激活维度 ≪ n_dim）。
        传入 ±1 稠密模式（如 LongTermMemory 的 Hebb 接口）会一次性激活全部维度，
        生成 n_dim×k_hash 个哈希索引，造成性能崩塌 + 语义错误且无任何报错。
        此处显式拦截并提示正确用法。
        """
        dims = int((np.asarray(rate) > 0.0).sum())
        # P66：记录 rate 的活跃维度数——`M4b_ltm` 的组合数 = f(活跃维度)，
        # 若它随训练单调上升，说明「稀疏度退化」是设计缺口（而非某次改动的
        # bug）；同时也能反查 bf16 读出退化是否间接推高了活跃度。
        self._last_dims = dims
        cap = max(1, self.n_dim // 4)
        if dims > cap:
            raise ValueError(
                f"SparseLTM 期望稀疏率输入（激活维度应 ≤ {cap}，实测 {dims}/{self.n_dim}）。"
                f"是否误传了 ±1 稠密模式？请改传稀疏率（rate），"
                f"或使用 LongTermMemory（稠密 Hebb 接口）。")

    def encode(self, rate: np.ndarray) -> list[int]:
        """稀疏率 → 大空间活跃神经元索引（去重，确定性）。

        ⚠ P68 试过 numba 化（`ltm_kernel._encode_hash_uniq`），实测 **0.84×
        负收益**并已回滚：核内去重是 O(n²) 线性扫描，输给 Python `set` 的
        O(1) 哈希（200 维 → 799 索引时 304 → 362 µs）。要 numba 化得改成
        排序+相邻去重或小型开放寻址哈希表（参照 P18 的词表合并做法）。
        """
        dims = np.nonzero(rate > 0.0)[0]
        if dims.size == 0:
            return []
        # P95b：去重才是瓶颈（1024 次 `in` + append 的 Python 循环），
        # `dict.fromkeys` 在 C 层完成「保序去重」——顺序与逐个 `if i not in seen`
        # 完全一致（dict 保序），故**逐位相同**。
        # P98 实测分解（256 活跃维，约 58 µs 合计）：`rate>0`+`nonzero` 仅 2.3 µs
        # （**不是瓶颈**）、`dims.tolist()` 1.3、list-of-lists 5.7、chain 13.8、
        # **dict.fromkeys 38.2（最大头，且是必要语义）**。已试并**否决**的替代：
        # np.unique(return_index) 保序版 94.3 µs（更慢，因需 argsort + gather）、
        # 手写 set 循环（回到 Python 层）。**结论：这里已是最优，不要再"优化"。**
        from itertools import chain
        idx_lists = self._idx_lists
        return list(dict.fromkeys(
            chain.from_iterable([idx_lists[j] for j in dims.tolist()])))

    def imprint(self, rate: np.ndarray) -> None:
        self._check_sparse(rate)                       # A2：契约校验（稠密模式显式报错）
        cur = self.encode(rate)
        if self._prev is not None and cur:
            self.table.learn(self._prev, cur)
            # P66 诊断（2026-09-29）：服务器 1B 档 `M4b_ltm` 段从 0.14 涨到
            # 19.5 ms/tok 且**超线性**——`learn` 的代价 = |prev|×|cur| 次
            # `_find_slot`（且 `ltp<=0` 短路永不生效，因为 `_touch` 把 cur 每个
            # 键都置 ≥1.0）；两者都来自 `encode(rate)`，随表示稠密化变大。
            # ⚠ 门槛为每 10 次（fhz 2026-09-29 指示）：imprint 是**条件触发**
            # （mode=="encode" 且 gate 达标），设 1000 次时 3500 token 一行都
            # 不出；recall 同理。
            self._diag_n = getattr(self, "_diag_n", 0) + 1
            if self._diag_n % 10 == 0:
                t = self.table
                rows = len(getattr(t, "keys", getattr(t, "out", {})))
                print(f"[ltm-diag] imprints={self._diag_n} "
                      f"prev={len(self._prev) if self._prev else 0} "
                      f"cur={len(cur)} combos={len(self._prev or []) * len(cur)} "
                      f"rate_dims={getattr(self, '_last_dims', 0)}/{self.n_dim} "
                      f"rows={rows:,} k_hash={self.k_hash}", flush=True)
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
        out = np.zeros(self.n_dim)
        n_s = 0
        if NUMBA_LTM and hasattr(self.table, "predict_arr"):
            # P70：整条 recall 路径零 Python 内层循环 ——
            #   ①`predict_arr`（numpy gather，1.8 万元素级）替代 dict predict
            #     （原版每步 256 行 × 72 槽 ≈ 1.8 万次 dict 更新 ≈ 20–50 ms，
            #      服务器实测 M4b_ltm 占 63 ms/tok 的主因）；
            #   ②`_recall_project`（numba nogil）做反投影累加。
            # 重复键**不合并**、按「行序 → 槽位序」在核内逐次累加 → 与原
            # `p[k] += v` 浮点顺序完全一致 → 逐位相同。
            ks, ws = self.table.predict_arr(active)
            if ks.size == 0:
                return np.zeros(self.n_dim)
            _recall_project(ks, ws, out, self._rev_indptr, self._rev_indices)
            n_scores = int(ks.size)
        else:
            scores = self.table.predict(active)
            if not scores:
                return np.zeros(self.n_dim)
            n_scores = len(scores)
            ks = np.fromiter(scores.keys(), dtype=np.int64, count=n_scores)
            for big_i, s in scores.items():
                js = self.rev.get(big_i, ())            # 仅遍历被激活索引绑定的维度
                n_s += len(js)
                for j in js:
                    out[j] += s
        # P66：召回侧的遍历量（active × 平均绑定维度）也随表增长，一并记录
        # （门槛同 imprint：recall 同样是条件触发，不能设 1000）
        self._diag_r = getattr(self, "_diag_r", 0) + 1
        if self._diag_r % 10 == 0:
            # ⚠ 审计 B1：`rev_indptr` 按大空间最大 key 铺开 → 1B 档长
            # 16,774,731（134 MB）。`np.diff(indptr)` 会**完整物化 134 MB**
            # （本机 33 ms，每 8 步一次 ≈ 6 ms/tok）——只为算这一行诊断。
            # 改为按需取区间长度（O(|valid|)，数值完全相同）**且只在打印时算**。
            n_rev = self._rev_indptr.shape[0] - 1
            valid = ks[ks < n_rev] if ks.size else ks
            n_s = (int((self._rev_indptr[valid + 1]
                        - self._rev_indptr[valid]).sum())
                   if valid.size else 0)
            print(f"[ltm-diag] recalls={self._diag_r} active={len(active)} "
                  f"scores={n_scores} bindings={n_s}", flush=True)
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
