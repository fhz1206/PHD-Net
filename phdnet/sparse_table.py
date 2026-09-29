"""M4 容量层本体 —— 大空间事件驱动稀疏印迹表（N×M_OUT ≈ 1B 突触容量）。

自 bigltm.py 拆分：本文件只负责"容量层存储与 STDP 生长"，
与表示维度对接的适配器见 phdnet/bigltm.py（SparseLTM）。"""

import numpy as np

from .tokenizer import U64


class SparseSynapseTable:
    """大空间事件驱动联想表（容量 = n_neurons × m_out）。

    关键工程点（沿用 train_1b.py 已验证的实现）：
      1. 惰性迹衰减：触碰时才按 λ^Δt 补偿，避免每步扫描全网；
      2. pre/post 迹各自独立时间戳（共用会导致 Δt=0，迹无衰减累加爆炸）；
      3. 相邻对约定下新突触生长只由 LTP 驱动，LTD 仅作用于已存在突触。
    改进（相对 train_1b）：迹与时间戳也用 dict 按需存储，
    内存正比于"被触碰过的神经元"而非 N（2^24 长度数组在该处占 ~200 MB）。
    """

    def __init__(self, n_neurons: int = 1 << 24, m_out: int = 60,
                 lam: float = 0.7, eta: float = 0.08, w_max: float = 1.0,
                 seed: int = 0xA5A55A5A, prune: int = 0,
                 growth_guidance: bool = False, int8_store: bool = False):
        self.n, self.m_out, self.lam, self.eta = n_neurons, m_out, lam, eta
        self.w_max = w_max
        self.seed = U64(seed)
        self.out: dict[int, dict[int, float]] = {}   # 邻接表（已生长突触）
        self.t_pre: dict[int, float] = {}
        self.t_post: dict[int, float] = {}
        self.stamp_pre: dict[int, int] = {}
        self.stamp_post: dict[int, int] = {}
        # C1 修复：迹/时间戳字典惰性淘汰阈值（0=关闭，保持旧行为）。
        # 启用后，被触碰过的神经元条目超阈值时周期性清理最旧迹，避免长程训练内存单调增长。
        self.prune = int(prune)
        self.step_count = 0
        # T5.2 拓扑生长引导（默认关闭）：新突触优先连接低入度目标神经元
        self.growth_guidance = growth_guidance
        self.in_deg: dict[int, int] = {}
        # T5.3 int8 量化存储（默认关闭）：权重以整数 0..127 存储，读取时反量化。
        # 值对象从 Python float（24B）降为缓存小 int，已生长突触内存 ~×8↓。
        self.int8_store = int8_store
        self._q = 127.0 / w_max       # 量化尺度：w = code / _q

    # ---------- T5.3 量化辅助 ----------
    def _enc(self, w: float):
        """写入编码：int8 模式存整数码，否则存浮点。"""
        if self.int8_store:
            return int(round(min(max(w, 0.0), self.w_max) * self._q))
        return float(min(max(w, 0.0), self.w_max))

    def _dec(self, code) -> float:
        """读取解码：int8 模式反量化，否则原样。"""
        return code / self._q if self.int8_store else code

    # ---------- 惰性迹衰减（dict 按需存储） ----------
    def _touch(self, idx: list[int], trace: dict, stamp: dict) -> None:
        t = self.step_count
        lam = self.lam
        for i in idx:
            dt = t - stamp.get(i, 0)
            trace[i] = trace.get(i, 0.0) * (lam ** dt) + 1.0
            stamp[i] = t
        # C1 修复：启用且超阈值时，周期性清理最旧 25% 的迹（按 stamp 升序淘汰）
        if self.prune and len(trace) > self.prune and (t & 1023) == 0:
            self._prune_oldest(trace, stamp)

    @staticmethod
    def _prune_oldest(trace: dict, stamp: dict) -> None:
        """淘汰最旧的一部分迹：下次触碰时该迹从 1.0 重新积累（遗忘旧痕，释放内存）。"""
        drop = sorted(stamp.items(), key=lambda kv: kv[1])[: max(1, len(stamp) // 4)]
        for i, _ in drop:
            trace.pop(i, None)
            stamp.pop(i, None)

    # ---------- 预测：p[k] = Σ_i w(i,k)（只遍历活跃神经元的出边） ----------
    def predict(self, active: list[int]) -> dict[int, float]:
        p: dict[int, float] = {}
        out = self.out
        int8 = self.int8_store
        q = self._q
        for i in active:
            bucket = out.get(i)
            if not bucket:
                continue
            if int8:
                for k, code in bucket.items():
                    p[k] = p.get(k, 0.0) + code / q
            else:
                for k, w in bucket.items():
                    p[k] = p.get(k, 0.0) + w
        return p

    # ---------- 学习：LTP 生长/强化 + LTD 稳态（仅在已存在突触上） ----------
    def learn(self, prev: list[int], cur: list[int]) -> None:
        if not prev or not cur:
            return
        self._touch(cur, self.t_post, self.stamp_post)
        self._touch(cur, self.t_pre, self.stamp_pre)   # 同时开始积累 pre 迹
        out, eta, w_max = self.out, self.eta, self.w_max
        int8 = self.int8_store
        q = self._q
        # T5.2 拓扑生长引导：生长候选按当前入度升序（低入度优先建立连接，
        # 避免 hub 过载；对已存在突触的强化顺序不影响结果——逐键独立更新）
        grow_order = cur
        if self.growth_guidance and len(cur) > 1:
            grow_order = sorted(cur, key=lambda k: self.in_deg.get(k, 0))
        for i in prev:
            tpi = self.t_pre.get(i, 0.0)
            if tpi <= 0.0:
                continue
            bucket = out.get(i)
            for k in grow_order:
                tpk = self.t_post.get(k, 0.0)
                ltp = eta * tpi * tpk
                if ltp <= 0.0:
                    continue
                if bucket is not None and k in bucket:
                    w_now = bucket[k] / q if int8 else bucket[k]
                    ltd = eta * self.t_post.get(i, 0.0) * self.t_pre.get(k, 0.0)
                    bucket[k] = self._enc(min(max(w_now + ltp - ltd, 0.0), w_max))
                else:                                   # 结构可塑性：生长新突触
                    if bucket is None:
                        bucket = out[i] = {}
                    if len(bucket) < self.m_out:
                        bucket[k] = self._enc(min(ltp, w_max))
                        if self.growth_guidance:
                            self.in_deg[k] = self.in_deg.get(k, 0) + 1

    # ---------- T5.3 CSR 扁平快照（读密集阶段的紧凑表示） ----------
    def compact_csr(self) -> dict:
        """把邻接表快照为 CSR（indptr/indices/data），int8 模式下 data 为量化的
        np.int8 数组（同内存容量进一步 ×8↓）。快照只读，不改变在线结构；
        供分析/导出/读密集阶段使用（在线生长仍走 dict 事件驱动路径）。
        """
        n_with_out = sorted(self.out.keys())
        indptr = np.zeros(len(n_with_out) + 1, dtype=np.int64)
        indices: list[int] = []
        data: list[int] = []
        for r, i in enumerate(n_with_out):
            bucket = self.out[i]
            for k in sorted(bucket.keys()):
                indices.append(k)
                data.append(int(bucket[k]) if self.int8_store
                            else int(round(bucket[k] * self._q)))
            indptr[r + 1] = len(indices)
        return {
            "row_ids": np.asarray(n_with_out, dtype=np.int64),
            "indptr": indptr,
            "indices": np.asarray(indices, dtype=np.int64),
            "data": (np.asarray(data, dtype=np.int8) if self.int8_store
                     else np.asarray(data, dtype=np.float64) / self._q),
            "scale": 1.0 / self._q,
        }

    def stats(self) -> dict:
        grown = sum(len(b) for b in self.out.values())
        d = {
            "capacity": self.n * self.m_out,
            "grown_synapses": grown,
            "neurons_with_out": len(self.out),
            "utilization": grown / (self.n * self.m_out),
            "touched_neurons": len(self.t_pre),
        }
        if self.growth_guidance and self.in_deg:
            degs = np.asarray(list(self.in_deg.values()))
            d["in_deg_mean"] = float(degs.mean())
            d["in_deg_max"] = int(degs.max())
        return d


class OnlineCSRTable(SparseSynapseTable):
    """O1/O3（2026-09-22）：**在线可写 CSR** 稀疏印迹表。

    与 `SparseSynapseTable` 接口完全对齐（`predict` / `learn` / `stats` /
    `compact_csr`），但邻接存储由 dict-of-dict 换成「定长行 + 预留槽」的 CSR 风格数组：

        keys[i] : np.int64   长度 row_cap[i]，前 size[i] 项为有效出边目标
        vals[i] : np.float64 或 np.int16（int8 量化码），同长度前 size[i] 项有效

    动机（对应 O3）：dict 版每条目 = Python dict entry + boxed int + boxed float
    ≈ 100+ B；CSR 版每条目 = 8 B（int64 目标）+ 2 B（int16 码）或 8 B（float）
    → 结构化紧凑表示，是「100B 在线 CSR 迁移」的前提（本机可验证到 1B 容量）。

    逐位等价（关键设计）：行的槽位顺序 = 首次生长顺序，与 dict 版的插入顺序一致；
    `predict` 按槽位顺序累加、`learn` 逐键独立更新 → 与 dict 版**逐位相同**，
    可用 `tests/verifiers/verify_csr_equiv.py` 对拍验证。

    行扩容：满 `row_cap` 后按 2× 增长（amortized O(1)），上限 m_out（硬容量）。
    """

    def __init__(self, n_neurons: int = 1 << 24, m_out: int = 60,
                 lam: float = 0.7, eta: float = 0.08, w_max: float = 1.0,
                 seed: int = 0xA5A55A5A, prune: int = 0,
                 growth_guidance: bool = False, int8_store: bool = False,
                 grow_chunk: int = 64):
        super().__init__(n_neurons, m_out, lam, eta, w_max, seed, prune,
                         growth_guidance, int8_store)
        # 注意：父类的 self.out 保留为空 dict（不参与本类逻辑，仅为接口兼容）。
        self.grow_chunk = max(1, min(int(grow_chunk), m_out))
        cap0 = min(self.grow_chunk, m_out)
        self.keys: dict[int, np.ndarray] = {}     # 行 i → 目标索引数组
        self.vals: dict[int, np.ndarray] = {}     # 行 i → 权重（或量化码）数组
        self.size: dict[int, int] = {}            # 行 i → 有效槽数
        self._cap0 = cap0

    # ---------- 行级操作 ----------
    def _new_row(self, i: int) -> np.ndarray:
        dt = np.int16 if self.int8_store else np.float64
        self.keys[i] = np.zeros(self._cap0, dtype=np.int64)
        self.vals[i] = np.zeros(self._cap0, dtype=dt)
        self.size[i] = 0
        return self.keys[i]

    def _grow_row(self, i: int) -> None:
        sz = self.size[i]
        new_cap = min(len(self.keys[i]) * 2, self.m_out)
        k2 = np.zeros(new_cap, dtype=np.int64)
        v2 = np.zeros(new_cap, dtype=self.vals[i].dtype)
        k2[:sz] = self.keys[i][:sz]
        v2[:sz] = self.vals[i][:sz]
        self.keys[i], self.vals[i] = k2, v2

    def _find_slot(self, i: int, k: int):
        """线性扫描定位键（m_out ≤ 60，扫描成本可忽略；避免额外 dict 开销）。"""
        idx = self.keys[i]
        for s in range(self.size[i]):
            if int(idx[s]) == k:
                return s
        return None

    def _write_slot(self, i: int, slot: int, w: float) -> None:
        if self.int8_store:
            self.vals[i][slot] = np.int16(int(round(min(max(w, 0.0), self.w_max) * self._q)))
        else:
            self.vals[i][slot] = np.float64(min(max(w, 0.0), self.w_max))

    def _append(self, i: int, k: int, w: float) -> None:
        sz = self.size[i]
        if sz >= len(self.keys[i]):
            self._grow_row(i)
        self.keys[i][sz] = k
        self._write_slot(i, sz, w)
        self.size[i] = sz + 1

    # ---------- 预测（与 dict 版逐位同序累加） ----------
    # ---------- 预测：p[k] = Σ_i w(i,k)（只遍历活跃神经元的出边） ----------
    def predict_arr(self, active):
        """P70：`predict` 的数组版——返回 (keys, vals)，**不做同键合并**。

        `SparseLTM.recall` 用它 + `ltm_kernel._recall_project`（numba）替代
        「dict predict + Python 双循环」：重复键在核内按「行序 → 槽位序」逐次
        累加，浮点顺序与原 `p[k] += v` 完全一致 → **逐位相同**。

        为什么这一步不用 numba 核：服务器实测 active=256 行 × 约 72 槽
        ≈ **1.8 万元素**（P67 的 learn 场景是 64 万组合，小两个数量级），
        Python 开销只有「每行一次切片」→ 纯 numpy gather 就够（微秒级），
        上核的固定开销反而吃掉收益（P11/P12/P13 三次负收益的教训）。
        """
        ks, vs = [], []
        keys, vals, size = self.keys, self.vals, self.size
        for i in active:
            idx = keys.get(i)
            if idx is None:
                continue
            sz = size[i]
            if sz <= 0:
                continue
            ks.append(idx[:sz])
            vs.append(vals[i][:sz])
        if not ks:
            return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
        k_arr = np.concatenate(ks)
        v_arr = np.concatenate(vs).astype(np.float64)
        if self.int8_store:                     # 码值 → 权重（与 `code / q` 一致）
            v_arr /= float(self._q)
        return k_arr, v_arr

    def predict(self, active: list[int]) -> dict[int, float]:
        p: dict[int, float] = {}
        int8 = self.int8_store
        q = self._q
        keys, vals, size = self.keys, self.vals, self.size
        for i in active:
            idx = keys.get(i)
            if idx is None:
                continue
            sz = size[i]
            row_v = vals[i]
            if int8:
                for s in range(sz):
                    k = int(idx[s])
                    p[k] = p.get(k, 0.0) + int(row_v[s]) / q
            else:
                for s in range(sz):
                    k = int(idx[s])
                    p[k] = p.get(k, 0.0) + float(row_v[s])
        return p

    # ---------- 学习（LTP 生长/强化 + LTD 稳态；语义与 dict 版逐键一致） ----------
    def learn(self, prev: list[int], cur: list[int]) -> None:
        if not prev or not cur:
            return
        self._touch(cur, self.t_post, self.stamp_post)
        self._touch(cur, self.t_pre, self.stamp_pre)
        eta, w_max, int8, q = self.eta, self.w_max, self.int8_store, self._q
        # 与 dict 版一致：生长顺序在 prev 循环外确定一次（按当前入度升序）
        grow_order = cur
        if self.growth_guidance and len(cur) > 1:
            grow_order = sorted(cur, key=lambda k: self.in_deg.get(k, 0))
        for i in prev:
            tpi = self.t_pre.get(i, 0.0)
            if tpi <= 0.0:
                continue
            row = self.keys.get(i)
            for k in grow_order:
                tpk = self.t_post.get(k, 0.0)
                ltp = eta * tpi * tpk
                if ltp <= 0.0:
                    continue
                if row is None:
                    row = self._new_row(i)
                slot = self._find_slot(i, k)
                if slot is not None:
                    w_now = (int(self.vals[i][slot]) / q) if int8 else float(self.vals[i][slot])
                    ltd = eta * self.t_post.get(i, 0.0) * self.t_pre.get(k, 0.0)
                    self._write_slot(i, slot, w_now + ltp - ltd)
                elif self.size[i] < self.m_out:          # 结构可塑性：生长新突触
                    self._append(i, k, min(ltp, w_max))
                    if self.growth_guidance:
                        self.in_deg[k] = self.in_deg.get(k, 0) + 1

    # ---------- CSR 快照（与本类内部表示同构，直接导出） ----------
    def compact_csr(self) -> dict:
        """与 dict 版逐位一致：每行按目标索引**排序**输出（语义对齐
        `SparseSynapseTable.compact_csr` 的 `for k in sorted(bucket.keys())`）。"""
        rows = sorted(self.keys.keys())
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        indices: list[int] = []
        data: list[int] = []
        for r, i in enumerate(rows):
            sz = self.size[i]
            idx, v = self.keys[i], self.vals[i]
            for k, s in sorted((int(idx[s]), s) for s in range(sz)):
                indices.append(k)
                data.append(int(v[s]) if self.int8_store
                            else int(round(float(v[s]) * self._q)))
            indptr[r + 1] = len(indices)
        return {
            "row_ids": np.asarray(rows, dtype=np.int64),
            "indptr": indptr,
            "indices": np.asarray(indices, dtype=np.int64),
            "data": (np.asarray(data, dtype=np.int8) if self.int8_store
                     else np.asarray(data, dtype=np.float64) / self._q),
            "scale": 1.0 / self._q,
        }

    def stats(self) -> dict:
        grown = sum(self.size.values())
        alloc = sum(len(v) for v in self.keys.values())
        d = {
            "capacity": self.n * self.m_out,
            "grown_synapses": grown,
            "neurons_with_out": len(self.keys),
            "utilization": grown / (self.n * self.m_out),
            "touched_neurons": len(self.t_pre),
            "row_slots_allocated": alloc,     # 预留槽（含未用）
            "slot_efficiency": grown / max(1, alloc),
        }
        if self.growth_guidance and self.in_deg:
            degs = np.asarray(list(self.in_deg.values()))
            d["in_deg_mean"] = float(degs.mean())
            d["in_deg_max"] = int(degs.max())
        return d

    # ---------- O3：分片落盘 / mmap 读回（大容量表的物理分片存储） ----------
    def export_shards(self, out_dir: str, n_shards: int = 8) -> dict:
        """把在线 CSR 表按行哈希分片写入 n_shards 个 np.memmap 文件（O3 存储路径）。

        每片一个 `.npz`（元数据）+ 一组定长数组；写回后 `import_shards` 可
        逐位恢复。返回各片条目数统计。
        """
        from pathlib import Path
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        shards: list[list[int]] = [[] for _ in range(n_shards)]
        for i in sorted(self.keys.keys()):
            shards[i % n_shards].append(i)
        counts = []
        for si, rows in enumerate(shards):
            arrs_k, arrs_v, sizes, indptr = [], [], [], [0]
            for i in rows:
                sz = self.size[i]
                arrs_k.append(self.keys[i][:sz])
                arrs_v.append(self.vals[i][:sz])
                sizes.append(sz)
                indptr.append(indptr[-1] + sz)
            flat_k = (np.concatenate(arrs_k) if arrs_k else np.zeros(0, dtype=np.int64))
            flat_v = (np.concatenate(arrs_v) if arrs_v else
                      np.zeros(0, dtype=np.int16 if self.int8_store else np.float64))
            np.savez(str(d / f"shard_{si:02d}.npz"),
                     row_ids=np.asarray(rows, dtype=np.int64),
                     sizes=np.asarray(sizes, dtype=np.int64),
                     indptr=np.asarray(indptr, dtype=np.int64),
                     keys=flat_k, vals=flat_v)
            counts.append(int(len(flat_k)))
        return {"shards": n_shards, "entries_per_shard": counts,
                "total_entries": int(sum(counts)), "dir": str(d)}

    def import_shards(self, in_dir: str) -> None:
        """从 `export_shards` 产物逐位恢复在线 CSR 状态（覆盖当前 keys/vals/size）。"""
        from pathlib import Path
        d = Path(in_dir)
        self.keys, self.vals, self.size = {}, {}, {}
        for f in sorted(d.glob("shard_*.npz")):
            with np.load(str(f)) as z:
                rows = z["row_ids"]; sizes = z["sizes"]
                indptr = z["indptr"]; keys = z["keys"]; vals = z["vals"]
                for r, i in enumerate(rows.tolist()):
                    a, b = int(indptr[r]), int(indptr[r + 1])
                    self.keys[i] = np.array(keys[a:b], dtype=np.int64)
                    self.vals[i] = np.array(vals[a:b], dtype=vals.dtype)
                    self.size[i] = int(sizes[r])
