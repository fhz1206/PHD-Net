"""PHD-Net 主模型 —— M1–M6 的组装与信息流调度（架构文档 §1 / §5）。

单步推理流程（step 方法）：
    1. M1 稀疏编码          4. M5 调制（surprise → gate/模式）
    2. M2 预测编码推理       5. M4a 工作记忆（门控写入）
    3. M3 时序预测           6. M4b 长期记忆（编码/检索）
    7. M6 读出（可选监督）    8. 全模块局部规则学习

注：Readout（M6 读出头）已拆分至 phdnet/readout.py，此处再导出以兼容旧引用。"""

import time
import warnings

import numpy as np

from .config import PHDNetConfig
from .device import use_torch_backend
from .bigltm import SparseLTM
from .sparse_pc import SparsePCStack
from .memory import LongTermMemory, WorkingMemory
from .modulator import Neuromodulator, MultiModulator
from .plasticity import STDPCore
from .readout import Readout
from .sparse_encoder import SparseEncoder

__all__ = ["PHDNet", "Readout"]



class PHDNet:
    """PHD-Net：预测-赫布-双记忆网络。"""

    def __init__(self, cfg: PHDNetConfig):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.encoder = SparseEncoder(cfg.n_input, cfg.n_sdr, cfg.k_sparse, rng)   # M1
        # M2 主干：结构性稀疏 CSR（唯一实现）。2026-09-28 按 fhz 指令删除稠密
        # PredictiveCodingStack（phdnet/pc.py）；cfg.sparse_conn 仅为兼容保留，
        # 传 False 会在 config 校验期 fail-fast。
        self.pc = SparsePCStack(cfg.n_sdr, cfg.n_mid, cfg.n_top,                  # M2（结构性稀疏）
                                cfg.eta_pc, cfg.eta_oja, rng, conn_k=cfg.conn_k,
                                w_max=cfg.pc_w_max,
                                lognormal_init=cfg.lognormal_init,                # O1-4
                                exc_ratio=cfg.exc_ratio)
        # M3 关联核：默认 numpy(+numba)；显式指定或检测到加速器时改用 torch 后端
        # （同一算子语义，覆盖 CPU / CUDA / ROCm(HIP) / 昇腾 NPU）
        # M3 关联核：**只走 numpy(+numba) CPU**（P30 定稿）。
        # 旧 torch 后端分支已随 torch 栈删除（P30）——加速的正确姿势是
        # "只迁移读出"（P19/P28 的 AccelReadout）；STDP 的逐突触状态
        # （自适应 LR / 稳态 / 元可塑性 / E-I）在 torch 上无法保留，迁移会丢机制。
        if use_torch_backend(cfg.backend) and cfg.backend not in ("auto", "numpy", "cpu"):
            raise ValueError(
                f"backend={cfg.backend!r}：M3 STDP 的 torch 后端已随旧 torch 栈删除"
                "（P30，机制保留在 numba CPU 路径）。计算加速请用读出加速"
                "（--accel auto，P19/P28）；本参数如非显式指定请保持默认 auto。")
        self.stdp = STDPCore(cfg.n_top, cfg.m_lateral,                        # M3
                             cfg.lambda_trace, cfg.eta_stdp, cfg.w_max, rng,
                             adaptive=cfg.adaptive_lr, adapt_rho=cfg.adapt_rho,
                             adapt_eps=cfg.adapt_eps, dual=cfg.dual_trace,
                             lam_slow=cfg.lambda_trace_slow,
                             beta_slow=cfg.beta_slow_trace,
                             homeostasis=cfg.stdp_homeostasis,
                             homeo_target=cfg.homeo_target,
                             metaplasticity=cfg.metaplasticity,
                             bcm_tau=cfg.bcm_tau,
                             ei_synapses=cfg.ei_synapses,
                             ei_ratio=cfg.ei_ratio)
        self.wm = WorkingMemory(cfg.n_top, cfg.n_wm_slots, cfg.gamma_wm,          # M4a
                                content_address=cfg.wm_content_address,           # T3.4
                                sim_thresh=cfg.wm_sim_thresh)
        if cfg.big_ltm:                                                           # M4b
            self.ltm = SparseLTM(cfg.n_top, n_neurons=cfg.big_ltm_N,
                                 m_out=cfg.big_ltm_m, k_hash=cfg.big_ltm_k,
                                 seed=cfg.seed,
                                 prune=cfg.sparse_table_prune,
                                 growth_guidance=cfg.growth_guidance,             # T5.2
                                 int8_store=cfg.sparse_int8,                      # T5.3
                                 csr_online=cfg.csr_online,                       # O1：在线可写 CSR
                                 csr_grow_chunk=cfg.csr_grow_chunk)               # O1
        else:
            self.ltm = LongTermMemory(cfg.n_top, cfg.eta_hip, cfg.eta_cortex,     # M4b
                                      cfg.beta_fast, cfg.beta_slow, cfg.n_recall_steps)
        self.modulator = (MultiModulator(cfg.mod_gain, cfg.gain_ach, cfg.gain_ne,
                                         cfg.gain_da, cfg.gain_ht)
                          if cfg.multi_modulation else Neuromodulator(cfg.mod_gain))  # M5
        self.task_mod = Neuromodulator(cfg.task_mod_gain)      # T1.1 任务误差调制器
        self._task_gate = 0.5                # 滞后一步的任务门控（供 T1.3 记忆写入）
        self._last_da = 1.0                  # 脑同构 M5：DA 巩固信号（供 sleep 回放强度）
        self._exp = 0.0                      # 脑同构 M6：发育经验计数（临界期衰减）
        self._last_rate = np.zeros(cfg.n_top)  # 每步无条件更新（供 T3.1 预测特征在评估时保持新鲜）
        n_out = cfg.n_readout if cfg.n_readout > 0 else cfg.n_input               # M6
        self.n_out = int(n_out)      # 读出输出维度：step(x, target=...) 的 target 契约维度
        n_h = cfg.n_top * (3 if cfg.pred_in_readout else 2)     # T3.1 预测通路三拼
        if cfg.retrieval_topk > 0:                              # T3.2 top-k 召回线索
            n_h += cfg.retrieval_topk
        from .backends.accel_readout import pick_readout_backend
        # P19：auto = 有加速器则读出走加速器，否则回落 numba CPU 原路径
        # （无加速器机器上逐位不变；构造失败亦回落并记原因）
        self.readout, self._readout_backend = pick_readout_backend(
            cfg, n_h, n_out, rng)
        self._ro_eta = cfg.eta_readout   # 当前读出学习率（eta_readout_anneal>0 时逐步退火）
        self._prev_rate = np.zeros(cfg.n_top)      # 上一时刻顶层发放率（供 STDP）
        self._prev_pc: dict | None = None          # T1.2：上一步 PC 推理缓存（时序预测目标）
        self._replay_buf: list = []                # PC 突破②：旧 (输入, 目标idx) 储备库
        self._replay_rng = np.random.default_rng(cfg.seed + 0x5EED)
        self.learn_pc = True                        # 发育调度开关（demo 可分阶段控制）
        self.learn_stdp = True
        self._prev_sup: np.ndarray | None = None    # P3：上一时刻支撑集
        self._instab_ema = 1.0                      # P3：支撑集不稳定度 EMA
        self.step_count = 0
        self._sleep_count = 0                       # C2：consolidate 节流计数

        # P35：step 分段计时（诊断 CPU 侧耗时分布；默认关，零开销）
        self._prof_on = bool(getattr(cfg, "step_profiling", False))
        self._prof: dict = {}

    # ---------- P35 分段计时 ----------
    def _prof_t(self, name: str):
        """段起点（未开启 profiling 时返回 None，零开销）。"""
        if self._prof_on:
            return time.perf_counter()
        return None

    def _prof_end(self, name: str, t0) -> None:
        if t0 is not None:
            self._prof[name] = self._prof.get(name, 0.0) + time.perf_counter() - t0

    @staticmethod
    def _rate(r2: np.ndarray) -> np.ndarray:
        """tanh 表示 → 稀疏发放率：半波整流后 k-WTA 竞争（~6% 激活）。

        稀疏化是 M3 STDP 正确工作的前提（生物合理：神经元稀疏放电）；
        低激活率保证不同输入的支撑集区分度，避免表征混淆。
        """
        full = (r2 + 1.0) * 0.5
        k = max(1, len(r2) // 16)
        thresh = np.partition(full, -k)[-k]
        return np.where(full >= thresh, full, 0.0)

    # ---------- 路线图 M9 辅助（全部默认关闭时零参与） ----------
    def _fused_feature(self, rate: np.ndarray) -> np.ndarray | None:
        """T3.2 检索常态化 + top-k 融合（retrieval_topk>0 启用）。

        每步对 LTM 做只读检索，把 |score| 最大的 top-k 分级召回线索
        （归一化到 [-1,1]）作为固定长度的位置无关特征返回 ——
        PFC 读出聚合情景线索（扩散激活的持续背景检索）。
        默认（retrieval_topk=0）返回 None，读出特征与旧行为逐位一致。
        """
        cfg = self.cfg
        if cfg.retrieval_topk <= 0:
            return None
        if cfg.big_ltm:
            cue = rate
        else:
            cue = np.sign(rate)
            cue[cue == 0] = 1.0
        scores = self.ltm.recall_scores(cue)          # 只读检索（两类 LTM 均不改状态）
        k = max(1, min(int(cfg.retrieval_topk), len(scores)))
        a = np.abs(scores)
        top = np.argpartition(-a, k - 1)[:k]
        top = top[np.argsort(-a[top])]                # |score| 降序 → 位置无关特征
        vals = np.asarray(scores[top], dtype=float)
        m = float(np.abs(vals).max())
        if m > 1e-12:
            vals = vals / m
        return vals * float(cfg.retrieval_gain)

    def _build_h(self, r2: np.ndarray, pred_feat: np.ndarray,
                 fused: np.ndarray | None) -> np.ndarray:
        """读出特征拼装（默认路径与旧实现逐位一致：[r2; WM] 或三拼）。"""
        parts = [r2, self.wm.read()]
        if self.cfg.pred_in_readout:
            parts.append(pred_feat)
        if fused is not None:
            parts.append(fused)
        return np.concatenate(parts)

    def _replay_readout(self) -> None:
        """PC 突破②：回放稳定读出（readout_replay=True 启用）。

        从储备库取一条旧 (输入, 目标)，在**当前表征**下只读前向
        （encoder/pc 纯函数；WM/STDP/调制器状态均不被改写），重训读出——
        旧目标把漂移后的表征重新锚定，缓解「表征漂移 × 恒定读出率」的
        灾难性遗忘（M8.A 诊断链的对症机制之一）。唯一被写入的是 readout.W。
        """
        cfg = self.cfg
        x0, t_idx = self._replay_buf[int(self._replay_rng.integers(len(self._replay_buf)))]
        s0, _ = self.encoder.encode(x0)
        c = self.pc.infer(s0, cfg.n_infer_steps)
        r2c, ratec = c["r2"], self._rate(c["r2"])
        pred_feat = self.stdp.predict(self._last_rate)
        fused = self._fused_feature(ratec)
        h = self._build_h(r2c, pred_feat, fused)
        n_out = cfg.n_readout if cfg.n_readout > 0 else cfg.n_input
        tgt = np.zeros(n_out)
        tgt[t_idx] = 1.0
        self.readout.learn_softmax(h, tgt, self._ro_eta * cfg.replay_eta)

    def step(self, x: np.ndarray, target: np.ndarray | None = None,
             learn: bool = True, readonly: bool = False,
             learn_scale: float = 1.0,
             recur_cue: np.ndarray | None = None) -> dict:
        """单个时间步：推理 +（可选）学习。返回诊断信息。

        readonly=True 时冻结所有持久状态（WM/LTM/STDP 迹/调制器/计数器），
        仅做前向推理——供评估复现（B5 修复：避免评估改写模型导致重复评估结果不一致）。
        learn_scale 逐步缩放读出学习率（B6 单遍课程加权）。
        recur_cue：O5 再入连接的上一步读出预测线索（`readout_recurrence=True`
        时由词级 LM 侧计算传入；默认 None = 零参与）。

        默认（readonly=False, learn_scale=1.0, recur_cue=None）行为逐位不变。
        """
        x = np.asarray(x)
        if x.shape[0] != self.cfg.n_input:
            raise ValueError(
                f"step(x) 输入维度不符：期望 cfg.n_input={self.cfg.n_input}，"
                f"实际 {x.shape[0]}。⚠ 注意 x 的维度是 **cfg.n_input**（字符 SDR 的"
                "拼接输入维度），**不是 cfg.k_sparse**（k_sparse 只决定 encoder 输出"
                "SDR 的活跃数）；target 的维度则是 **net.n_out**（读出输出维度）。")
        cfg = self.cfg
        _p = self._prof_t('M1_encode')
        s0, _ = self.encoder.encode(x)                       # 1. M1 稀疏编码
        self._prof_end('M1_encode', _p)
        _p = self._prof_t('M2_infer')
        cache = self.pc.infer(s0, cfg.n_infer_steps)         # 2. M2 预测编码推理
        self._prof_end('M2_infer', _p)
        r2, rate = cache["r2"], self._rate(cache["r2"])
        fused = self._fused_feature(rate)                    # T3.2（默认 None = 零参与）

        _p = self._prof_t('M3_pred')
        pred = self.stdp.predict(self._prev_rate)            # 3. M3 时序预测（诊断指标）
        pred_feat = self.stdp.predict(self._last_rate)       # T3.1 读出预测特征（评估时保持新鲜）
        self._prof_end('M3_pred', _p)
        cos = float(pred @ rate / (np.linalg.norm(pred) * np.linalg.norm(rate) + 1e-9))
        seq_err = 1.0 - cos

        surprise = float(np.linalg.norm(cache["e0"]) / np.sqrt(len(cache["e0"])))
        if not readonly:
            _p = self._prof_t('M5_mod')
            gate, mode = self.modulator.observe(surprise)    # 4. M5 神经调制
            self._prof_end('M5_mod', _p)
            if cfg.multi_modulation:
                self._last_da = self.modulator.da           # DA 巩固信号（供 sleep 回放）
        else:
            gate, mode = 0.0, ""                             # 只读：不更新调制器内部状态

        if not readonly:
            _p = self._prof_t('M4a_wm')
            self.wm.decay()                                  # 5. M4a 工作记忆
            # T1.3 错误驱动写入：任务误差门控（滞后一步）优先于输入惊讶
            mem_gate = self._task_gate if cfg.error_gated_memory else gate
            wm_written = self.wm.write(r2, mem_gate, cfg.gate_thresh)
            # O5 再入连接（readout_recurrence=True，默认关闭）：把上一步读出的
            # 预测线索回灌 WM（皮层反馈再入的抽象）——使下一步读出被"自预测"
            # 条件化，抑制自回归生成中的级联漂移（不引入位置编码/注意力）。
            if cfg.readout_recurrence and recur_cue is not None:
                self.wm.write(np.asarray(recur_cue, dtype=float)
                              * float(cfg.recur_gain), 1.0, 0.0)
        else:
            wm_written = False

        self._prof_end("M4a_wm", _p)
        recall_hit = False                                   # 6. M4b 长期记忆交互
        if not readonly:
            # T3.2-lite 错误触发检索：预测失败时额外检索，每 4 步至多一次
            # （默认关闭 = 每 8 步定频旧行为）
            mem_gate = self._task_gate if cfg.error_gated_memory else gate
            # 脑同构 M5：5-HT 耐心 → 检索节奏（默认每 8 步，高耐心则延长等待）
            retrieve_interval = (max(2, int(round(8 * (0.5 + self.modulator.ht))))
                                 if cfg.multi_modulation else 8)
            retrieve_now = (mode == "retrieve"
                            and self.step_count % retrieve_interval == 0) or (
                cfg.error_triggered_retrieval and mem_gate > 0.8
                and self.step_count % 4 == 0)
            if learn and mode == "encode" and mem_gate > cfg.ltm_imprint_gate:
                p = np.sign(rate); p[p == 0] = 1.0
                # 大容量表按**稀疏率**印迹：±1 稠密码会激活全部维度，破坏事件驱动稀疏性
                _p = self._prof_t('M4b_ltm')
                self.ltm.imprint(rate if cfg.big_ltm else p)     # 预测失败/极新颖 → 快速印迹
                self._prof_end('M4b_ltm', _p)
            elif retrieve_now:
                cue = np.sign(rate); cue[cue == 0] = 1.0
                rec = self.ltm.recall(rate if cfg.big_ltm else cue)  # 吸引子/联想补全
                recall_hit = True
                self.wm.write(rec.astype(float), 0.5, 0.0)

            # P5 摘要槽接线（默认关闭）：每 N 步把当前 WM 内容经吸引子压缩进免衰减保留槽
            if cfg.wm_summary_every > 0 and self.step_count > 0 \
                    and self.step_count % cfg.wm_summary_every == 0:
                self.wm.summarize(self.ltm)

        h = self._build_h(r2, pred_feat, fused)              # 7. M6 读出（拼装见 _build_h）
        # P20：读出计时（后端可能是 numba CPU 或加速器设备）。两次 perf_counter
        # ≈ 0.2 μs，相对读出本身（ms 级）可忽略；只累计时间不改变任何数值。
        _t_ro = time.perf_counter()
        # P36：训练步走**设备张量直通**（`forward_dev`，零同步）——y 的 numpy
        # 化（D2H 289 KiB + 硬同步）只在推理/评估路径需要（`sample_next` 与
        # 评估分支消费 `d["y"]`）。配合 P34 的 nll 设备累积，训练步的 CPU/NPU
        # 完全异步：墙钟从「NPU+CPU 串行相加」趋近 max()。
        _dev_readout = (learn and not readonly
                        and hasattr(self.readout, "forward_dev"))
        if _dev_readout:
            y_dev = self.readout.forward_dev(h)
            y = None                                          # 训练步不需要 numpy
        else:
            y_dev = None
            y = self.readout(h)
        nll = 0.0
        if target is not None and learn:
            if cfg.readout_softmax:
                # P6 性能修复：前向 y 已算得，传给 learn_softmax 省一次 W@h
                # （与内部重算逐位一致；y_pre 不被原地修改，下方返回值不受影响）
                nll = self.readout.learn_softmax(
                    h, target, self._ro_eta * learn_scale, y_pre=(y_dev if _dev_readout else y),
                    accumulate=int(cfg.minibatch_size))   # B7：默认 1 = 逐步更新（逐位不变）
            else:
                self.readout.learn(
                    h, target, self._ro_eta * learn_scale)   # A3 修复：非 softmax 路径同样尊重 eta_readout/退火
        elif target is not None and cfg.pred_in_readout:
            # T3.1 评估路径：直接在训练同款特征上算 NLL（绕开 _nll 回退的二次状态演化）
            yv = y - y.max()
            p_ = np.exp(yv); p_ /= p_.sum()
            nll = float(-np.log(p_[int(np.argmax(target))] + 1e-12))
        self._ro_ms_accum = getattr(self, "_ro_ms_accum", 0.0) \
            + (time.perf_counter() - _t_ro) * 1000.0
        self._ro_calls = getattr(self, "_ro_calls", 0) + 1

        # PC 突破②：储备库登记 + 周期回放（readout_replay=True 才启用；
        # 只写 readout.W，不动任何持久状态，默认路径零参与）
        if cfg.readout_replay and learn and not readonly:
            if target is not None:
                self._replay_buf.append((np.asarray(x, dtype=float).copy(),
                                         int(np.argmax(target))))
                if len(self._replay_buf) > cfg.replay_buffer:
                    self._replay_buf.pop(0)
            if self._replay_buf and self.step_count % max(1, cfg.replay_every) == 0:
                self._replay_readout()

        # T1.1 误差广播调制：读出 NLL 经 Welford z 标准化 → 门控本步主干学习率；
        # 同时更新滞后门控供下一步记忆写入（T1.3）
        if cfg.task_modulation and target is not None and learn:
            task_gate_now, _ = self.task_mod.observe(nll)
            self._task_gate = task_gate_now
        else:
            task_gate_now = self._task_gate
        # 读出学习率退火（0=关闭）：恒定学习率 + 表征漂移会导致后期样本
        # 覆盖早期学习（灾难性遗忘的入口之一），发育后需逐步巩固
        if cfg.eta_readout_anneal > 0.0 and learn and not readonly:
            self._ro_eta = max(cfg.eta_readout_floor,
                               self._ro_eta * cfg.eta_readout_anneal)

        if learn and not readonly:                           # 8. 全模块局部学习（×调制门）
            # 2026-09-28 修复：readonly=True 时此块此前未被冻结（PC/STDP/编码器
            # 学习、task_gate、_prev_rate/_exp 均被改写），与 docstring「冻结所有
            # 持久状态」的承诺不符。现行调用方均为 learn=False+readonly=True 配对，
            # 故默认路径数值不变； learn=True+readonly=True 从此真正只读。
            # P3 自动发育调度：支撑集 Jaccard 变化率 → 表征稳定度
            # 不稳定期：STDP 被门控压制（dev_floor 下限）、PC 快速发育；
            # 表征稳定后：PC 学习自动衰减至 0，STDP 全量放开（双通道互补）
            if cfg.auto_development:
                sup = rate > 0
                if self._prev_sup is not None and sup.any() and self._prev_sup.any():
                    jac = float((sup & self._prev_sup).sum()) / max(
                        1.0, float((sup | self._prev_sup).sum()))
                    instab = 1.0 - jac
                else:
                    instab = 1.0
                self._instab_ema = ((1.0 - cfg.dev_alpha) * self._instab_ema
                                    + cfg.dev_alpha * instab)
                self._prev_sup = sup
                stability = max(0.0, 1.0 - cfg.dev_gain * self._instab_ema)
            else:
                stability = 1.0
            # T1.1：任务误差门控 × 输入惊讶门控（任务开关关闭时 task_scale 恒为 1，保持旧行为）
            if cfg.task_modulation:
                task_scale = (cfg.task_scale_floor
                              + (1.0 - cfg.task_scale_floor) * task_gate_now)
            else:
                task_scale = 1.0
            # 脑同构 M6：发育期临界期——经验累积后学习率衰减（关闭可塑性窗口）
            cp_scale = (1.0 / (1.0 + cfg.crit_rate * self._exp)) if cfg.critical_period else 1.0
            mod_scale = (0.3 + 0.7 * gate) * task_scale * cp_scale
            # PC 突破①：发育期后强制巩固表征（pc_dev_steps>0 时，前 N 步可塑、
            # 之后冻结 PC/编码器——表征停止漂移，读出在稳定表征上继续学习）
            dev_frozen = (cfg.pc_dev_steps > 0 and self.step_count >= cfg.pc_dev_steps)
            if self.learn_pc:
                pc_scale = mod_scale * ((1.0 - stability) if cfg.auto_development else 1.0)
                _p = self._prof_t('PC_learn')
                if not dev_frozen:
                    if cfg.pc_predictive_target and self._prev_pc is not None:
                        # T1.2 预测编码主目标化：生成权重学习「下一时间步」表示
                        self.pc.learn_predictive(
                            self._prev_pc, cache, eta_scale=pc_scale,
                            mix=cfg.pc_pred_mix, homeostasis=cfg.homeostasis)
                    else:
                        self.pc.learn(cache, eta_scale=pc_scale, homeostasis=cfg.homeostasis)
                    # PC 突破③：逐神经元目标发放率（内在可塑性细粒度稳态）
                    if cfg.neuron_target_rate:
                        self.pc.homeostatic_rate(r2, cfg.target_rate, cfg.eta_homeo)
                    # T2.2 可学习稀疏词典：Foldiak 局部规则慢调 W_enc（同一可塑门控）
                    if cfg.learnable_encoder:
                        self.encoder.learn(x, s0, cfg.eta_enc * pc_scale)
                    self._prof_end('PC_learn', _p)
                # 脑同构 M6：突触修剪——PC 权重低于阈值归零（突触消除，降噪+释容）
                if cfg.critical_period:
                    if hasattr(self.pc, "prune_silence"):      # O1-2 结构性稀疏表
                        self.pc.prune_silence(cfg.prune_threshold)
                    else:
                        for W in (self.pc.W_up0, self.pc.W_up1,
                                  self.pc.W_dn0, self.pc.W_dn1):
                            W[np.abs(W) < cfg.prune_threshold] = 0.0
            if self.learn_stdp:
                # 脑同构 M5：NE 新颖性 → 强化 STDP 可塑性增益（默认 1.0 = 不变）
                ne_gain = self.modulator.ne if cfg.multi_modulation else 1.0
                stdp_scale = mod_scale * ne_gain * (
                    (cfg.dev_floor + (1.0 - cfg.dev_floor) * stability)
                    if cfg.auto_development else 1.0)
                _p = self._prof_t('STDP_learn')
                self.stdp.step(self._prev_rate, rate, eta_scale=stdp_scale)
                self._prof_end('STDP_learn', _p)
                # 脑同构 M6：突触修剪——STDP 权重低于阈值归零
                if cfg.critical_period:
                    self.stdp.W[np.abs(self.stdp.W) < cfg.prune_threshold] = 0.0
            self._exp += 1.0
            self._prev_rate = rate
            self._prev_pc = {"s0": cache["s0"], "r1": cache["r1"], "r2": r2}  # T1.2：供下一步时序预测目标
        if not readonly:
            self._last_rate = rate      # T3.1：每步跟踪预测特征（只读时冻结以保证评估可复现）
            self.step_count += 1

        return {"seq_err": seq_err, "surprise": surprise, "gate": gate,
                "mode": mode, "wm_written": wm_written, "recall_hit": recall_hit,
                "nll": nll, "y": y, "r2": r2, "pred": pred, "rate": rate}

    def sleep(self, forget: float = 1.0, n_replay: int = 0,
              replay_ratio: float = 0.5) -> int:
        """睡眠巩固（文档 §5）：快权重 → 慢权重 EMA，可选遗忘旧痕。

        P4 生成式回放：n_replay > 0 时，从慢权重吸引子采样梦境模式并以
        replay_ratio 的强度重新印迹到快权重 —— 新旧知识在快权重中混合，
        模拟睡眠回放对记忆保持的作用。返回回放的模式数。

        脑同构 M4b 分期睡眠（staged_sleep=True）：交替 SWS 与 REM 两阶段——
        SWS = 巩固（consolidate）+ 突触 downscaling（Tononi & Cirelli 突触稳态假说）；
        REM = 生成式回放（slow_only=False，新旧混合重组，模拟联想整合）。
        默认（staged_sleep=False）行为逐位不变：每 sleep 巩固 + 可选回放。
        """
        self._sleep_count += 1
        n_done = 0
        if self.cfg.staged_sleep:
            # 两阶段交替：奇数次 SWS，偶数次 REM
            is_sws = (self._sleep_count % 2 == 1)
            if is_sws:
                self.ltm.consolidate(forget)                 # SWS：快→慢巩固
                if hasattr(self.ltm, "downscale"):
                    self.ltm.downscale(self.cfg.sleep_downscale)   # SWS：突触 downscaling
            elif n_replay > 0 and hasattr(self.ltm, "replay"):
                # REM：生成式回放（slow_only=False 混合重组，联想整合）
                rng = np.random.default_rng(self.cfg.seed + self.step_count)
                # 脑同构 M5：DA 巩固信号 → 回放强度（默认 1.0 = 不变）
                da_scale = self._last_da if self.cfg.multi_modulation else 1.0
                eff = replay_ratio * da_scale
                for dream in self.ltm.replay(n_replay, rng, slow_only=False):
                    self.ltm.W_fast += eff * self.cfg.eta_hip * np.outer(dream, dream)
                    n_done += 1
                self.ltm.W_fast[self.ltm.diag_idx] = 0.0
                np.clip(self.ltm.W_fast, -1.0, 1.0, out=self.ltm.W_fast)
        else:
            # C2 修复：consolidate 遍历全部已生长突触（O(生长突触)），频繁 sleep 会拖慢；
            # 默认每 1 次 sleep 执行（旧行为），可调大 consolidate_every 以摊销开销。
            if self._sleep_count % max(1, self.cfg.consolidate_every) == 0:
                self.ltm.consolidate(forget)
            if n_replay > 0 and hasattr(self.ltm, "replay"):   # 大容量表无逐模式重放接口
                rng = np.random.default_rng(self.cfg.seed + self.step_count)
                for dream in self.ltm.replay(n_replay, rng):
                    self.ltm.W_fast += replay_ratio * self.cfg.eta_hip * np.outer(dream, dream)
                    n_done += 1
                self.ltm.W_fast[self.ltm.diag_idx] = 0.0
                np.clip(self.ltm.W_fast, -1.0, 1.0, out=self.ltm.W_fast)
        return n_done


def to_numpy(x, dtype=None):
    """张量 → numpy：兼容 torch（含 CUDA/NPU/ROCm 设备张量）与惰性数组。

    P17（服务器实测）：昇腾机器上 `np.asarray(npu_tensor)` 抛
    "can't convert npu:0 device type tensor to numpy"，检查点保存与参数统计
    都会踩到；torch 张量须先 `.detach().to('cpu')`。
    """
    if hasattr(x, "detach"):
        x = x.detach().to("cpu").numpy()
    else:
        x = np.asarray(x)
    return x if dtype is None else x.astype(dtype, copy=False)


def _nelem(x) -> int:
    """张量元素数：兼容 numpy / torch（含 CUDA / NPU / ROCm 设备张量）。

    P17（服务器实测）：昇腾机器（CANN 8.5 aarch64 + torch_npu）上模型张量在
    `npu:0`，`np.asarray(tensor)` 直接抛
    "can't convert npu:0 device type tensor to numpy" → 统计参数崩在训练末尾。
    torch 张量须先 `.detach().to('cpu')`（或 `.cpu()`）再 asarray。
    """
    if hasattr(x, "detach"):                     # torch.Tensor（含设备张量）
        try:
            x = x.detach().to("cpu")
        except Exception:                        # 已是 numpy-like 或后端特殊
            return int(x.numel()) if hasattr(x, "numel") else int(np.asarray(x).size)
    try:
        return int(np.asarray(x).size)
    except Exception:                            # 其他惰性数组
        return int(len(x))


def count_params(net: "PHDNet") -> int:
    """统计全部可塑参数（权重 + 记忆存储），供效率报告（C6 修复：集中实现，避免手工统计漂移）。

    口径：编码器 + PC 四权重 + STDP 权重 + 读出权重 + WM 槽位/强度 + LTM 存储
    （稠密 LTM 计入 W_fast/W_slow；大容量表计入「已生长突触数」而非容量上限）。
    不含推理期临时迹/时间戳（视为状态而非参数）—— 报告时须注明此口径。
    """
    n = 0
    n += _nelem(net.encoder.W) + _nelem(net.encoder.b)
    if hasattr(net.pc, "n_synapses"):
        # O1-2：主干结构性稀疏——按**实际存在的突触**计数（非稠密等价规模）
        n += int(net.pc.n_synapses())
    else:
        for w in (net.pc.W_up0, net.pc.W_up1, net.pc.W_dn0, net.pc.W_dn1):
            n += _nelem(w)
    n += _nelem(net.stdp.W)
    n += int(net.readout.n_synapses())    # O1-3：稀疏读出按存在连接计；稠密 = 元素数
    n += _nelem(net.wm.slots) + _nelem(net.wm.strength)
    if net.cfg.big_ltm:
        # O1（2026-09-23）：兼容两种大容量表实现——dict 邻接（out）与在线 CSR（size）
        t = net.ltm.table
        if hasattr(t, "size"):
            n += int(sum(t.size.values()))          # OnlineCSRTable：已生长突触
        else:
            n += sum(len(b) for b in t.out.values())   # SparseSynapseTable：事件驱动生长
    else:
        n += _nelem(net.ltm.W_fast) + _nelem(net.ltm.W_slow)
    return n
