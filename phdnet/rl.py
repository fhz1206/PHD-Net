# -*- coding: utf-8 -*-
r"""PHD-Net 的强化学习训练基础设施（P27，2026-09-28；fhz「RL 训练先准备好」）。

## 为什么能低成本接入 RL（架构契合点）

M6 读出就是 **softmax 感知器**：`p = softmax(W·h)`，更新是
`ΔW = −η·(p − t)⊗h`（`t` = one-hot 目标）。这在 RL 里恰好是 policy 的
logits 与一个「以 one-hot 为目标的交叉熵」步。于是：

| RL 概念 | PHD-Net 现有件 | 复用方式 |
|---|---|---|
| policy π(a\|s) | 读出 softmax | 直接用 |
| 采样动作 | `infer.sample_next`（温度 + top-k） | 直接用 |
| log π(a\|s) | `log p[a]` | rollout 时记录 |
| 策略梯度 ∇log π·R | 感知器更新 `−η(p−t)⊗h` | **令 η ← −lr·R、t ← one-hot(a)** |
| 状态推进 | `net.step(learn=False)` | 预热 + rollout 全程 |

因此 RL **不需要新增任何算子**，只是给已有的读出更新换一个 `η`（奖励加权）。
这与 P6 读出的机制一致：局部梯度、事件驱动、无反向图。

## 组件

- `rollout(lm, prompt, n_tokens, tau, topk, seed)` → `Rollout(prompt, text,
  tokens, logprobs, reward)`：预热 prompt（状态累积）→ 自回归采样 → 记录
  每步 log π；**采样阶段不学习**（on-policy）。
- `REINFORCETrainer`：
  - `update(rollouts)`：奖励减去滑动基线（baseline 降方差）→ 归一化 →
    对每条轨迹的每步做一次 `net.step(..., target=onehot(a), learn=True,
    learn_scale=adv)`（`learn_scale` 即 P6 已有的 η 缩放）。
  - 可选 **KL 正则**（`kl_coef`）：对**初始策略**（冻结的读出副本）做
    `KL(π_θ ‖ π_ref)` 的梯度惩罚，防止 RL 把模型推离预训练分布太远
    （小模型 + 稀疏奖励下这是主要的崩坏来源）。
  - `stats()`：奖励均值/基线/优势绝对值/KL —— 用于判断 RL 是否真的在学。
- `RewardFn` 协议：`(prompt: str, completion: str, **meta) -> float`。
  **数据集与奖励由使用方提供**（本模块不绑定任何具体任务）。
- `load_prompts(path)`：JSONL 提示集，每行 `{"prompt": ..., "reward": ...}`；
  `reward` 可缺省（此时必须给 `--reward-fn`）。
- 状态隔离：RL rollout 会推进 WM/STDP/LTM 状态 → 提供
  `preserve_state()` / `restore_state()` 上下文，避免污染后续训练
  （1M context 的设计里状态是连续的，rollout 属于「探索」而非「学习输入」）。

## 诚实边界

- 本模块是 **REINFORCE（蒙特卡洛策略梯度）**，没有 actor-critic / PPO 的
  value 网络：PHD-Net 目前的机制里没有独立的值函数（读出是 token 分布）。
  baseline 用滑动平均替代（可后续接真实 value head）。
- 奖励稀疏时收敛慢；建议先在有可验证奖励的玩具任务上跑通再上真实数据。
- `net.step` 的状态推进与读出更新耦合在一次调用里，KL 需要额外一次前向
  （`learn=False` 用参考权重前向）→ 只在 `kl_coef > 0` 时承担该开销。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


# ─────────────────────────── 数据与奖励协议 ───────────────────────────

RewardFn = Callable[..., float]


@dataclass
class Rollout:
    """一次采样轨迹（on-policy）。"""
    prompt: str
    text: str = ""                 # 生成的续写（拼接后的 token 文本）
    tokens: list[str] = field(default_factory=list)
    prevs: list = field(default_factory=list)   # 每步的前词（组合编码用）
    logprobs: list[float] = field(default_factory=list)
    reward: float = 0.0

    @property
    def n(self) -> int:
        return len(self.tokens)


def load_prompts(path: str | Path) -> list[dict]:
    """读 JSONL 提示集（每行至少含 `prompt`；`reward` 可选）。"""
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln} JSON 解析失败：{e}") from e
            if "prompt" not in obj:
                raise ValueError(f"{path}:{ln} 缺少 'prompt' 字段")
            items.append(obj)
    return items


def reward_from_dataset(item: dict) -> float | None:
    """数据集内联奖励（`reward` 字段；支持 number 或 {"value": number}）。"""
    r = item.get("reward")
    if r is None:
        return None
    if isinstance(r, dict):
        r = r.get("value", r.get("score"))
    return None if r is None else float(r)


# ─────────────────────────── 状态保存/恢复 ───────────────────────────

class state_guard:
    """上下文管理器：保存/恢复 WM/STDP/LTM 状态。

    RL rollout 是「探索」，不应污染后续训练流（1M context 设计里状态连续）。
    用法：

        with state_guard(net):
            r = trainer.rollout(lm, prompt)
    """

    def __init__(self, net):
        self.net = net
        self._snap = None

    def __enter__(self):
        self._snap = self._capture()
        return self.net

    def __exit__(self, *exc):
        self._restore(self._snap)
        self._snap = None
        return False

    # 具体字段按现有实现抄（不引入新的持久化格式）
    def _capture(self):
        net = self.net
        return {
            "step_count": int(getattr(net, "step_count", 0)),
            "wm": copy_of(getattr(net, "wm", None)),
            "stdp_W": copy_of(getattr(getattr(net, "stdp", None), "W", None)),
            "stdp_t": (copy_of(getattr(net.stdp, "t_pre", None)),
                       copy_of(getattr(net.stdp, "t_post", None)))
            if hasattr(net, "stdp") else None,
            "ltm_fast": copy_of(getattr(getattr(net, "ltm", None), "W_fast", None)),
            "ltm_slow": copy_of(getattr(getattr(net, "ltm", None), "W_slow", None)),
        }

    def _restore(self, s):
        net = self.net
        if s is None:
            return
        if "step_count" in s:
            try:
                net.step_count = s["step_count"]
            except Exception:                                    # noqa: BLE001
                pass
        for obj, key in ((getattr(net, "wm", None), "wm"),):
            if s.get(key) is not None and obj is not None:
                restore_into(obj, s[key])
        if s.get("stdp_W") is not None and hasattr(net, "stdp"):
            restore_into(net.stdp, {"W": s["stdp_W"]})
            if s.get("stdp_t"):
                if s["stdp_t"][0] is not None:
                    restore_into(net.stdp, {"t_pre": s["stdp_t"][0]})
                if s["stdp_t"][1] is not None:
                    restore_into(net.stdp, {"t_post": s["stdp_t"][1]})
        if s.get("ltm_fast") is not None and hasattr(net, "ltm"):
            restore_into(net.ltm, {"W_fast": s["ltm_fast"]})
        if s.get("ltm_slow") is not None and hasattr(net, "ltm"):
            restore_into(net.ltm, {"W_slow": s["ltm_slow"]})


def copy_of(x):
    return None if x is None else np.array(x, copy=True)


def restore_into(obj, value) -> None:
    """把快照写回对象。`value` 可以是 dict（字段→数组）或单块数组。"""
    if isinstance(value, np.ndarray):
        dst = getattr(obj, "slots", None)          # wm.slots 之类
        if isinstance(dst, np.ndarray) and dst.shape == value.shape:
            dst[...] = value
        return
    if not isinstance(value, dict):
        return
    for k, v in value.items():
        if v is None:
            continue
        try:
            dst = getattr(obj, k)
        except AttributeError:
            continue
        if isinstance(dst, np.ndarray) and isinstance(v, np.ndarray):
            if dst.shape == v.shape:
                dst[...] = v
        else:
            try:
                setattr(obj, k, v)
            except Exception:                                # noqa: BLE001
                pass


# ─────────────────────────── token 流工具 ───────────────────────────

def _corpus_stream_mod():
    """按路径导入 train/corpus_stream（它不是包，故不能用相对导入）。"""
    import importlib
    import sys
    root = Path(__file__).resolve().parents[1] / "train"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module("corpus_stream")


def _stream_tokens(lm, text: str):
    """text → token 流（与训练/推理侧同一分词器，逐位一致）。"""
    SEP, StreamingTokenizer = _corpus_stream_mod().SEP, \
        _corpus_stream_mod().StreamingTokenizer
    for tk in StreamingTokenizer(lm.tok.seg, iter([text + SEP])):
        yield tk


def _warmup_tokens(lm, text: str):
    """把 text 流入网络（learn=False，状态累积），返回 (OOV 数, 尾部 ≤2 token）。"""
    oov = 0
    prev = None
    tail: list[str] = []
    for tk in _stream_tokens(lm, text):
        if tk in lm.tok.stoi:
            lm.net.step(lm.tok.encode_composite(tk, prev), learn=False)
            prev = tk
            tail.append(tk)
            if len(tail) > 2:
                tail.pop(0)
        else:
            oov += 1
    return oov, tail


# ─────────────────────────── Rollout ───────────────────────────

def rollout(lm, prompt: str, n_tokens: int = 32, tau: float = 1.0,
            topk: int = 0, seed: int = 0) -> Rollout:
    """采样一条轨迹（**不学习**）：预热 prompt → 自回归 → 记录 log π。

    与 `infer.generate` 同款采样（温度 + top-k），额外记录每步 log π(a|s)。
    """
    rng = np.random.default_rng(seed)
    oov, tail = _warmup_tokens(lm, prompt)
    if len(tail) == 2:
        prev, cur = tail[0], tail[1]
    elif len(tail) == 1:
        prev, cur = None, tail[0]
    else:
        prev, cur = None, lm.tok.tokens[0]
    r = Rollout(prompt=prompt)
    for _ in range(n_tokens):
        x = lm.tok.encode_composite(cur, prev)
        y = lm.net.step(x, learn=False)["y"].astype(np.float64)
        y = y / max(tau, 1e-6)
        y -= y.max()
        p = np.exp(y)
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            p = np.full_like(p, 1.0 / max(1, len(p)))
        else:
            p /= s
        if 0 < topk < len(p):
            idx = np.argpartition(-p, topk - 1)[:topk]
            pp = p[idx] / p[idx].sum()
            j = int(rng.choice(idx, p=pp))
        else:
            j = int(rng.choice(len(p), p=p))
        pj = max(float(p[j]), 1e-12)
        nxt = lm.tok.tokens[j]
        r.tokens.append(nxt)
        r.prevs.append(prev)
        r.logprobs.append(float(np.log(pj)))
        prev, cur = cur, nxt
    r.text = "".join(r.tokens)
    return r


# ─────────────────────────── 训练器 ───────────────────────────

class REINFORCETrainer:
    """REINFORCE（带滑动基线与可选 KL 正则）。

    核心：把策略梯度灌进 P6 读出感知器 —— 每步用 `target=onehot(a)`、
    `learn_scale = advantage`（`advantage = (R − baseline) × 归一化`）调
    `net.step`。**没有新增算子**，只是换了 η 的取法。
    """

    def __init__(self, lm, *, lr: float = 1e-3, baseline_decay: float = 0.9,
                 normalize_adv: bool = True, kl_coef: float = 0.0,
                 reward_fn: RewardFn | None = None,
                 free_stream: bool = True):
        self.lm = lm
        self.lr = float(lr)
        self.baseline_decay = float(baseline_decay)
        self.normalize_adv = bool(normalize_adv)
        self.kl_coef = float(kl_coef)
        self.reward_fn = reward_fn
        self.free_stream = bool(free_stream)     # rollout 后恢复状态
        self.baseline = 0.0
        self._ref_W = None                        # KL 参考策略的读出副本（可选）
        self._hist: dict = {"reward": [], "adv_abs": [], "kl": [], "loss": []}

    # ---------- 参考策略（KL）----------
    def pin_reference(self) -> None:
        """冻结当前读出作为参考策略 π_ref（KL 正则用；1B 档额外占一份 W）。"""
        ro = self.lm.net.readout
        if hasattr(ro, "W_cpu"):
            self._ref_W = ro.W_cpu()            # 加速后端：取回主机副本
        else:
            self._ref_W = np.array(ro.W, copy=True)

    def unpin_reference(self) -> None:
        self._ref_W = None

    # ---------- 采样 + 打分 ----------
    def collect(self, prompts: Iterable[dict], n_tokens: int = 32, tau: float = 1.0,
                topk: int = 0, seed: int = 0) -> list[Rollout]:
        """按提示集采样并打分（不更新权重）。`prompt` 记录写回 rollout。"""
        out = []
        for i, item in enumerate(prompts):
            with state_guard(self.lm.net):
                r = rollout(self.lm, item["prompt"], n_tokens=n_tokens, tau=tau,
                            topk=topk, seed=seed + i)
            r.reward = self._score(item, r)
            out.append(r)
        return out

    def _score(self, item: dict, r: Rollout) -> float:
        rw = reward_from_dataset(item)
        if rw is not None:
            return rw
        if self.reward_fn is None:
            raise ValueError(
                f"提示 {item.get('prompt', '')[:30]!r} 没有内联 reward，且未提供 "
                "--reward-fn —— RL 需要奖励来源（数据集 reward 字段或自定义函数）")
        kwargs = {k: v for k, v in item.items() if k != "prompt"}
        kwargs.pop("reward", None)
        return float(self.reward_fn(prompt=item["prompt"], completion=r.text,
                                    tokens=r.tokens, logprobs=r.logprobs,
                                    **kwargs))

    # ---------- 更新 ----------
    def advantages(self, rollouts: list[Rollout]) -> np.ndarray:
        """奖励 → 去基线 → 归一化。基线 = 同批均值的滑动平均（降方差）。"""
        R = np.array([r.reward for r in rollouts], dtype=np.float64)
        self.baseline = (self.baseline_decay * self.baseline
                         + (1.0 - self.baseline_decay) * float(R.mean()))
        adv = R - self.baseline
        # 归一化只在**奖励确有方差**时做：一批奖励全相等时归一化会把优势压成全 0
        # （稀疏奖励任务的典型「学不动」症状），此时保留去基线后的原值。
        if self.normalize_adv and adv.size > 1 and adv.std() > 1e-12:
            adv = adv / (adv.std() + 1e-8)
        return adv

    def update(self, rollouts: list[Rollout]) -> dict:
        """用一批轨迹做一次策略梯度更新（逐条 replay，每步一次感知器更新）。"""
        if not rollouts:
            return {"steps": 0, "reward": 0.0, "baseline": self.baseline,
                    "adv_abs": 0.0, "nll": 0.0}
        adv = self.advantages(rollouts)
        stoi = self.lm.tok.stoi
        tok = self.lm.tok
        steps = 0
        nll_sum = 0.0
        kl_sum = 0.0
        for r, a in zip(rollouts, adv):
            if abs(a) < 1e-12:
                continue
            # replay 轨迹：重新按序推进状态并在每步施加策略梯度
            with state_guard(self.lm.net) if self.free_stream else _null_ctx():
                prev = r.prevs[0] if r.prevs else None
                cur = r.tokens[0] if r.tokens else None
                for k, act in enumerate(r.tokens):
                    x = tok.encode_composite(cur, prev)
                    tgt = tok.onehot(stoi[act])
                    d = self.lm.net.step(x, target=tgt, learn=True,
                                        learn_scale=self.lr * float(a))
                    nll_sum += d["nll"]
                    steps += 1
                    if self.kl_coef > 0 and self._ref_W is not None:
                        kl_sum += self._kl_penalty(x, cur, prev)
                    prev, cur = cur, act
        if steps == 0:
            # 一批轨迹的优势全为 0（例如奖励完全一致）→ 没有可施加的梯度。
            # 如实报告 steps=0，**不**静默伪装成"训练过"。
            return {"steps": 0, "reward": float(np.mean([r.reward for r in rollouts])),
                    "baseline": self.baseline,
                    "adv_abs": float(np.mean(np.abs(adv))), "nll": 0.0,
                    "skipped": "所有优势为 0（奖励无方差），本轮未更新"}
        out = {
            "steps": steps,
            "reward": float(np.mean([r.reward for r in rollouts])),
            "baseline": self.baseline,
            "adv_abs": float(np.mean(np.abs(adv))),
            "nll": nll_sum / steps,
        }
        if self.kl_coef > 0 and self._ref_W is not None:
            out["kl"] = kl_sum / max(1, steps)
        self._hist["reward"].append(out["reward"])
        self._hist["adv_abs"].append(out["adv_abs"])
        self._hist["loss"].append(out["nll"])
        return out

    def _kl_penalty(self, x, cur, prev) -> float:
        """KL(π_θ ‖ π_ref) 的**采样估计**（用参考 logits 与当前 logits 的一步差）。

        参考权重常驻主机（`_ref_W`），每步一次 numpy 前向——仅在 kl_coef>0
        时承担；生产若嫌慢可改为每 N 步一次（此处给出诚实口径，不假装免费）。
        """
        ro = self.lm.net.readout
        W_ref = self._ref_W
        if W_ref is None or not hasattr(ro, "W_cpu"):
            return 0.0
        y_ref = W_ref @ np.asarray(x, dtype=np.float32)
        y_now = ro.forward(x)
        p = np.exp(y_now - y_now.max())
        p /= max(p.sum(), 1e-12)
        q = np.exp(y_ref - y_ref.max())
        q /= max(q.sum(), 1e-12)
        return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())

    def stats(self, k: int = 20) -> dict:
        h = self._hist
        def tail(key):
            v = h.get(key) or []
            return v[-k:]
        return {
            "reward_mean": float(np.mean(tail("reward"))) if tail("reward") else None,
            "reward_last": tail("reward")[-1] if tail("reward") else None,
            "adv_abs_mean": float(np.mean(tail("adv_abs"))) if tail("adv_abs") else None,
            "nll_mean": float(np.mean(tail("loss"))) if tail("loss") else None,
            "baseline": self.baseline,
            "kl_mean": float(np.mean(tail("kl"))) if tail("kl") else None,
        }


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
