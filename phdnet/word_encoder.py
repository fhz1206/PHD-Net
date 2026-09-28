"""M7 组合编码器 —— 词涌现（无词典统计分词）与词级 SDR 编码。

脑对应：视觉词形区（VWFA）的组合调谐；词表由语料统计自动涌现，
是 M3 STDP 共现绑定的工程等价物。T2.3 上下文绑定编码亦在此（双库拼接）。"""

import numpy as np

from .tokenizer import U64, _mix64


def _induce_length(codes, inv_codes, n: int, L: int, min_count: int,
                   min_entropy: float) -> set[str]:
    """单层 L-gram 词涌现（__init__ 主循环的 per-L 体，供串行/多核共用）。

    输入为**全文级**统计量（字符表 codes、位置编码 inv_codes、全文长度 n），
    输出该 L 接受的词集合。与串行版逐位一致：同一 numpy 运算序列，
    只是执行位置不同——各 L 相互独立，因此可安全并行（train_1b/vocab_parallel.py）。
    """
    out: set[str] = set()
    B = int(codes.shape[0])
    codes_list = [c for c in codes]              # list[str]，单个字符，id→字符反查
    m = n - L + 1
    if m <= 0:
        return out
    # 滑窗矩阵 W[i, j] = 第 i 个窗口第 j 个位置的字符 id
    W = np.empty((m, L), dtype=np.int64)
    for j in range(L):
        W[:, j] = inv_codes[j:j + m]
    uniq, inv, counts = np.unique(W, axis=0, return_inverse=True,
                                 return_counts=True)
    K = int(uniq.shape[0])
    cand = np.where(counts >= min_count)[0]
    if cand.size == 0:
        return out
    Kc = int(cand.shape[0])
    # 候选 → 紧凑局部 id（仅候选行参与后续聚合）
    local = np.full(K, -1, dtype=np.int64)
    local[cand] = np.arange(Kc, dtype=np.int64)

    # 左邻：窗口位置 i 有左邻 iff i > 0（左邻字符 = codes[i-1]）
    posL = np.arange(1, m)
    keep = local[inv[posL]] >= 0
    posL = posL[keep]
    cl = local[inv[posL]]
    lc = inv_codes[posL - 1]
    idxL = cl * B + lc
    left_counts = np.bincount(idxL, minlength=Kc * B).reshape(Kc, B)

    # 右邻：窗口位置 i 有右邻 iff i + L < n（右邻字符 = codes[i+L]）
    posR = np.arange(0, m - 1)
    keep = local[inv[posR]] >= 0
    posR = posR[keep]
    cr = local[inv[posR]]
    rc = inv_codes[posR + L]
    idxR = cr * B + rc
    right_counts = np.bincount(idxR, minlength=Kc * B).reshape(Kc, B)

    for k in range(Kc):
        c = int(cand[k])
        total = int(counts[c])
        if total < min_count:
            continue
        # 左邻熵（空计数 → 0.0，与原 _entropy 语义一致）
        lv = left_counts[k]
        lt = int(lv.sum())
        if lt > 0:
            nz = lv[lv > 0].astype(np.float64)
            p = nz / lt
            h_l = -float(np.sum(p * np.log(p)))
        else:
            h_l = 0.0
        # 右邻熵
        rv = right_counts[k]
        rt = int(rv.sum())
        if rt > 0:
            nz = rv[rv > 0].astype(np.float64)
            p = nz / rt
            h_r = -float(np.sum(p * np.log(p)))
        else:
            h_r = 0.0
        if min(h_l, h_r) >= min_entropy:
            ids = [int(d) for d in uniq[c]]
            w = ''.join(codes_list[i] for i in ids)
            out.add(w)
    return out


class WordSegmenter:
    """T2.1 词涌现：无词典的统计分词。

    判据（Zhao & Kit 式边界熵的简化版）：
      候选 = 频次 ≥ min_count 的 n-gram（2..max_len 字）；
      接受条件 = 左邻熵与右邻熵均 ≥ min_entropy——
      词的两侧可以接/被接多种字符（高熵），词内部的续接高度确定（低熵）。
    脑对应：STDP 迹对高频共现序列的因果绑定（工程实现为统计涌现）。

    实现（O4 向量化改造）：
      原实现在「每个候选扫描全文求位置」上是 O(候选×n) 纯 Python，是主要热点。
      现改用 numpy 向量化，且输出（vocab 集合 / tokenize 结果）与原实现逐位一致：
        - 将 text 编码为字符 id 序列 codes（np.unique(..., return_inverse=True)）；
        - 对每个 L，用 (m, L) 整数滑窗矩阵 + np.unique(axis=0) 一次得到每个
          L-gram 的频次与「位置→候选」归属 inv（避免 B-radix 在 max_len 较大时
          的 int64 溢出，对任意 max_len 安全）；
        - 用 np.bincount 在「候选 × 字符表」展平索引上批量累加左/右邻字符计数；
        - 自然对数 Shannon 熵按候选向量化计算，阈值比较用掩码。
      2026-09-25 重构：per-L 体抽为模块级 `_induce_length`（串行/多核共用一份
      实现，各 L 相互独立可安全并行）；串行路径运算序列逐位不变。
      详见 ci/verifiers/verify_seg_equiv.py 的对拍验证。
    """

    def __init__(self, text: str, max_len: int = 6, min_count: int = 5,
                 min_entropy: float = 1.0):
        n = len(text)
        self.max_len = max_len                       # B1 修复：保存以备 tokenize 使用，避免与分词硬编码不一致
        self.vocab: set[str] = set()
        if n < 2 or max_len < 2:
            return
        chars_list = list(text)
        codes, inv_codes = np.unique(chars_list, return_inverse=True)
        inv_codes = inv_codes.astype(np.int64)

        for L in range(2, max_len + 1):
            self.vocab |= _induce_length(codes, inv_codes, n, L,
                                         min_count, min_entropy)

    def tokenize(self, text: str) -> list[str]:
        """贪心最长匹配；词表外字符回退为单字符 token。"""
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            for L in range(min(self.max_len, n - i), 1, -1):   # B1 修复：用 self.max_len（默认 6 逐位不变）
                if text[i:i + L] in self.vocab:
                    out.append(text[i:i + L])
                    i += L
                    break
            else:
                out.append(text[i])
                i += 1
        return out


class WordTokenizer:
    """词级分词器：涌现词表 + token → SDR 确定性哈希（与 CharTokenizer 同构）。"""

    def __init__(self, vocab_text: str, n_sdr: int = 256, n_active: int = 32,
                 seed: int = 0, seg_kwargs: dict | None = None):
        self.seg = WordSegmenter(vocab_text, **(seg_kwargs or {}))
        self.tokens = sorted(set(self.seg.tokenize(vocab_text)))
        self.stoi = {t: i for i, t in enumerate(self.tokens)}
        self.n_sdr = n_sdr
        self.n_active = n_active
        self.seed = seed
        self._sdrs: dict[str, np.ndarray] = {}
        for i, t in enumerate(self.tokens):
            x = U64(i) * U64(2654435761) + np.arange(n_active, dtype=np.uint64)
            idx = (_mix64(x + U64(seed)) % U64(n_sdr)).astype(np.int64)
            s = np.zeros(n_sdr)
            s[idx] = 1.0
            self._sdrs[t] = s

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, tok: str) -> np.ndarray:
        return self._sdrs[tok]

    def encode_composite(self, tok: str, prev: str | None) -> np.ndarray:
        """T2.3 上下文绑定：[当前词库 ; 前词库] 双库拼接（各 n_sdr 维）。

        OOV 安全（2026-09-28 修复）：prev 不在词表时按 None 退化（无前词上下文），
        与 train_1b/train.py 的 p2 OOV 修复同语义；tok 自身的 OOV 由调用方守卫
        （word_lm._pass / evaluate 均为「任一端 OOV 整步跳过」）。
        """
        s = np.concatenate([self._sdrs[tok], np.zeros(self.n_sdr)])
        if prev is not None and prev in self._sdrs:
            s[self.n_sdr:] = self._sdrs[prev]
        return s

    def onehot(self, idx: int) -> np.ndarray:
        t = np.zeros(len(self.tokens))
        t[idx] = 1.0
        return t
